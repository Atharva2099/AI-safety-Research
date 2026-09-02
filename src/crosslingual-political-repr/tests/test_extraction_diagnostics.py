import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import run_extraction_diagnostics as diag  # noqa: E402


class TinyTokenizer:
    all_special_ids = [99, 100]
    unk_token_id = 98
    pad_token_id = 0
    model_max_length = 128

    def __call__(self, text, add_special_tokens=True, truncation=False):
        ids = [ord(c) % 20 + 1 for c in text]
        if add_special_tokens:
            ids = [99] + ids + [100]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def convert_ids_to_tokens(self, ids):
        return ["<s>" if i == 99 else "</s>" if i == 100 else ("." if i == 2 else "x") for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(t for t in tokens if not (t.startswith("<") and t.endswith(">")))

    def pad(self, values, padding=True, return_tensors="pt"):
        max_len = max(len(x["input_ids"]) for x in values)
        ids = [x["input_ids"] + [self.pad_token_id] * (max_len - len(x["input_ids"])) for x in values]
        mask = [[1] * len(x["input_ids"]) + [0] * (max_len - len(x["input_ids"])) for x in values]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class TinyModel:
    def __call__(self, input_ids, attention_mask, output_hidden_states=True):
        # Four hidden-state entries, each vector is a simple deterministic
        # function of token ID and position.
        batch, length = input_ids.shape
        base = input_ids.float().unsqueeze(-1) + torch.arange(3).float()
        states = tuple(base + layer for layer in range(4))
        return type("Output", (), {"hidden_states": states})


class RecordingModel(TinyModel):
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, input_ids, attention_mask, output_hidden_states=True):
        self.batch_sizes.append(int(input_ids.shape[0]))
        return super().__call__(input_ids, attention_mask, output_hidden_states)


def test_exact_legacy_extraction_parity_and_padding_exclusion():
    tok = TinyTokenizer()
    batch = tok.pad([tok("ab"), tok("abcdef")])
    hidden = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(2, 8, 2)
    positions = batch["attention_mask"].sum(dim=1) - 1
    expected = hidden[torch.arange(2), positions]
    assert torch.equal(expected[0], hidden[0, 3])
    assert torch.equal(expected[1], hidden[1, 7])
    assert int(positions[0]) != 7


def test_special_token_handling_and_nonempty_content_masks():
    result = diag.tokenize_diagnostics(TinyTokenizer(), "abc")
    assert result["special_count"] == 2
    assert result["content_positions"]
    assert 0 not in result["content_positions"]


@pytest.mark.parametrize("text,expected", [
    ("नमस्ते।", "नमस्ते"),
    ("मराठी.", "मराठी"),
    ("Time to act!", "Time to act"),
    ("Question?", "Question"),
    ("look...", "look"),
    ("trailing space. ", "trailing space"),
    ("no punctuation", "no punctuation"),
])
def test_unicode_terminal_punctuation_handling(text, expected):
    assert diag._strip_terminal_punctuation(text) == expected


def test_select_c_tie_resolves_to_smallest_candidate(monkeypatch):
    calls = []

    def fake_score(self, X, y):
        calls.append(1)
        return 0.9

    monkeypatch.setattr(diag.LogisticRegression, "score", fake_score)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = np.tile([0, 1], 10)
    ids = np.repeat(np.arange(10), 2)
    chosen, inner_fits = diag.select_c(X, y, ids, np.arange(20), candidates=[10, 0.1, 0.0001, 1, 0.01])
    assert chosen == 0.0001
    assert len(inner_fits) == 5 * 3


def test_probe_fits_pin_solver_iterations_and_record_convergence():
    features, labels, ids = _toy_features()
    records, fits = diag.probe_records(features, labels, ids, lane="legacy",
                                       source_language="en", conditions=("current_raw",))
    assert len(fits) == 5
    assert all(f["selected_C"] == 1.0 for f in fits)
    assert all(f["n_iter"] <= diag.MAX_ITER for f in fits)
    assert all(isinstance(f["converged"], bool) for f in fits)


def test_mean_pooling_computation_and_l2_norm():
    tok, model = TinyTokenizer(), TinyModel()
    values, _ = diag.extract_conditions(model, tok, ["abc"], layer=1)
    raw = values["content_raw"][0]
    mean = values["mean_raw"][0]
    assert np.allclose(mean, raw) is False
    assert np.isclose(np.linalg.norm(values["current_l2"][0]), 1)
    assert np.isclose(np.linalg.norm(values["mean_l2"][0]), 1)


def test_extraction_batches_without_changing_statement_order():
    model = RecordingModel()
    values, _ = diag.extract_conditions(model, TinyTokenizer(), ["a", "bb", "ccc", "dddd", "fffff"], layer=1, batch_size=2)
    assert model.batch_sizes == [2, 2, 2, 2, 1, 1]
    assert values["current_raw"].shape[0] == 5
    assert values["content_raw"][0, 0] != values["content_raw"][1, 0]


def test_zero_vector_assertion():
    with pytest.raises(AssertionError):
        diag.l2_normalize(np.zeros(3))


def test_no_statement_truncation():
    with pytest.raises(ValueError):
        diag.tokenize_diagnostics(TinyTokenizer(), "abcdef", max_length=3)


def test_pair_grouping_in_five_outer_splits():
    ids = np.repeat(np.arange(10), 2)
    folds = diag.grouped_folds(ids)
    assert len(folds) == 5
    for train, test in folds:
        assert set(ids[train]).isdisjoint(ids[test])


def _toy_features():
    ids = np.repeat(np.arange(10), 2)
    labels = np.tile([0, 1], 10)
    base = np.arange(20 * 3, dtype=np.float32).reshape(20, 3) + labels[:, None]
    features = {language: {c: base + n for c in diag.CONDITIONS} for n, language in enumerate(diag.LANGUAGES)}
    labels_by_language = {language: labels.copy() for language in diag.LANGUAGES}
    ids_by_language = {language: ids.copy() for language in diag.LANGUAGES}
    return features, labels_by_language, ids_by_language


def test_one_fit_reused_across_all_six_targets_and_records_are_complete():
    features, labels, ids = _toy_features()
    records, fits = diag.probe_records(features, labels, ids, lane="legacy", source_language="en")
    assert len(records) == 20 * 6  # strictly 1 OOF prediction per row for all 6 targets
    assert len({r["fit_id"] for r in records}) == 5
    assert len(fits) == 5
    assert all({"truth", "pred", "proba", "decision_score", "fold", "selected_C", "fit_id", "row_index", "question_id"} <= r.keys() for r in records)


def test_legacy_lane_covers_only_current_raw():
    features, labels, ids = _toy_features()
    records, fits = diag.probe_records(features, labels, ids, lane="legacy", source_language="en")
    assert {r["condition"] for r in records} == {"current_raw"}


def test_oof_join_selects_exactly_one_prediction_per_row():
    features, labels, ids_by_lang = _toy_features()
    records, _ = diag.probe_records(features, labels, ids_by_lang, lane="legacy", source_language="en")
    target_ids = ids_by_lang["en"]
    en_records = [r for r in records if r["target_language"] == "en"]
    assert len(en_records) == len(target_ids)
    assert sorted(r["row_index"] for r in en_records) == list(range(len(target_ids)))
    assert [r["question_id"] for r in sorted(en_records, key=lambda x: x["row_index"])] == list(target_ids)


def test_layer_pin_rejects_mismatched_layer():
    with pytest.raises(ValueError):
        diag.validate_layer("allenai/Olmo-3-7B-Instruct", 5)
    assert diag.validate_layer("allenai/Olmo-3-7B-Instruct", 17) is None


def test_qwen_layer_pin_uses_earliest_english_peak():
    with pytest.raises(ValueError):
        diag.validate_layer("Qwen/Qwen3.5-9B", 14)
    assert diag.validate_layer("Qwen/Qwen3.5-9B", 12) is None


def test_finite_checks_and_norm_only_controls():
    features, labels, ids = _toy_features()
    records, fits = diag.probe_records(features, labels, ids, lane="fair", source_language="en", norm_only=True)
    assert records and all(np.isfinite([r["proba"], r["decision_score"]]).all() for r in records)
    assert fits and len(fits) == 5 * 4  # 5 folds x 4 norm_only conditions


def test_text_control_leakage_isolation(monkeypatch):
    import run_text_surface_controls as controls
    data = {"en": [{"question_id": i, "statement": ("a" if i % 2 else "b") * 5, "label": i % 2} for i in range(10)]}
    seen = []
    original = controls.TfidfVectorizer.fit_transform

    def fit(self, raw_documents, y=None):
        seen.append(tuple(raw_documents))
        return original(self, raw_documents, y)

    monkeypatch.setattr(controls.TfidfVectorizer, "fit_transform", fit)
    controls.grouped_control_predictions(data, n_splits=5)
    assert seen
    assert all(len(set(documents)) < 10 for documents in seen)


def test_crosslingual_ngram_is_source_fit_and_auditable():
    import run_text_surface_controls as controls

    data = {
        "source": [
            {"question_id": qid, "statement": text, "label": label}
            for qid in range(10)
            for text, label in (("aaaaaa", 0), ("bbbbbb", 1))
        ],
        "target": [
            {"question_id": qid, "statement": text, "label": label}
            for qid in range(10)
            for text, label in (("yyyyyy", 0), ("zzzzzz", 1))
        ],
    }
    result = controls.crosslingual_char_ngram_results(data, n_splits=5)

    assert result["software"]["scikit_learn"]
    assert len(result["predictions"]) == 2 * 2 * 20
    assert len(result["fits"]) == 2 * 5
    assert np.asarray(result["matrix"]).shape == (2, 2)
    for fit in result["fits"]:
        assert len(fit["feature_names"]) == len(fit["idf"]) == len(fit["coefficients"])
        assert len(fit["train_question_ids"]) == 8
        assert len(fit["test_question_ids"]) == 2
    source_fits = [fit for fit in result["fits"] if fit["source_language"] == "source"]
    assert all(fit["target_overlap"]["target"]["empty_rows"] == 4 for fit in source_fits)
    assert all(not any("y" in name or "z" in name for name in fit["feature_names"]) for fit in source_fits)
