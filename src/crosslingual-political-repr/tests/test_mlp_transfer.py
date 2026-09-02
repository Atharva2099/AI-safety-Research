import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import compute_mlp_transfer_matrices as mlp  # noqa: E402


def toy_data(n_groups=10, in_dim=3):
    groups = np.repeat(np.arange(n_groups), 2)
    labels = np.tile([0, 1], n_groups)
    base = np.arange(len(groups) * in_dim, dtype=np.float32).reshape(len(groups), in_dim)
    features = {lang: base + offset for offset, lang in enumerate(mlp.LANGUAGES)}
    labels_by_lang = {lang: labels.copy() for lang in mlp.LANGUAGES}
    folds = list(GroupKFold(n_splits=2).split(base, labels, groups))
    return features, labels_by_lang, folds


def test_one_frozen_probe_is_reused_for_all_targets(monkeypatch):
    features, labels, folds = toy_data()
    fit_calls = []
    score_calls = []

    class FrozenPart:
        def __init__(self):
            self.state = {"value": 1}

    def fake_fit(*args):
        probe = mlp.FittedProbe(FrozenPart(), FrozenPart())
        fit_calls.append(probe)
        return probe, {"best_epoch": 1}

    def fake_metrics(probe, X, y, device):
        # Retain each slice so its identity cannot be recycled by Python.
        score_calls.append((id(probe), X, probe.model.state["value"], probe.scaler.state["value"]))
        return {"loss": 0.5, "accuracy": 0.5, "f1": 0.5}

    monkeypatch.setattr(mlp, "_metrics", fake_metrics)
    result = mlp.evaluate_mlp_matrix(
        features, labels, folds, torch.device("cpu"), seeds=(0, 1), progress=False,
        fit_fn=fake_fit,
    )

    expected_fits = len(mlp.LANGUAGES) * len(folds) * 2
    assert len(fit_calls) == expected_fits
    assert len(score_calls) == expected_fits * 6
    for start in range(0, len(score_calls), len(mlp.LANGUAGES)):
        batch = score_calls[start:start + len(mlp.LANGUAGES)]
        assert len({call[0] for call in batch}) == 1
        assert len({id(call[1]) for call in batch}) == 6
        assert {(call[2], call[3]) for call in batch} == {(1, 1)}
    assert result["training_count"] == len(fit_calls)


def test_control_reports_one_effective_initialization_seed(monkeypatch):
    features, labels, folds = toy_data()
    monkeypatch.setattr(
        mlp, "_metrics",
        lambda probe, X, y, device: {"loss": 0.5, "accuracy": 0.5, "f1": 0.5},
    )
    result = mlp.evaluate_mlp_matrix(
        features, labels, folds, torch.device("cpu"), seeds=(0, 1), progress=False,
        shuffled_labels=True, control_seed=91,
        fit_fn=lambda *args: (mlp.FittedProbe(object(), object()), {"best_epoch": 1}),
    )
    assert result["seed_list"] == [91]
    assert result["initialization_seeds"] == [91]
    assert result["initialization_seed_count"] == 1
    assert result["label_shuffle_seed"] == 91
    assert result["training_count"] == len(mlp.LANGUAGES) * len(folds)


def test_scaler_uses_source_outer_train_only():
    features, labels, folds = toy_data()
    train_idx, test_idx = folds[0]
    device = torch.device("cpu")

    probe, _ = mlp.fit_probe(
        features["en"], labels["en"], np.repeat(np.arange(10), 2), train_idx, 0, 0,
        device, max_epochs=3, patience=2,
    )
    expected_mean = features["en"][train_idx].mean(axis=0)
    assert np.allclose(probe.scaler.mean_, expected_mean)
    assert not np.allclose(probe.scaler.mean_, features["mr"][test_idx].mean(axis=0))


def test_inner_split_is_group_disjoint_and_keeps_pairs():
    groups = np.repeat(np.arange(10), 2)
    outer_train = np.arange(12)
    inner_train, inner_validation = mlp.grouped_inner_split(outer_train, groups, 0)
    assert set(inner_train).isdisjoint(inner_validation)
    assert set(groups[inner_train]).isdisjoint(groups[inner_validation])
    for split in (inner_train, inner_validation):
        assert all(np.sum(groups[split] == group) == 2 for group in np.unique(groups[split]))


def test_fixed_seed_fit_is_reproducible():
    features, labels, folds = toy_data()
    kwargs = dict(
        X_outer_train=features["en"], y_outer_train=labels["en"],
        groups=np.repeat(np.arange(10), 2), outer_train_idx=folds[0][0],
        outer_fold=0, seed=7, device=torch.device("cpu"), max_epochs=4, patience=2,
    )
    first, first_meta = mlp.fit_probe(**kwargs)
    second, second_meta = mlp.fit_probe(**kwargs)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(left, right)
    assert first_meta == second_meta
    assert 0.0 <= first_meta["source_training"]["accuracy"] <= 1.0
    assert 0.0 <= first_meta["source_validation"]["accuracy"] <= 1.0


def test_matrix_schema_and_parameter_count(monkeypatch):
    features, labels, folds = toy_data()
    monkeypatch.setattr(
        mlp, "_metrics",
        lambda probe, X, y, device: {"loss": 0.5, "accuracy": 0.5, "f1": 0.5},
    )
    result = mlp.evaluate_mlp_matrix(
        features, labels, folds, torch.device("cpu"), seeds=(0,), progress=False,
        fit_fn=lambda *args: (mlp.FittedProbe(object(), object()), {"best_epoch": 1}),
    )
    assert len(result["matrix"]) == len(mlp.LANGUAGES)
    assert all(len(row) == len(mlp.LANGUAGES) for row in result["matrix"])
    assert mlp.parameter_count(4096) == 32785

    record = mlp.build_record("toy/model", 3, features, folds, result)
    assert record["probe_type"] == "one_hidden_layer_mlp"
    assert record["architecture"]["hidden_dim"] == 8
    assert record["architecture"]["preprocessing"].startswith("inner scaler")
    assert record["training"]["seeds"] == list(mlp.SEEDS)
    assert record["training"]["outer_folds"] == len(folds)
    assert record["training"]["patience"] == mlp.PATIENCE
    assert record["training"]["inner_split_seed_rule"] == "NumPy default_rng(10000 + outer_fold)"
    assert record["training"]["improvement_rule"] == "strict improvement with min_delta=0.0"
    assert "selected epoch count" in record["training"]["final_refit"]
    assert "epoch-selection model" in record["training"]["metric_semantics"]["source_validation"]
    assert "final fresh-refit model" in record["training"]["metric_semantics"]["source_training"]

    ids = np.repeat(np.arange(10), 2)
    expected = folds
    assert record["outer_splitter"]["folds"] == [
        {"train_indices": train.tolist(), "test_indices": test.tolist()}
        for train, test in expected
    ]


def test_aggregation_units_are_explicit():
    entries = []
    for seed in (0, 1):
        for fold in (0, 1):
            entries.append({"seed": seed, "outer_fold": fold, "accuracy": seed + fold})
    cells = {source: {target: entries for target in mlp.LANGUAGES} for source in mlp.LANGUAGES}
    summary = mlp.summarize_accuracy(cells, (0, 1), 2)
    assert summary["pooled_fold_seed_accuracy_sd"][0][0] == np.std([0, 1, 1, 2])
    assert summary["seed_level_accuracy_sd"][0][0] == np.std([0.5, 1.5])
    assert summary["fold_level_accuracy_sd"][0][0] == np.std([0.5, 1.5])


def test_nonfinite_training_features_raise():
    features, labels, folds = toy_data()
    bad = features["en"].copy()
    bad[0, 0] = np.nan
    with np.testing.assert_raises(ValueError):
        mlp.fit_probe(
            bad, labels["en"], np.repeat(np.arange(10), 2), folds[0][0], 0, 0,
            torch.device("cpu"), max_epochs=2,
        )


def test_nonfinite_training_loss_raises(monkeypatch):
    class NaNLoss:
        def __call__(self, logits, labels):
            return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(mlp.nn, "BCEWithLogitsLoss", lambda: NaNLoss())
    model = mlp.MLPProbe(2)
    with np.testing.assert_raises(FloatingPointError):
        mlp._fit_epochs(
            model, np.ones((2, 2), dtype=np.float32), np.array([0, 1], dtype=np.float32),
            torch.device("cpu"), 1, 1e-3, 1e-2,
        )
