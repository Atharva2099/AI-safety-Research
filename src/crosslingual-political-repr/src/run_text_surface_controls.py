"""Grouped, out-of-fold tests for surface-text political-label cues.

This script deliberately fits every text transform inside the training fold.  In
particular, the TF-IDF vocabulary and IDF statistics never see an outer test
statement.
"""

import argparse
import gzip
import json
import unicodedata
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
PUNCTUATION = set(".!?。！？॥")


def load_data(path: Path) -> dict[str, list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    by_id = {}
    for record in records:
        if record.get("polarity") in (-1, 1):
            by_id.setdefault(record["id"], []).append(record)
    if len(by_id) != 580 or any(len(v) != 2 for v in by_id.values()):
        raise ValueError("Expected 580 question IDs with two records each")
    if any({r["polarity"] for r in v} != {-1, 1} for v in by_id.values()):
        raise ValueError("Every question must contain one record of each polarity")
    result = {}
    for language in LANGUAGES:
        result[language] = [
            {"question_id": qid, "statement": row[language], "label": int(row["polarity"] == 1)}
            for qid, pair in by_id.items() for row in sorted(pair, key=lambda x: x["polarity"])
        ]
    ids = [x["question_id"] for x in result["en"]]
    if any([x["question_id"] for x in result[l]] != ids for l in LANGUAGES):
        raise ValueError("Language records are not aligned")
    return result


def _surface_features(texts, tokenizer=None) -> np.ndarray:
    rows = []
    for text in texts:
        chars = len(text)
        segments = max(1, len(text.split()))
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if tokenizer else segments
        unknowns = 0
        byte_fallback = 0
        if tokenizer:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            unknown_id = getattr(tokenizer, "unk_token_id", None)
            unknowns = sum(i == unknown_id for i in ids) if unknown_id is not None else 0
            byte_fallback = sum("byte" in str(t).lower() or str(t).startswith("<0x")
                                for t in tokenizer.convert_ids_to_tokens(ids))
        final = text.rstrip()[-1:] or ""
        rows.append([chars, segments, tokens, tokens / max(chars, 1),
                     unknowns, byte_fallback, int(final in PUNCTUATION),
                     int(unicodedata.category(final).startswith("P"))])
    return np.asarray(rows, dtype=np.float64)


def grouped_control_predictions(data, tokenizer=None, n_splits=5, seed=42):
    """Return records for three controls: character n-grams, surface counts, terminal ID."""
    outputs = []
    for language, items in data.items():
        texts = [x["statement"] for x in items]
        y = np.asarray([x["label"] for x in items])
        groups = np.asarray([x["question_id"] for x in items])
        folds = list(GroupKFold(n_splits=n_splits).split(texts, y, groups))
        controls = {
            "char_ngram": lambda: make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5)),
                LogisticRegression(C=1.0, max_iter=1000)),
            "length_punctuation": lambda: LogisticRegression(C=1.0, max_iter=1000),
            "terminal_token": lambda: LogisticRegression(C=1.0, max_iter=1000),
        }
        for name, factory in controls.items():
            predictions = {}
            for fold, (train, test) in enumerate(folds):
                if name == "char_ngram":
                    X_train, X_test = [texts[i] for i in train], [texts[i] for i in test]
                elif name == "length_punctuation":
                    X_train, X_test = _surface_features([texts[i] for i in train], tokenizer), _surface_features([texts[i] for i in test], tokenizer)
                else:
                    if tokenizer is None:
                        # This fallback keeps the utility usable without a
                        # model; production runs should pass the target tokenizer.
                        token_id = lambda t: ord(t.rstrip()[-1])
                    else:
                        token_id = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"][-1]
                    X_train = np.asarray([[token_id(t)] for t in [texts[i] for i in train]])
                    X_test = np.asarray([[token_id(t)] for t in [texts[i] for i in test]])
                model = factory().fit(X_train, y[train])
                values = model.predict(X_test)
                scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
                for i, pred, score in zip(test, values, scores):
                    predictions[i] = {"truth": int(y[i]), "pred": int(pred), "score": float(score), "fold": fold}
            outputs.extend({"language": language, "control": name, "question_id": item["question_id"], **predictions[i]}
                           for i, item in enumerate(items))
    return outputs


def crosslingual_char_ngram_results(data, n_splits=5):
    """Train character n-gram classifiers on each source language and transfer unchanged."""
    languages = tuple(data)
    reference_ids = [x["question_id"] for x in data[languages[0]]]
    if any([x["question_id"] for x in data[language]] != reference_ids for language in languages):
        raise ValueError("Language records are not aligned")

    predictions = []
    fits = []
    for source in languages:
        source_items = data[source]
        source_texts = [x["statement"] for x in source_items]
        source_y = np.asarray([x["label"] for x in source_items])
        groups = np.asarray([x["question_id"] for x in source_items])
        folds = GroupKFold(n_splits=n_splits).split(source_texts, source_y, groups)

        for fold, (train, test) in enumerate(folds):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
            X_train = vectorizer.fit_transform([source_texts[i] for i in train])
            classifier = LogisticRegression(C=1.0, max_iter=1000).fit(X_train, source_y[train])
            feature_names = vectorizer.get_feature_names_out()
            overlap = {}

            for target in languages:
                target_items = data[target]
                target_texts = [target_items[i]["statement"] for i in test]
                target_y = np.asarray([target_items[i]["label"] for i in test])
                X_target = vectorizer.transform(target_texts)
                values = classifier.predict(X_target)
                scores = classifier.predict_proba(X_target)[:, 1]
                nonzero = np.diff(X_target.indptr)
                overlap[target] = {
                    "rows": int(len(test)),
                    "empty_rows": int(np.sum(nonzero == 0)),
                    "nonempty_rate": float(np.mean(nonzero > 0)),
                    "mean_nonzero_features": float(np.mean(nonzero)),
                }
                predictions.extend({
                    "source_language": source,
                    "target_language": target,
                    "row_index": int(i),
                    "question_id": int(target_items[i]["question_id"]),
                    "truth": int(truth),
                    "pred": int(pred),
                    "score": float(score),
                    "fold": fold,
                } for i, truth, pred, score in zip(test, target_y, values, scores))

            fits.append({
                "source_language": source,
                "fold": fold,
                "train_question_ids": sorted({int(groups[i]) for i in train}),
                "test_question_ids": sorted({int(groups[i]) for i in test}),
                "feature_names": feature_names.tolist(),
                "idf": vectorizer.idf_.tolist(),
                "coefficients": classifier.coef_[0].tolist(),
                "intercept": float(classifier.intercept_[0]),
                "n_iter": int(classifier.n_iter_[0]),
                "target_overlap": overlap,
            })

    matrix = [
        [
            float(np.mean([
                row["pred"] == row["truth"] for row in predictions
                if row["source_language"] == source and row["target_language"] == target
            ]))
            for target in languages
        ]
        for source in languages
    ]
    matrix_array = np.asarray(matrix)
    off_diagonal = matrix_array[~np.eye(len(languages), dtype=bool)]
    return {
        "schema_version": 1,
        "languages": list(languages),
        "method": {
            "features": "character TF-IDF",
            "analyzer": "char",
            "ngram_range": [3, 5],
            "lowercase": True,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": False,
            "classifier": "LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
        },
        "software": {
            "scikit_learn": sklearn.__version__,
        },
        "cross_validation": {
            "splitter": "GroupKFold",
            "n_splits": n_splits,
            "group": "question_id",
        },
        "matrix": matrix,
        "diagonal_mean": float(np.mean(np.diag(matrix_array))),
        "off_diagonal_macro_mean": float(np.mean(off_diagonal)),
        "predictions": predictions,
        "fits": fits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "multilingual_statements.json")
    parser.add_argument("--tokenizer-model", default="allenai/Olmo-3-7B-Instruct",
                        help="Tokenizer whose final token IDs and unknown/byte statistics are measured")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "results" / "text_surface_controls.json")
    parser.add_argument(
        "--crosslingual-output",
        type=Path,
        default=ROOT / "artifacts" / "results" / "crosslingual_char_ngram_controls.json.gz",
    )
    parser.add_argument("--crosslingual-only", action="store_true")
    args = parser.parse_args()
    data = load_data(args.data)
    if not args.crosslingual_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
        records = grouped_control_predictions(data, tokenizer=tokenizer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"records": len(records), "accuracy": float(accuracy_score([r["truth"] for r in records], [r["pred"] for r in records]))}))

    crosslingual = crosslingual_char_ngram_results(data)
    import hashlib
    crosslingual["data_hash"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.crosslingual_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.crosslingual_output.with_suffix(args.crosslingual_output.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(crosslingual, handle, ensure_ascii=False)
    temporary.replace(args.crosslingual_output)
    print(json.dumps({
        "crosslingual_output": str(args.crosslingual_output),
        "predictions": len(crosslingual["predictions"]),
        "fits": len(crosslingual["fits"]),
        "diagonal_mean": crosslingual["diagonal_mean"],
        "off_diagonal_macro_mean": crosslingual["off_diagonal_macro_mean"],
    }))


if __name__ == "__main__":
    main()
