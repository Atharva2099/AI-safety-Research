"""Layerwise in-language and zero-shot cross-lingual political probes."""

import argparse
import json
import re
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(path: Path, languages: list[str]) -> dict[str, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for record in json.loads(path.read_text(encoding="utf-8")):
        if record.get("polarity") in {-1, 1}:
            grouped.setdefault(record["id"], []).append(record)
    paired = []
    for question_id, records in grouped.items():
        if len(records) != 2 or {r["polarity"] for r in records} != {-1, 1}:
            raise ValueError(f"Question {question_id} is not a strict +/-1 pair")
        if any(not record.get(language) for record in records for language in languages):
            raise ValueError(f"Question {question_id} is missing a language statement")
        paired.append((question_id, records))
    if len(paired) < 5:
        raise ValueError("At least five paired questions are required for 5-fold CV")
    return {
        language: [
            {"question_id": question_id, "statement": record[language],
             "polarity": record["polarity"]}
            for question_id, records in paired for record in sorted(records, key=lambda r: r["polarity"])
        ]
        for language in languages
    }


def canonical_folds(question_ids: list[int]) -> list[tuple[set[int], set[int]]]:
    unique_ids = list(dict.fromkeys(question_ids))
    if len(unique_ids) < 5:
        raise ValueError("At least five unique question IDs are required")
    return [
        ({unique_ids[i] for i in train}, {unique_ids[i] for i in test})
        for train, test in GroupKFold(n_splits=5).split(unique_ids, groups=unique_ids)
    ]


def extract(model, tokenizer, items: list[dict], device: torch.device) -> list[list]:
    vectors = []
    with torch.inference_mode():
        for item in items:
            inputs = tokenizer(item["statement"], return_tensors="pt", truncation=True).to(device)
            hidden = model(**inputs, output_hidden_states=True).hidden_states
            vectors.append([state[0, -1].float().cpu().numpy() for state in hidden])
    return [[vectors[row][layer] for row in range(len(items))]
            for layer in range(len(vectors[0]))]


def indices(items: list[dict], question_ids: set[int]) -> list[int]:
    return [i for i, item in enumerate(items) if item["question_id"] in question_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--data", default="data/multilingual_statements.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    if len(languages) != len(LANGUAGES) or set(languages) != set(LANGUAGES):
        raise ValueError(f"Exactly these six languages are required: {', '.join(LANGUAGES)}")
    data = load_data(Path(args.data), languages)
    question_ids = [item["question_id"] for item in data["en"]]
    folds = canonical_folds(question_ids)
    if any([item["question_id"] for item in data[language]] != question_ids
           for language in languages):
        raise ValueError("Language records are not strictly paired and aligned")

    device = choose_device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type in {"cuda", "mps"} else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    features = {language: extract(model, tokenizer, data[language], device) for language in languages}
    labels = {language: [item["polarity"] for item in data[language]] for language in languages}
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)

    for language in languages:
        output = Path(args.output_dir) / "results" / f"multilingual_probe_{model_slug}_{language}.jsonl"
        temporary = output.with_suffix(output.suffix + ".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for layer, target_features in enumerate(features[language]):
                    name = "Embedding" if layer == 0 else f"Block {layer - 1}"
                    modes = [("in_language", language)]
                    if language != "en":
                        modes.append(("zero_shot", "en"))
                    for mode, source_language in modes:
                        predictions = [None] * len(target_features)
                        sample_folds = [None] * len(target_features)
                        truth, predicted = [], []
                        for fold, (train_questions, test_questions) in enumerate(folds):
                            test = indices(data[language], test_questions)
                            train = indices(data[source_language], train_questions)
                            probe = LogisticRegression(C=1.0, max_iter=1000).fit(
                                [features[source_language][layer][i] for i in train],
                                [labels[source_language][i] for i in train])
                            values = probe.predict([target_features[i] for i in test])
                            for i, value in zip(test, values):
                                predictions[i], sample_folds[i] = int(value), fold
                            truth.extend(labels[language][i] for i in test)
                            predicted.extend(map(int, values))
                        accuracy = accuracy_score(truth, predicted)
                        f1 = f1_score(truth, predicted, pos_label=1)
                        for i, item in enumerate(data[language]):
                            handle.write(json.dumps({
                                "mode": mode, "source_language": source_language,
                                "target_language": language, "layer": name,
                                "question_id": item["question_id"], "statement": item["statement"],
                                "true_polarity": labels[language][i], "predicted_polarity": predictions[i],
                                "fold": sample_folds[i], "accuracy": accuracy, "f1": f1,
                            }, ensure_ascii=False) + "\n")
                        print(f"{language} {mode} {name}: accuracy={accuracy:.4f} f1={f1:.4f}")
                    handle.flush()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
