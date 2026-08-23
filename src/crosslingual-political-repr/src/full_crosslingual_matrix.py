"""Plot full source-language to target-language probe transfer matrices."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
PLOTS = ROOT / "artifacts" / "plots"
MODELS = {
    "OLMo 3": "allenai_Olmo-3-7B-Instruct",
    "Qwen 3.5": "Qwen_Qwen3.5-9B",
    "Gemma 2": "google_gemma-2-9b-it",
    "Ministral": "mistralai_Ministral-8B-Instruct-2410",
}


def layer_number(layer: str) -> int:
    return 0 if layer == "Embedding" else int(layer.removeprefix("Block ")) + 1


def load_model(model_slug: str) -> tuple[dict[str, int], list[list[float]]]:
    """Return language peak layers and source x target accuracies."""
    matrix_path = RESULTS / f"cross_lingual_matrix_{model_slug}.json"
    if matrix_path.exists():
        result = json.loads(matrix_path.read_text(encoding="utf-8"))
        return {language: result["layer"] for language in LANGUAGES}, result["matrix"]

    entries: dict[tuple[str, str, str], float] = {}
    paths = list(RESULTS.glob(f"multilingual_probe_{model_slug}_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No multilingual probe files found for {model_slug}")

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["source_language"], row["target_language"], row["layer"])
            entries[key] = row["accuracy"]

    peaks: dict[str, tuple[float, str]] = {}
    for language in LANGUAGES:
        candidates = [
            (accuracy, layer)
            for (source, target, layer), accuracy in entries.items()
            if source == target == language
        ]
        if not candidates:
            raise ValueError(f"Missing in-language results for {model_slug}: {language}")
        peaks[language] = max(candidates, key=lambda item: (item[0], -layer_number(item[1])))

    matrix = []
    for source in LANGUAGES:
        row = []
        for target in LANGUAGES:
            _, layer = peaks[source]
            try:
                row.append(entries[(source, target, layer)])
            except KeyError as exc:
                raise ValueError(
                    f"Missing transfer result for {model_slug}: "
                    f"{source} -> {target} at {layer}"
                ) from exc
        matrix.append(row)
    return {language: layer_number(layer) for language, (_, layer) in peaks.items()}, matrix


def print_table(name: str, peaks: dict[str, int], matrix: list[list[float]]) -> None:
    print(f"\n### {name} (peak layers: " + ", ".join(
        f"{language}={layer}" for language, layer in peaks.items()
    ) + ")")
    print("| train \\ test | " + " | ".join(LANGUAGES) + " |")
    print("|---|" + "---:|" * len(LANGUAGES))
    for language, values in zip(LANGUAGES, matrix):
        print(f"| {language} | " + " | ".join(f"{value:.3f}" for value in values) + " |")


def plot(matrices: dict[str, tuple[dict[str, int], list[list[float]]]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, (name, (_peaks, matrix)) in zip(axes.flat, matrices.items()):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set(title=name, xticks=range(len(LANGUAGES)), yticks=range(len(LANGUAGES)),
                 xticklabels=LANGUAGES, yticklabels=LANGUAGES, xlabel="Test language",
                 ylabel="Train language")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axes, label="Accuracy", shrink=0.8)
    fig.savefig(PLOTS / "rq2_full_6x6_heatmaps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, help="Only inspect one model; skips the 4-panel plot")
    args = parser.parse_args()
    selected = {args.model: MODELS[args.model]} if args.model else MODELS
    matrices = {name: load_model(slug) for name, slug in selected.items()}
    for name, (peaks, matrix) in matrices.items():
        print_table(name, peaks, matrix)
    if len(matrices) == len(MODELS):
        plot(matrices)


if __name__ == "__main__":
    main()
