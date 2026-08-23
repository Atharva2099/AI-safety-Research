"""Generate source-language peak-layer cross-transfer heatmaps."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
OUTPUT = ROOT / "artifacts" / "plots" / "rq2_annotated_4panel_heatmaps.png"

# The fourth model is stored under its repository/Hugging Face slug, Ministral.
MODELS = (
    ("OLMo 3", "allenai_Olmo-3-7B-Instruct"),
    ("Qwen 3.5", "Qwen_Qwen3.5-9B"),
    ("Gemma 2", "google_gemma-2-9b-it"),
    ("Ministral 8B", "mistralai_Ministral-8B-Instruct-2410"),
)


def layer_number(layer: str) -> int:
    """Return a sortable layer index, with the embedding before block 0."""
    if layer in {"Embedding", "Embedding Layer"}:
        return 0
    return int(layer.removeprefix("Block ")) + 1


def load_entries(model_slug: str) -> dict[tuple[str, str, str], float]:
    """Load accuracy by source language, target language, and layer."""
    entries: dict[tuple[str, str, str], list[float]] = {}
    paths = sorted(RESULTS.glob(f"multilingual_probe_{model_slug}_*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No multilingual result files found for {model_slug}"
        )

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["source_language"], row["target_language"], row["layer"])
            entries.setdefault(key, []).append(float(row["accuracy"]))

    return {key: sum(values) / len(values) for key, values in entries.items()}


def compute_optimal_matrix(model_slug: str) -> tuple[list[list[float]], dict[str, str]]:
    """Load a complete matrix, or the exact pairs available in the JSONL files."""
    entries = load_entries(model_slug)
    matrix_path = RESULTS / f"cross_lingual_matrix_{model_slug}.json"
    transfer = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.exists() else None
    if transfer is not None and tuple(transfer["languages"]) != LANGUAGES:
        raise ValueError(f"Unexpected language order in {matrix_path}")

    peaks: dict[str, tuple[float, str]] = {}
    for language in LANGUAGES:
        candidates = [
            (accuracy, layer)
            for (source, target, layer), accuracy in entries.items()
            if source == target == language
        ]
        if candidates:
            peaks[language] = max(candidates, key=lambda item: (item[0], -layer_number(item[1])))
        else:
            print(f"Warning: missing {language} in-language results for {model_slug}")

    if transfer is not None:
        matrix = transfer["matrix"]
        if len(matrix) != len(LANGUAGES) or any(len(row) != len(LANGUAGES) for row in matrix):
            raise ValueError(f"Expected a 6x6 transfer matrix in {matrix_path}")
        # Matrix artifacts contain one exact 6x6 run at this layer.  Do not
        # attach independently selected JSONL peak layers to those values.
        tags = {language: f"Block {transfer['layer']}" for language in LANGUAGES}
        return matrix, tags

    print(f"Warning: missing complete transfer matrix for {model_slug}; plotting available JSONL pairs")
    matrix = [[float("nan")] * len(LANGUAGES) for _ in LANGUAGES]
    tags = {}
    for row_index, source in enumerate(LANGUAGES):
        if source not in peaks:
            continue
        _, layer = peaks[source]
        tags[source] = layer
        for column_index, target in enumerate(LANGUAGES):
            value = entries.get((source, target, layer))
            if value is not None:
                matrix[row_index][column_index] = value
    return matrix, tags


def layer_tag(layer: str) -> str:
    return "Emb" if layer in {"Embedding", "Embedding Layer"} else layer.removeprefix("Block ")


def plot_heatmaps(results: dict[str, tuple[list[list[float]], dict[str, str]]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=1.0)
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    last_heatmap = None

    for axis, (name, (matrix, peaks)) in zip(axes.flat, results.items()):
        annotations = [
            [
                "—" if np.isnan(value) else f"{value:.1%}\nL{layer_tag(peaks[source])}"
                for value in row
            ]
            for source, row in zip(LANGUAGES, matrix)
        ]
        last_heatmap = sns.heatmap(
            matrix,
            ax=axis,
            annot=annotations,
            fmt="",
            cmap="YlGnBu",
            vmin=0.50,
            vmax=0.90,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"fontsize": 12, "weight": "bold"},
            cbar=False,
            xticklabels=LANGUAGES,
            yticklabels=LANGUAGES,
        )
        axis.set_title(name, fontsize=16, weight="bold", pad=10)
        if np.isnan(matrix).all():
            axis.text(0.5, 0.5, "Complete 6×6\ntransfer matrix unavailable", transform=axis.transAxes,
                      ha="center", va="center", fontsize=13, weight="bold")
        axis.set_xlabel("Test Residuals Language", fontsize=12)
        axis.set_ylabel("Train Probe Language", fontsize=12)
        axis.tick_params(axis="both", labelsize=11, length=0)

    figure.colorbar(last_heatmap.collections[0], ax=axes, label="Transfer accuracy", shrink=0.82)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    results = {}
    for name, model_slug in MODELS:
        matrix, peaks = compute_optimal_matrix(model_slug)
        results[name] = (matrix, peaks)
        print(name, "peak layers:", ", ".join(f"{language}={layer}" for language, layer in peaks.items()))
    plot_heatmaps(results)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
