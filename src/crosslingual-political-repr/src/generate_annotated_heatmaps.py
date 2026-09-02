"""Generate fixed-layer cross-lingual probe-transfer heatmaps."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
FIXED_OUTPUT = ROOT / "artifacts" / "plots" / "rq2_annotated_4panel_heatmaps.png"
SOURCE_OPTIMAL_OUTPUT = (
    ROOT / "artifacts" / "plots" / "rq2_source_optimal_4panel_heatmaps.png"
)
OLD_PAPER = "#E0D4C5"
HUNTER_GREEN = "#355E3B"
TRANSFER_CMAP = LinearSegmentedColormap.from_list(
    "old_paper_to_hunter_green",
    [OLD_PAPER, HUNTER_GREEN],
)

# The fourth model is stored under its repository/Hugging Face slug, Ministral.
MODELS = (
    ("OLMo 3", "allenai_Olmo-3-7B-Instruct"),
    ("Qwen 3.5", "Qwen_Qwen3.5-9B"),
    ("Gemma 2", "google_gemma-2-9b-it"),
    ("Ministral 8B", "mistralai_Ministral-8B-Instruct-2410"),
)

PREFERRED_LAYERS = {
    "allenai_Olmo-3-7B-Instruct": {
        "en": 17, "es": 10, "de": 17, "zh": 17, "hi": 15, "mr": 18,
    },
    "Qwen_Qwen3.5-9B": {
        "en": 14, "es": 14, "de": 14, "zh": 12, "hi": 16, "mr": 10,
    },
    "google_gemma-2-9b-it": {
        "en": 15, "es": 25, "de": 22, "zh": 18, "hi": 24, "mr": 13,
    },
    "mistralai_Ministral-8B-Instruct-2410": {
        "en": 31, "es": 32, "de": 31, "zh": 20, "hi": 9, "mr": 33,
    },
}


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


def load_source_optimal_matrix(
    model_slug: str,
) -> tuple[list[list[float]], dict[str, str]]:
    """Select each source row from that language's preferred-layer matrix."""
    preferred = PREFERRED_LAYERS[model_slug]
    matrix = []

    for row_index, source in enumerate(LANGUAGES):
        layer = preferred[source]
        matrix_path = RESULTS / f"cross_lingual_matrix_{model_slug}_block{layer}.json"
        transfer = json.loads(matrix_path.read_text(encoding="utf-8"))

        if tuple(transfer["languages"]) != LANGUAGES:
            raise ValueError(f"Unexpected language order in {matrix_path}")
        if transfer["layer"] != layer:
            raise ValueError(f"Unexpected layer in {matrix_path}")
        if len(transfer["matrix"]) != len(LANGUAGES):
            raise ValueError(f"Expected a 6x6 transfer matrix in {matrix_path}")

        row = transfer["matrix"][row_index]
        if len(row) != len(LANGUAGES):
            raise ValueError(f"Expected a 6x6 transfer matrix in {matrix_path}")
        matrix.append(row)

    tags = {
        language: f"Block {preferred[language]}"
        for language in LANGUAGES
    }
    return matrix, tags


def layer_tag(layer: str) -> str:
    return "Emb" if layer in {"Embedding", "Embedding Layer"} else layer.removeprefix("Block ")


def plot_heatmaps(
    results: dict[str, tuple[list[list[float]], dict[str, str]]],
    output: Path,
    show_source_layers: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=1.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12),
        constrained_layout=True,
        facecolor=OLD_PAPER,
    )

    for axis, (name, (matrix, peaks)) in zip(axes.flat, results.items()):
        axis.set_facecolor(OLD_PAPER)
        annotations = [
            [
                "—" if np.isnan(value) else f"{value:.1%}"
                for value in row
            ]
            for source, row in zip(LANGUAGES, matrix)
        ]
        source_labels = (
            [
                f"{language} · L{layer_tag(peaks[language])}"
                for language in LANGUAGES
            ]
            if show_source_layers
            else LANGUAGES
        )
        sns.heatmap(
            matrix,
            ax=axis,
            annot=annotations,
            fmt="",
            cmap=TRANSFER_CMAP,
            vmin=0.50,
            vmax=0.90,
            linewidths=0.8,
            linecolor=OLD_PAPER,
            annot_kws={"fontsize": 12, "weight": "bold"},
            cbar=False,
            xticklabels=LANGUAGES,
            yticklabels=source_labels,
        )
        for text, value in zip(axis.texts, np.asarray(matrix).ravel()):
            if not np.isnan(value):
                text.set_color(HUNTER_GREEN if value < 0.70 else OLD_PAPER)

        selected_layers = sorted({layer_tag(layer) for layer in peaks.values()})
        layer_suffix = (
            f" · Layer {selected_layers[0]}" if len(selected_layers) == 1 else ""
        )
        axis.set_title(
            f"{name}{layer_suffix}",
            fontsize=16,
            weight="bold",
            color=HUNTER_GREEN,
            pad=10,
        )
        if np.isnan(matrix).all():
            axis.text(0.5, 0.5, "Complete 6×6\ntransfer matrix unavailable", transform=axis.transAxes,
                      ha="center", va="center", fontsize=13, weight="bold", color=HUNTER_GREEN)
        axis.set_xlabel("Evaluation language", fontsize=12, color=HUNTER_GREEN)
        axis.set_ylabel("Probe training language", fontsize=12, color=HUNTER_GREEN)
        axis.tick_params(axis="both", labelsize=11, length=0, colors=HUNTER_GREEN)

    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor=OLD_PAPER)
    plt.close(figure)


def main() -> None:
    fixed_results = {}
    source_optimal_results = {}
    for name, model_slug in MODELS:
        matrix, peaks = compute_optimal_matrix(model_slug)
        fixed_results[name] = (matrix, peaks)

        source_matrix, source_peaks = load_source_optimal_matrix(model_slug)
        source_optimal_results[name] = (source_matrix, source_peaks)
        print(
            name,
            "source-optimal layers:",
            ", ".join(
                f"{language}={layer}"
                for language, layer in source_peaks.items()
            ),
        )

    plot_heatmaps(fixed_results, FIXED_OUTPUT)
    plot_heatmaps(
        source_optimal_results,
        SOURCE_OPTIMAL_OUTPUT,
        show_source_layers=True,
    )
    print(f"Saved {FIXED_OUTPUT}")
    print(f"Saved {SOURCE_OPTIMAL_OUTPUT}")


if __name__ == "__main__":
    main()
