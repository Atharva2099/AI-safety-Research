"""Plot corrected width-8 MLP 6x6 cross-lingual transfer heatmaps."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
PLOTS = ROOT / "artifacts" / "plots"

MODELS = [
    ("OLMo 3 (7B)", "allenai_Olmo-3-7B-Instruct"),
    ("Ministral (8B)", "mistralai_Ministral-8B-Instruct-2410"),
    ("Gemma 2 (9B)", "google_gemma-2-9b-it"),
    ("Qwen 3.5 (9B)", "Qwen_Qwen3.5-9B"),
]


def plot_mlp_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    sns.set_theme(style="white", font_scale=1.0)
    last_heatmap = None

    for ax, (display_name, model_id) in zip(axes.flat, MODELS):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
        file_path = RESULTS / f"mlp_cross_lingual_matrix_{slug}.json"
        if not file_path.exists():
            print(f"Missing {file_path.name}")
            continue

        data = json.loads(file_path.read_text(encoding="utf-8"))
        matrix = data["matrix"]
        layer = data["layer"]

        diag_mean = sum(matrix[i][i] for i in range(6)) / 6
        off_diag_vals = [matrix[i][j] for i in range(6) for j in range(6) if i != j]
        off_diag_mean = sum(off_diag_vals) / len(off_diag_vals)

        annot = [[f"{val:.1%}" for val in row] for row in matrix]

        last_heatmap = sns.heatmap(
            matrix,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="YlGnBu",
            vmin=0.50,
            vmax=0.90,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"fontsize": 11, "weight": "bold"},
            cbar=False,
            xticklabels=LANGUAGES,
            yticklabels=LANGUAGES,
        )
        ax.set_title(
            f"{display_name} — Block {layer} (width-8 MLP)\n(Diag: {diag_mean*100:.1f}%, Off-Diag: {off_diag_mean*100:.1f}%)",
            fontsize=13,
            weight="bold",
            pad=8,
        )
        ax.set_xlabel("Test Language Residuals", fontsize=11)
        ax.set_ylabel("Train Language MLP Probe", fontsize=11)
        ax.tick_params(axis="both", labelsize=10, length=0)

    fig.suptitle("Corrected width-8 MLP Cross-Lingual Probe Transfer (Peak Layers)", fontsize=16, weight="bold", y=1.02)
    if last_heatmap is not None:
        cbar = fig.colorbar(last_heatmap.collections[0], ax=axes, label="Transfer Accuracy", shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

    output_png = PLOTS / "rq2_mlp_cross_lingual_heatmaps.png"
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png.name}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_mlp_heatmaps()


if __name__ == "__main__":
    main()
