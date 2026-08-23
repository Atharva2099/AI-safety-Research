"""Plot deduplicated multi-panel peak-layer cross-lingual heatmaps for each model."""

import json
import math
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
    ("OLMo 3 (7B)", "allenai/Olmo-3-7B-Instruct"),
    ("Ministral (8B)", "mistralai/Ministral-8B-Instruct-2410"),
    ("Gemma 2 (9B)", "google/gemma-2-9b-it"),
    ("Qwen 3.5 (9B)", "Qwen/Qwen3.5-9B"),
]


def plot_model_panels(model_display_name: str, model_id: str):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    matrix_files = sorted(RESULTS.glob(f"cross_lingual_matrix_{slug}_block*.json"))
    
    # Fallback to legacy single-layer matrix if new files aren't found
    if not matrix_files:
        legacy_file = RESULTS / f"cross_lingual_matrix_{slug}.json"
        if legacy_file.exists():
            matrix_files = [legacy_file]
        else:
            print(f"No matrix files found for {model_display_name}")
            return
            
    records = [json.loads(f.read_text(encoding="utf-8")) for f in matrix_files]
    # Sort records by layer index
    records.sort(key=lambda r: r["layer"])
    
    n_panels = len(records)
    if n_panels == 0:
        return
        
    cols = min(3, n_panels)
    rows = math.ceil(n_panels / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.2 * rows), constrained_layout=True)
    if n_panels == 1:
        axes = np.array([axes])
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    
    sns.set_theme(style="white", font_scale=1.0)
    last_heatmap = None
    
    for idx, (ax, rec) in enumerate(zip(axes_flat, records)):
        matrix = rec["matrix"]
        layer = rec["layer"]
        anchors = rec.get("anchoring_languages", [])
        anchor_str = ", ".join(a.upper() for a in anchors) if anchors else "EN (RQ1)"
        
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
        ax.set_title(f"Block {layer} (Peak: {anchor_str})", fontsize=14, weight="bold", pad=8)
        ax.set_xlabel("Test Language", fontsize=11)
        ax.set_ylabel("Train Language", fontsize=11)
        ax.tick_params(axis="both", labelsize=10, length=0)
        
    # Hide any unused subplots
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
        
    fig.suptitle(f"{model_display_name} — Peak-Layer 6×6 Transfer Geometry", fontsize=16, weight="bold", y=1.02)
    if last_heatmap is not None:
        cbar = fig.colorbar(last_heatmap.collections[0], ax=axes, label="Transfer Accuracy", shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        
    output_png = PLOTS / f"rq2_{slug}_peak_layers_heatmaps.png"
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png.name} ({n_panels} panels)")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    for display_name, model_id in MODELS:
        plot_model_panels(display_name, model_id)


if __name__ == "__main__":
    main()
