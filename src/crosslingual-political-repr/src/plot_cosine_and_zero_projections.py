"""Plot probe weight cosine similarity heatmaps and 0-label neutral calibration distributions."""

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


def plot_weight_cosine_heatmaps():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    sns.set_theme(style="white", font_scale=1.0)
    last_heatmap = None
    
    for ax, (display_name, model_id) in zip(axes.flat, MODELS):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
        file_path = RESULTS / f"probe_weight_cosine_matrix_{slug}.json"
        if not file_path.exists():
            print(f"Missing {file_path.name}")
            continue
            
        data = json.loads(file_path.read_text(encoding="utf-8"))
        matrix = data["cosine_matrix"]
        layer = data["layer"]
        off_diag_mean = data.get("off_diagonal_mean", 0.0)
        
        annot = [[f"{val:.2f}" for val in row] for row in matrix]
        
        last_heatmap = sns.heatmap(
            matrix,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="mako",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"fontsize": 11, "weight": "bold"},
            cbar=False,
            xticklabels=LANGUAGES,
            yticklabels=LANGUAGES,
        )
        ax.set_title(f"{display_name} — Block {layer}\n(Mean Off-Diag Cosine = {off_diag_mean:.2f})", fontsize=13, weight="bold", pad=8)
        ax.set_xlabel("Language Probe Weight", fontsize=11)
        ax.set_ylabel("Language Probe Weight", fontsize=11)
        ax.tick_params(axis="both", labelsize=10, length=0)
        
    fig.suptitle("Probe Weight Direction Cosine Similarity (Peak Layers)", fontsize=16, weight="bold", y=1.02)
    if last_heatmap is not None:
        cbar = fig.colorbar(last_heatmap.collections[0], ax=axes, label="Cosine Similarity", shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        
    output_png = PLOTS / "rq2_probe_weight_cosine_similarity.png"
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png.name}")


def plot_zero_calibration_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    
    colors = {"Negative (-1)": "#dc2626", "Neutral (0)": "#eab308", "Positive (+1)": "#2563eb"}
    
    for ax, (display_name, model_id) in zip(axes.flat, MODELS):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
        file_path = RESULTS / f"neutral_zero_projection_{slug}.json"
        if not file_path.exists():
            print(f"Missing {file_path.name}")
            continue
            
        data = json.loads(file_path.read_text(encoding="utf-8"))
        layer = data["layer"]
        
        # English probe on English residuals
        en_proj = data["projections"]["en"]["en"]
        
        neg_scores = en_proj["raw_neg_scores"]
        zero_scores = en_proj["raw_zero_scores"]
        pos_scores = en_proj["raw_pos_scores"]
        
        sns.kdeplot(neg_scores, ax=ax, fill=True, color=colors["Negative (-1)"], label=f"Polarity -1 (μ={np.mean(neg_scores):.1f})", linewidth=2)
        sns.kdeplot(zero_scores, ax=ax, fill=True, color=colors["Neutral (0)"], label=f"Neutral 0 (μ={np.mean(zero_scores):.1f})", linewidth=2)
        sns.kdeplot(pos_scores, ax=ax, fill=True, color=colors["Positive (+1)"], label=f"Polarity +1 (μ={np.mean(pos_scores):.1f})", linewidth=2)
        
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Decision Boundary")
        ax.set_title(f"{display_name} — Block {layer} (EN Probe)", fontsize=13, weight="bold")
        ax.set_xlabel("1D Probe Projection Score (W·x + b)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        
    fig.suptitle("1D Stance Calibration: Neutral (0) vs. Opposing (±1) Statements", fontsize=16, weight="bold", y=1.02)
    output_png = PLOTS / "rq2_neutral_zero_calibration_projections.png"
    fig.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png.name}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_weight_cosine_heatmaps()
    plot_zero_calibration_distributions()


if __name__ == "__main__":
    main()
