"""Plot layerwise probe accuracy and separation from chance."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
PLOTS = ROOT / "artifacts" / "plots"
MODELS = {
    "OLMo 3 (7B)": "probe_results_allenai_Olmo-3-7B-Instruct.jsonl",
    "Qwen 2.5 (7B)": "probe_results_Qwen_Qwen2.5-7B-Instruct.jsonl",
    "Qwen 3.5 (9B)": "probe_results_Qwen_Qwen3.5-9B.jsonl",
    "Gemma 2 (9B)": "probe_results_google_gemma-2-9b-it.jsonl",
    "Ministral (8B)": "probe_results_mistralai_Ministral-8B-Instruct-2410.jsonl",
}


def load_metrics(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_layer = {}
    for row in rows:
        by_layer[row["layer"]] = row
    return list(by_layer.values())


def layer_position(layer: str, total_blocks: int) -> tuple[float, int]:
    """Return normalized depth (%) and transformer block index for a layer."""
    if layer == "Embedding Layer":
        return -3.0, -1
    block = int(layer.removeprefix("Block "))
    return block / (total_blocks - 1) * 100.0, block


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    metrics = {name: load_metrics(RESULTS / filename) for name, filename in MODELS.items()
               if (RESULTS / filename).exists()}
    missing = [name for name in MODELS if name not in metrics]
    if not metrics:
        raise FileNotFoundError("No probe result files found")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rows in metrics.items():
        total_blocks = max(
            int(row["layer"].removeprefix("Block "))
            for row in rows
            if row["layer"] != "Embedding Layer"
        ) + 1
        x = [layer_position(row["layer"], total_blocks)[0] for row in rows]
        ax.plot(x, [row["accuracy"] for row in rows], marker="o", linewidth=2, markersize=3, label=name)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set(title="RQ1: Layerwise political-label emergence", xlabel="Normalized Layer Depth (0% to 100%)", ylabel="5-fold accuracy", xlim=(-5, 100))
    block_axis = ax.secondary_xaxis("top")
    block_axis.set_xlabel("Transformer Block Index (Embedding = -1; blocks = 0 to N-1)")
    block_axis.set_xticks([0, 100])
    block_axis.set_xticklabels(["-1 / 0 (Embedding / Block 0)", "N-1"])
    ax.legend()
    if missing:
        ax.text(0.01, 0.02, "Unavailable: " + ", ".join(missing), transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "rq1_layerwise_emergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rows in metrics.items():
        total_blocks = max(
            int(row["layer"].removeprefix("Block "))
            for row in rows
            if row["layer"] != "Embedding Layer"
        ) + 1
        x = [layer_position(row["layer"], total_blocks)[0] for row in rows]
        ax.plot(x, [row["accuracy"] - 0.5 for row in rows], marker="o", linewidth=2, markersize=3, label=name)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title="RQ1: Residual separation from chance", xlabel="Normalized Layer Depth (0% to 100%)", ylabel="Accuracy − 0.5", xlim=(-5, 100))
    block_axis = ax.secondary_xaxis("top")
    block_axis.set_xlabel("Transformer Block Index (Embedding = -1; blocks = 0 to N-1)")
    block_axis.set_xticks([0, 100])
    block_axis.set_xticklabels(["-1 / 0 (Embedding / Block 0)", "N-1"])
    ax.legend()
    if missing:
        ax.text(0.01, 0.02, "Unavailable: " + ", ".join(missing), transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "rq1_residual_separation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
