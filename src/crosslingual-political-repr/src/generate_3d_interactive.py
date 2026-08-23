"""Generate an interactive 3D explorer for cross-lingual probe transfer.

Fixes over the previous version:
- Bilinearly densified surface so 6x6 data does not render as a sparse sheet.
- Hover only on true 6x6 cell markers (interpolated surface hover disabled).
- Explicit camera angle so the scene never opens edge-on or blank.
- Dropdown moved inside the plot area; z-axis starts at the 50% chance level.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
OUTPUT = ROOT / "artifacts" / "plots" / "rq2_interactive_3d_explorer.html"
MODELS = (
    ("OLMo 3", "allenai_Olmo-3-7B-Instruct"),
    ("Qwen 3.5", "Qwen_Qwen3.5-9B"),
    ("Gemma 2", "google_gemma-2-9b-it"),
    ("Ministral 8B", "mistralai_Ministral-8B-Instruct-2410"),
)
CHANCE, PEAK = 50.0, 90.0


def load_matrix(model_slug: str) -> tuple[np.ndarray, int]:
    """Load a source-language-by-target-language transfer matrix."""
    path = RESULTS / f"cross_lingual_matrix_{model_slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-lingual matrix: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if tuple(result["languages"]) != LANGUAGES:
        raise ValueError(f"Unexpected language order in {path}")
    matrix = np.asarray(result["matrix"], dtype=float) * 100.0
    if matrix.shape != (len(LANGUAGES), len(LANGUAGES)):
        raise ValueError(f"Expected a {len(LANGUAGES)}x{len(LANGUAGES)} matrix in {path}")
    return matrix, int(result["layer"])


def densify(grid: np.ndarray, factor: int = 6) -> np.ndarray:
    """Bilinearly upscale a grid by `factor` for smooth surface rendering."""
    n, m = grid.shape
    ys = np.linspace(0, n - 1, (n - 1) * factor + 1)
    xs = np.linspace(0, m - 1, (m - 1) * factor + 1)
    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, n - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, m - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def make_traces(name: str, matrix: np.ndarray, layer: int) -> list[go.Surface | go.Scatter3d]:
    """One smooth surface plus true-data markers carrying all hover detail."""
    axis = np.arange(len(LANGUAGES))
    dense_axis = np.linspace(0, len(LANGUAGES) - 1, (len(LANGUAGES) - 1) * 6 + 1)
    surface = go.Surface(
        x=dense_axis,
        y=dense_axis,
        z=densify(matrix),
        colorscale="Viridis",
        cmin=CHANCE,
        cmax=PEAK,
        opacity=0.92,
        showscale=True,
        colorbar={"title": {"text": "Accuracy (%)"}, "x": 1.02},
        contours={"z": {"show": True, "usecolormap": True, "project": {"z": True}}},
        hoverinfo="skip",
        name=name,
        visible=False,
        showlegend=False,
    )
    marker_x, marker_y, marker_z, custom = [], [], [], []
    for i, source in enumerate(LANGUAGES):
        for j, target in enumerate(LANGUAGES):
            marker_x.append(j)
            marker_y.append(i)
            marker_z.append(float(matrix[i, j]))
            custom.append([source, target, f"{matrix[i, j]:.1f}", layer])
    markers = go.Scatter3d(
        x=marker_x,
        y=marker_y,
        z=marker_z,
        mode="markers",
        marker={"size": 4, "color": marker_z, "colorscale": "Viridis",
                "cmin": CHANCE, "cmax": PEAK, "line": {"width": 2, "color": "white"}},
        customdata=custom,
        hovertemplate=(
            "Source probe: %{customdata[0]}<br>"
            "Target residuals: %{customdata[1]}<br>"
            "Transfer accuracy: %{customdata[2]}%<br>"
            "Layer: Block %{customdata[3]}<extra>" + name + "</extra>"
        ),
        name=f"{name} (true cells)",
        visible=False,
    )
    return [surface, markers]


def main() -> None:
    traces, names = [], []
    for name, model_slug in MODELS:
        matrix, layer = load_matrix(model_slug)
        traces.extend(make_traces(name, matrix, layer))
        names.append(name)
        print(f"Loaded {name}: 6x6 matrix at Block {layer}, diagonal mean {np.mean(np.diag(matrix)):.1f}%")

    traces[0].visible = True
    traces[1].visible = True
    buttons = [
        {
            "label": name,
            "method": "update",
            "args": [{"visible": [index // 2 == selected for index in range(len(traces))]},
                     {"title": f"RQ2: Cross-Lingual Transfer Accuracy - {name} (chance = 50%)"}],
        }
        for selected, name in enumerate(names)
    ]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title={"text": f"RQ2: Cross-Lingual Transfer Accuracy - {names[0]} (chance = 50%)",
               "y": 0.98, "x": 0.5, "xanchor": "center"},
        scene={
            "xaxis": {"title": "Target Language (Residuals)", "tickmode": "array",
                      "tickvals": list(range(len(LANGUAGES))), "ticktext": LANGUAGES},
            "yaxis": {"title": "Source Language (Trained Probe)", "tickmode": "array",
                      "tickvals": list(range(len(LANGUAGES))), "ticktext": LANGUAGES},
            "zaxis": {"title": "Transfer Accuracy (%)", "range": [CHANCE - 2, PEAK]},
            "camera": {"eye": {"x": 1.7, "y": -1.7, "z": 0.85}, "center": {"x": 0, "y": 0, "z": -0.1}},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.1, "y": 1.1, "z": 0.7},
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "xanchor": "left",
                      "y": 1.06, "yanchor": "top"}],
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        height=800,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(OUTPUT, include_plotlyjs=True, full_html=True)
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
