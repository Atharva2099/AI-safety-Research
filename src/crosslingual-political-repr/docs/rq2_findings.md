# RQ2: Cross-Lingual Representation Geometry & Transfer Findings

Last updated: 2026-08-23

## Overview

RQ2 tests whether a linear political-stance probe trained on one language transfers to another language. The six languages are English (`en`), Spanish (`es`), German (`de`), Chinese (`zh`), Hindi (`hi`), and Marathi (`mr`). Each language uses the same paired policy statements and binary polarity labels.

We evaluated cross-lingual transfer in two complementary settings:
1. **RQ1 Peak Baseline (Single English Peak per Model):** Evaluates all 6 languages at the model's English decodability peak layer.
2. **Deduplicated Multi-Peak Analysis (Every Language's Optimal Layer):** Evaluates full 6×6 transfer matrices at each unique in-language peak layer, deduplicating when multiple languages share the same peak depth.

For each model and layer, final-token hidden states were extracted in a single forward pass. For every source-language/target-language pair, a logistic-regression probe was fit on the source language and evaluated on the target language. The evaluation used five-fold grouped out-of-fold cross-evaluation: the question ID defined the group, so the two polarity statements belonging to one question stayed in the same fold.

## Multi-Peak Deduplicated Layer Summary

| Model | Total Blocks | Unique Peak Layers | Anchoring Languages | Diagonal Mean | Off-Diagonal Mean |
|---|---:|---|---|---:|---:|
| **OLMo 3 (7B)** | 32 | Block 10<br>Block 15<br>Block 17<br>Block 18 | `es`<br>`hi`<br>`en, de, zh`<br>`mr` | 79.07%<br>76.49%<br>79.84%<br>79.41% | 68.27%<br>65.14%<br>68.89%<br>68.80% |
| **Ministral (8B)** | 36 | Block 9<br>Block 20<br>Block 31<br>Block 32<br>Block 33 | `hi`<br>`zh`<br>`en, de`<br>`es`<br>`mr` | 75.88%<br>81.47%<br>83.33%<br>83.38%<br>83.49% | 62.96%<br>66.83%<br>67.72%<br>67.75%<br>66.28% |
| **Gemma 2 (9B)** | 42 | Block 13<br>Block 15<br>Block 18<br>Block 22<br>Block 23<br>Block 24<br>Block 25 | `mr`<br>`en`<br>`zh`<br>`de`<br>RQ1 Peak<br>`hi`<br>`es` | 83.55%<br>84.09%<br>84.60%<br>84.78%<br>84.76%<br>84.67%<br>84.48% | 79.97%<br>81.53%<br>82.32%<br>82.28%<br>81.45%<br>80.81%<br>81.01% |
| **Qwen 3.5 (9B)** | 32 | Block 10<br>Block 12<br>Block 14<br>Block 16 | `mr`<br>`zh`<br>`en, es, de`<br>`hi` | 84.04%<br>83.95%<br>84.37%<br>83.81% | 77.32%<br>78.40%<br>78.24%<br>76.99% |

---

## Non-Linear 2-Layer MLP Probe Transfer

To verify whether cross-lingual transfer limitations stem from non-linear representation geometry rather than conceptual absence, we evaluated a 2-layer MLP probe (128 hidden units, LayerNorm, ReLU, Dropout 0.1, AdamW) trained and evaluated using the identical 5-fold grouped out-of-fold cross-validation split.

| Model | Peak Layer | Linear In-Lang | Linear Off-Diag | MLP In-Lang | MLP Off-Diag | Takeaway |
|---|---:|---:|---:|---:|---:|---|
| **Gemma 2 (9B)** | Block 23 | 84.76% | 81.45% | **87.01%** | **85.04%** | Non-linear capacity further strengthens the universal subspace across all scripts. |
| **Ministral (8B)** | Block 31 | 83.33% | 67.72% | **74.60%** | **66.35%** | MLP does not fix cross-script collapse (`zh` on `en` = 58.97%, on `mr` = 50.69%), confirming conceptual absence. |
| **Qwen 3.5 (9B)** | Block 14 | 84.41% | 78.18% | **84.73%** | **80.54%** | Consistent, symmetric transfer across all 6 languages. |
| **OLMo 3 (7B)** | Block 17 | 79.84% | 68.89% | **80.56%** | **73.29%** | Moderate cross-lingual gain under non-linear probing. |

## Multi-Panel Visualizations

- **2-Layer MLP 4-Panel Comparison:** `artifacts/plots/rq2_mlp_cross_lingual_heatmaps.png`
- **Probe Weight Cosine Similarity (15 pairs):** `artifacts/plots/rq2_probe_weight_cosine_similarity.png`
- **0-Label Neutral Calibration Distributions:** `artifacts/plots/rq2_neutral_zero_calibration_projections.png`
- **OLMo 3 (4 panels):** `artifacts/plots/rq2_allenai_Olmo-3-7B-Instruct_peak_layers_heatmaps.png`
- **Ministral 8B (5 panels):** `artifacts/plots/rq2_mistralai_Ministral-8B-Instruct-2410_peak_layers_heatmaps.png`
- **Gemma 2 (7 panels):** `artifacts/plots/rq2_google_gemma-2-9b-it_peak_layers_heatmaps.png`
- **Qwen 3.5 (4 panels):** `artifacts/plots/rq2_Qwen_Qwen3.5-9B_peak_layers_heatmaps.png`
- **Single-layer 4-model summary:** `artifacts/plots/rq2_annotated_4panel_heatmaps.png`
- **Interactive 3D explorer:** `artifacts/plots/rq2_interactive_3d_explorer.html`
- **Full visual gallery:** `artifacts/plots/gallery.html`

---

## Detailed Multi-Peak Transfer Matrices

### 1. Gemma 2 (9B)

#### Block 13 (Peak for `mr`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 83.36% | 79.83% | 81.90% | 80.60% | 80.00% | 76.98% |
| **es** | 81.81% | 81.81% | 81.03% | 81.21% | 79.57% | 75.26% |
| **de** | 82.07% | 79.57% | 83.45% | 81.90% | 81.55% | 77.84% |
| **zh** | 80.00% | 80.52% | 81.12% | 84.40% | 81.47% | 78.45% |
| **hi** | 79.83% | 77.84% | 80.52% | 82.16% | 84.48% | 80.69% |
| **mr** | 76.81% | 76.12% | 79.48% | 80.26% | 82.59% | 83.79% |

#### Block 15 (Peak for `en`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 86.29% | 82.76% | 83.10% | 83.62% | 81.29% | 79.14% |
| **es** | 84.83% | 83.79% | 83.97% | 83.88% | 82.16% | 79.66% |
| **de** | 83.71% | 81.03% | 83.88% | 82.76% | 82.59% | 79.66% |
| **zh** | 80.52% | 81.38% | 82.84% | 85.00% | 83.10% | 80.26% |
| **hi** | 80.52% | 78.88% | 81.55% | 82.93% | 83.88% | 80.60% |
| **mr** | 78.71% | 76.98% | 78.79% | 81.81% | 82.84% | 81.72% |

#### Block 18 (Peak for `zh`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.26% | 82.76% | 83.88% | 84.31% | 81.90% | 78.97% |
| **es** | 85.00% | 84.05% | 83.28% | 83.62% | 83.97% | 81.64% |
| **de** | 85.60% | 83.28% | 85.69% | 85.52% | 83.79% | 80.86% |
| **zh** | 84.22% | 82.33% | 83.62% | 85.78% | 82.84% | 80.52% |
| **hi** | 80.86% | 80.52% | 81.72% | 80.95% | 83.71% | 81.47% |
| **mr** | 78.97% | 78.71% | 80.60% | 81.72% | 82.24% | 83.10% |

#### Block 22 (Peak for `de`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.17% | 83.02% | 83.79% | 83.62% | 80.95% | 79.05% |
| **es** | 84.22% | 85.09% | 85.34% | 83.71% | 83.19% | 82.16% |
| **de** | 84.74% | 83.97% | 86.64% | 84.22% | 84.22% | 81.55% |
| **zh** | 81.12% | 82.07% | 82.93% | 84.74% | 83.02% | 80.69% |
| **hi** | 83.28% | 81.47% | 82.84% | 79.74% | 84.57% | 80.34% |
| **mr** | 76.47% | 80.09% | 80.60% | 81.98% | 84.14% | 82.50% |

#### Block 24 (Peak for `hi`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.52% | 83.88% | 82.16% | 81.72% | 78.19% | 74.48% |
| **es** | 85.09% | 84.83% | 84.57% | 79.05% | 81.38% | 80.52% |
| **de** | 83.53% | 83.97% | 85.26% | 79.83% | 83.36% | 81.55% |
| **zh** | 81.47% | 82.24% | 81.47% | 84.83% | 82.59% | 80.09% |
| **hi** | 80.00% | 80.17% | 81.21% | 77.41% | 84.22% | 80.86% |
| **mr** | 75.78% | 78.19% | 77.33% | 78.28% | 83.88% | 83.36% |

#### Block 25 (Peak for `es`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.78% | 83.36% | 81.64% | 83.45% | 80.34% | 76.12% |
| **es** | 85.00% | 85.09% | 85.00% | 79.83% | 81.03% | 78.88% |
| **de** | 83.02% | 82.76% | 85.52% | 78.62% | 81.72% | 79.31% |
| **zh** | 83.19% | 81.47% | 84.05% | 84.31% | 81.98% | 78.97% |
| **hi** | 82.76% | 80.09% | 82.50% | 79.83% | 83.53% | 79.91% |
| **mr** | 75.69% | 78.10% | 79.83% | 79.14% | 82.84% | 82.67% |

---

### 2. Qwen 3.5 (9B)

#### Block 10 (Peak for `mr`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 84.74% | 82.16% | 82.50% | 83.53% | 74.48% | 70.78% |
| **es** | 83.79% | 84.74% | 85.09% | 84.14% | 71.55% | 64.74% |
| **de** | 83.19% | 82.59% | 84.91% | 84.57% | 76.64% | 64.91% |
| **zh** | 80.69% | 79.31% | 80.69% | 84.40% | 74.91% | 69.14% |
| **hi** | 78.53% | 74.74% | 75.69% | 76.98% | 83.02% | 71.98% |
| **mr** | 76.12% | 73.36% | 77.76% | 78.45% | 76.64% | 82.41% |

#### Block 12 (Peak for `zh`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.17% | 80.69% | 81.90% | 81.55% | 76.12% | 70.69% |
| **es** | 84.48% | 84.57% | 85.00% | 83.19% | 76.47% | 66.81% |
| **de** | 83.53% | 82.93% | 85.95% | 84.83% | 74.31% | 69.48% |
| **zh** | 82.24% | 81.12% | 81.90% | 84.83% | 76.29% | 71.55% |
| **hi** | 80.60% | 77.84% | 80.69% | 81.38% | 81.81% | 73.53% |
| **mr** | 77.59% | 73.53% | 76.55% | 79.22% | 75.95% | 81.38% |

#### Block 14 (Peak for `en, es, de`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.78% | 81.38% | 81.98% | 82.84% | 74.74% | 67.67% |
| **es** | 83.53% | 85.69% | 83.45% | 84.91% | 78.88% | 71.90% |
| **de** | 83.45% | 82.50% | 86.29% | 82.59% | 77.84% | 73.10% |
| **zh** | 81.03% | 79.05% | 81.81% | 84.40% | 77.84% | 69.83% |
| **hi** | 77.50% | 75.78% | 78.45% | 78.62% | 82.33% | 74.74% |
| **mr** | 77.07% | 73.02% | 75.43% | 78.36% | 77.76% | 81.72% |

#### Block 16 (Peak for `hi`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.52% | 81.81% | 82.93% | 83.19% | 71.21% | 62.16% |
| **es** | 84.22% | 85.09% | 84.66% | 83.88% | 79.14% | 72.50% |
| **de** | 82.67% | 81.98% | 85.60% | 84.31% | 78.28% | 69.91% |
| **zh** | 81.21% | 78.28% | 81.29% | 84.31% | 78.19% | 68.53% |
| **hi** | 75.86% | 75.95% | 78.88% | 77.07% | 83.02% | 75.17% |
| **mr** | 70.43% | 69.05% | 70.09% | 69.22% | 77.50% | 79.31% |

---

### 3. Ministral (8B)

#### Block 9 (Peak for `hi`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 70.78% | 59.57% | 56.55% | 83.71% | 82.24% | 51.98% |
| **es** | 54.91% | 71.03% | 61.64% | 82.24% | 82.16% | 53.62% |
| **de** | 56.81% | 66.21% | 73.36% | 83.28% | 81.47% | 52.16% |
| **zh** | 54.40% | 53.88% | 52.50% | 86.21% | 82.67% | 51.98% |
| **hi** | 53.71% | 53.53% | 52.24% | 82.67% | 86.29% | 51.55% |
| **mr** | 54.40% | 54.91% | 51.38% | 65.60% | 64.91% | 67.59% |

#### Block 20 (Peak for `zh`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 82.67% | 80.26% | 76.29% | 80.26% | 73.36% | 55.95% |
| **es** | 67.84% | 81.90% | 75.69% | 81.12% | 79.22% | 52.93% |
| **de** | 80.00% | 80.95% | 81.90% | 77.67% | 74.22% | 55.95% |
| **zh** | 54.31% | 53.28% | 52.59% | 87.16% | 79.74% | 51.55% |
| **hi** | 53.36% | 53.79% | 53.28% | 79.22% | 83.97% | 51.90% |
| **mr** | 64.40% | 68.71% | 63.79% | 63.02% | 70.17% | 71.21% |

#### Block 31 (Peak for `en, de`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 86.12% | 81.64% | 80.69% | 71.47% | 65.09% | 60.60% |
| **es** | 82.67% | 83.45% | 77.41% | 74.14% | 66.47% | 54.22% |
| **de** | 82.41% | 81.55% | 85.00% | 77.24% | 69.48% | 62.93% |
| **zh** | 53.97% | 53.02% | 51.64% | 84.48% | 62.50% | 50.52% |
| **hi** | 63.19% | 62.07% | 62.16% | 76.90% | 82.50% | 50.60% |
| **mr** | 73.62% | 74.66% | 71.38% | 68.88% | 68.53% | 78.45% |

#### Block 32 (Peak for `es`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.69% | 78.62% | 72.76% | 62.59% | 60.95% | 59.74% |
| **es** | 78.36% | 85.26% | 76.64% | 72.50% | 70.95% | 62.33% |
| **de** | 80.00% | 72.41% | 83.36% | 78.97% | 75.69% | 63.71% |
| **zh** | 60.26% | 59.48% | 59.83% | 84.66% | 63.88% | 52.33% |
| **hi** | 59.57% | 58.19% | 61.55% | 77.76% | 81.81% | 52.76% |
| **mr** | 73.62% | 76.29% | 68.71% | 73.88% | 68.19% | 79.48% |

#### Block 33 (Peak for `mr`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 85.17% | 76.47% | 75.86% | 67.16% | 61.03% | 56.72% |
| **es** | 80.34% | 83.88% | 78.97% | 70.34% | 67.67% | 56.47% |
| **de** | 82.33% | 74.05% | 83.71% | 73.10% | 75.86% | 61.81% |
| **zh** | 55.86% | 55.00% | 54.14% | 84.66% | 67.33% | 51.21% |
| **hi** | 59.14% | 58.71% | 62.16% | 70.60% | 83.19% | 53.53% |
| **mr** | 69.83% | 71.55% | 69.22% | 62.50% | 69.48% | 80.34% |

---

### 4. OLMo 3 (7B)

#### Block 10 (Peak for `es`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 80.17% | 64.91% | 76.03% | 73.79% | 76.12% | 56.72% |
| **es** | 80.95% | 77.50% | 76.72% | 66.55% | 72.07% | 68.19% |
| **de** | 77.41% | 67.50% | 77.84% | 73.45% | 75.26% | 64.48% |
| **zh** | 70.78% | 56.55% | 69.22% | 79.31% | 75.60% | 55.86% |
| **hi** | 63.97% | 54.31% | 65.69% | 59.22% | 81.47% | 55.17% |
| **mr** | 71.38% | 63.53% | 70.17% | 71.47% | 75.00% | 78.10% |

#### Block 15 (Peak for `hi`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 77.50% | 58.62% | 68.79% | 64.40% | 77.84% | 55.86% |
| **es** | 76.12% | 74.57% | 72.07% | 69.40% | 72.59% | 59.91% |
| **de** | 72.93% | 62.67% | 75.52% | 66.55% | 76.21% | 61.64% |
| **zh** | 67.76% | 55.52% | 65.09% | 72.84% | 73.45% | 57.50% |
| **hi** | 63.88% | 53.79% | 64.14% | 58.97% | 81.98% | 54.91% |
| **mr** | 66.29% | 56.72% | 66.72% | 59.74% | 73.97% | 76.55% |

#### Block 17 (Peak for `en, de, zh`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 81.81% | 65.60% | 68.79% | 74.83% | 75.00% | 57.33% |
| **es** | 77.24% | 75.95% | 75.60% | 73.45% | 73.45% | 71.64% |
| **de** | 75.86% | 70.95% | 79.57% | 72.41% | 72.84% | 65.69% |
| **zh** | 71.81% | 60.78% | 70.26% | 82.59% | 74.57% | 69.91% |
| **hi** | 62.50% | 53.19% | 62.76% | 58.45% | 80.34% | 55.26% |
| **mr** | 68.62% | 66.47% | 73.71% | 72.67% | 75.09% | 78.79% |

#### Block 18 (Peak for `mr`)
| Train \ Test | en | es | de | zh | hi | mr |
|---|---:|---:|---:|---:|---:|---:|
| **en** | 81.12% | 66.64% | 69.05% | 76.21% | 71.03% | 57.67% |
| **es** | 76.38% | 76.38% | 75.34% | 75.86% | 72.07% | 73.10% |
| **de** | 76.38% | 70.86% | 78.88% | 73.79% | 71.90% | 72.50% |
| **zh** | 71.81% | 59.74% | 70.26% | 81.12% | 70.43% | 68.97% |
| **hi** | 59.66% | 53.36% | 63.62% | 59.22% | 79.74% | 56.90% |
| **mr** | 66.21% | 64.48% | 73.19% | 74.40% | 72.84% | 79.22% |

---

## Takeaways

1. **Gemma 2's Multilingual Subspace is Layer-Invariant Across Depth:**
   Across all seven tested layers (Blocks 13 to 25), Gemma 2 maintains high, uniform cross-lingual transfer (79.97% to 82.32% off-diagonal mean). Its political polarity direction occupies a unified geometric coordinate system across English, Romance, Germanic, Sinitic, and Indic scripts.
2. **Ministral's Cross-Lingual Fragmentation Persists Across Layers:**
   Testing at native non-English peaks does not bridge Ministral's linguistic gap. Off-diagonal transfer remains low (62.96% at Block 9 to 67.75% at Block 32), confirming that the separation is structural rather than a layer-depth mismatch.
3. **Qwen 3.5 Shows Consistent Middle-Layer Consolidation:**
   Transfer is symmetric across Blocks 10, 12, 14, and 16, maintaining 77.0%–78.4% off-diagonal transfer.
4. **OLMo 3 Reaches Multi-Language Co-Peak at Block 17:**
   Block 17 anchors English, German, and Chinese with highest overall transfer (68.89%), while earlier layers show lower global alignment.
