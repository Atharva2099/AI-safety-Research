# RQ2: Cross-Lingual Representation & Transfer Findings

Last updated: 2026-09-01

## Overview

RQ2 tests whether a linear political-stance probe trained on one language transfers to another language. The six languages are English (`en`), Spanish (`es`), German (`de`), Chinese (`zh`), Hindi (`hi`), and Marathi (`mr`). Each language uses the same paired policy statements and binary polarity labels.

We evaluated cross-lingual transfer in two complementary settings:
1. **RQ1 Peak Baseline (Single English Peak per Model):** Evaluates all 6 languages at the model's English decodability peak layer.
2. **Deduplicated Multi-Peak Analysis (Every Language's Optimal Layer):** Evaluates full 6×6 transfer matrices at each unique in-language peak layer, deduplicating when multiple languages share the same peak depth.

For each model and layer, the final non-padding token's intermediate hidden/residual-stream vector was extracted in a single forward pass. For every source-language/target-language pair, a logistic-regression probe was fit on the source language and evaluated on the target language. The evaluation used five-fold grouped out-of-fold cross-evaluation: the question ID defined the group, so the two polarity statements belonging to one question stayed in the same fold.

## Multi-Peak Deduplicated Layer Summary

| Model | Total Blocks | Unique Peak Layers | Anchoring Languages | Diagonal Mean | Off-Diagonal Mean |
|---|---:|---|---|---:|---:|
| **OLMo 3 (7B)** | 32 | Block 10<br>Block 15<br>Block 17<br>Block 18 | `es`<br>`hi`<br>`en, de, zh`<br>`mr` | 79.07%<br>76.49%<br>79.84%<br>79.41% | 68.27%<br>65.14%<br>68.89%<br>68.80% |
| **Ministral (8B)** | 36 | Block 9<br>Block 20<br>Block 31<br>Block 32<br>Block 33 | `hi`<br>`zh`<br>`en, de`<br>`es`<br>`mr` | 75.88%<br>81.47%<br>83.33%<br>83.38%<br>83.49% | 62.96%<br>66.83%<br>67.72%<br>67.75%<br>66.28% |
| **Gemma 2 (9B)** | 42 | Block 13<br>Block 15<br>Block 18<br>Block 22<br>Block 23<br>Block 24<br>Block 25 | `mr`<br>`en`<br>`zh`<br>`de`<br>RQ1 Peak<br>`hi`<br>`es` | 83.55%<br>84.09%<br>84.60%<br>84.78%<br>84.76%<br>84.67%<br>84.48% | 79.97%<br>81.53%<br>82.32%<br>82.28%<br>81.45%<br>80.81%<br>81.01% |
| **Qwen 3.5 (9B)** | 32 | Block 10<br>Block 12<br>Block 14<br>Block 16 | `mr`<br>`zh`<br>`en, es, de`<br>`hi` | 84.04%<br>83.95%<br>84.37%<br>83.81% | 77.32%<br>78.40%<br>78.24%<br>76.99% |

---

## Corrected Non-Linear MLP Probe Transfer

The corrected evaluation uses a one-hidden-layer MLP trained on frozen language-model activations, with hidden width 8 and ReLU (`Linear(in_dim, 8) → ReLU → Linear(8, 1)`), source-only standardization, grouped five-fold outer splits, source-only inner validation for epoch selection, and a fresh outer-training refit. Five initialization seeds were used for the primary result (150 fits); the shuffled-label control used seed 1729 (30 fits). After fitting on one source language, that fitted probe was held fixed and reused unchanged across all six target languages. Earlier oversized-probe numbers and conclusions are not used here.

| Model | Layer | Linear diagonal | Corrected MLP diagonal | Δ diagonal | Linear off-diagonal | Corrected MLP off-diagonal | Δ off-diagonal | Shuffled control range / mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OLMo 3 (7B)** | 17 | 79.84% | 70.23% | −9.61 pp | 68.89% | 64.84% | −4.05 pp | 46.55–54.14% / 50.04% |
| **Ministral (8B)** | 31 | 83.33% | 83.06% | −0.28 pp | 67.72% | 67.32% | −0.40 pp | 47.24–57.41% / 52.68% |
| **Gemma 2 (9B)** | 23 | 84.76% | 85.26% | +0.50 pp | 81.45% | 84.33% | +2.89 pp | 46.38–59.22% / 53.50% |
| **Qwen 3.5 (9B)** | 14 | 84.37% | 83.36% | −1.00 pp | 78.24% | 81.11% | +2.87 pp | 46.12–62.59% / 54.97% |

The source-language diagonals are the first comparison gate. After rounding, the Ministral, Gemma, and Qwen diagonal means differ from their linear values by approximately one percentage point or less; this is a descriptive screening rule, not statistical equivalence. OLMo is 9.61 points lower, so its transfer comparison should not be interpreted as an MLP improvement over its linear baseline. For the full source→target matrix, mean off-diagonal changes relative to the linear artifact were −4.05, −0.40, +2.89, and +2.87 percentage points for OLMo, Ministral, Gemma, and Qwen respectively.

Variability summaries for the corrected matrices: mean pooled fold/seed accuracy SD was 3.84, 3.69, 2.49, and 2.46 percentage points; mean SD across the five seed-level means was 1.63, 1.56, 0.39, and 0.47 points; mean SD across fold-level means was 2.77, 2.48, 2.36, and 2.25 points, in the same model order. Fold-level variability exceeds seed-level variability for every model in these summaries. In plain terms, changing the held-out questions moved results more than changing initialization. These SDs describe variation in this evaluation; they are not uncertainty intervals or statistical tests. The one shuffled-label control produced the values shown in the table: OLMo 46.55–54.14% (mean 50.04%), Ministral 47.24–57.41% (mean 52.68%), Gemma 46.38–59.22% (mean 53.50%), and Qwen 46.12–62.59% (mean 54.97%). Multiple independent shuffles or a permutation analysis are needed for inference.

These results establish measured transfer behavior for this width-8 probe and split procedure. They do not establish that the underlying representations lack a concept, explain why OLMo's width-8 fit trails its linear baseline, or justify claims about larger MLPs.

### Corrected diagnostics and extraction limitations

Across the 150 final refits per model, mean `source_training` accuracy was: OLMo 76.11%; Ministral 90.18%; Gemma 89.73%; Qwen 89.61%. Mean `source_validation` accuracy from the separate inner epoch-selection models was: OLMo 70.52%; Ministral 82.37%; Gemma 84.12%; Qwen 82.77%. These are means of the recorded fold/seed metrics, not confidence intervals.

OLMo has a language-specific oddity: Hindi has 90.30% `source_training` accuracy from the final refit and 81.16% `source_validation` accuracy from the separate inner epoch-selection model, compared with Spanish at 67.90% and 63.42%, and Chinese at 68.98% and 65.81%, respectively. The reason Hindi performs best here is unknown. The official OLMo model card labels the model language as English and does not document Hindi coverage there; that does not prove Hindi was absent from Dolma 3 pretraining. Tokenizer coverage, training-corpus composition, and token-position diagnostics are plausible checks, not established explanations.

---

## 8-Condition Representation Extraction & Diagnostic Findings

The final diagnostic evaluation compares eight representation extraction strategies across all four models, using batch size 16 and grouped five-fold out-of-fold cross-validation with inner 3-fold grouped hyperparameter selection for `C ∈ {0.0001, 0.001, 0.01, 0.1, 1, 10}` (ties resolving to the smallest `C`). Probes were fit once per source/fold/condition and evaluated strictly on out-of-fold target rows (1,160 predictions per cell). The diagnostic reference layers are 17 for OLMo, 31 for Ministral, 23 for Gemma, and 12 for Qwen. Qwen Layer 12 follows the earliest-layer rule for its tied English-only peak at Layers 12–13.

| Model | Layer | Baseline `current_raw` Off-Diag (Diag) | `mean_raw` Off-Diag (Diag) | `stripped_raw` Off-Diag (Diag) | `current_l2` Off-Diag (Diag) | `stripped_l2` Off-Diag (Diag) |
|---|---:|---:|---:|---:|---:|---:|
| **OLMo 3 (7B)** | 17 | 70.64% (80.70%) | 68.50% (81.97%) | 72.07% (81.26%) | 72.50% (80.39%) | 73.28% (81.51%) |
| **Ministral (8B)** | 31 | 68.13% (83.45%) | 67.64% (81.21%) | 68.18% (83.03%) | 62.22% (72.47%) | 70.27% (81.87%) |
| **Gemma 2 (9B)** | 23 | 84.67% (86.09%) | 76.01% (83.89%) | 80.28% (84.43%) | 85.11% (85.93%) | 80.90% (84.27%) |
| **Qwen 3.5 (9B)** | 12 | 80.68% (84.60%) | 68.63% (82.34%) | 71.48% (80.52%) | 81.48% (84.05%) | 73.04% (80.07%) |

### Predeclared Primary Contrasts (28 Simultaneous 95% Bootstrap Intervals)

Simultaneous 95% max-statistic bootstrap confidence intervals (critical half-width ±1.21 pp across all 28 contrasts, 5,000 resamples over question IDs):

1. **Gemma 2 (9B):**
   - `mean_raw - current_raw`: −8.66 pp [−9.86, −7.46] pp (Significant drop under mean pooling)
   - `stripped_raw - current_raw`: −4.39 pp [−5.60, −3.19] pp (Significant drop when terminal punctuation is stripped)
   - `current_l2 - current_raw`: +0.43 pp [−0.77, +1.64] pp (Invariant to L2 vector normalization)
2. **Qwen 3.5 (9B):**
   - `mean_raw - current_raw`: −12.05 pp [−13.26, −10.84] pp (Significant drop under mean pooling)
   - `stripped_raw - current_raw`: −9.20 pp [−10.41, −7.99] pp (Significant drop when terminal punctuation is stripped)
   - `current_l2 - current_raw`: +0.80 pp [−0.41, +2.01] pp (Interval includes zero)
3. **OLMo 3 (7B):**
   - `mean_raw - current_raw`: −2.14 pp [−3.34, −0.94] pp
   - `stripped_raw - current_raw`: +1.42 pp [+0.22, +2.63] pp (Modest gain when terminal punctuation is stripped)
   - `current_l2 - current_raw`: +1.86 pp [+0.65, +3.06] pp (Significant gain under L2 normalization)
   - `stripped_l2 - stripped_raw`: +1.21 pp [+0.01, +2.41] pp
4. **Ministral (8B):**
   - `mean_raw - current_raw`: −0.49 pp [−1.70, +0.71] pp
   - `current_l2 - current_raw`: −5.91 pp [−7.12, −4.71] pp (Sharp drop under raw final L2 normalization)
   - `content_l2 - content_raw`: +2.17 pp [+0.97, +3.37] pp
   - `stripped_l2 - stripped_raw`: +2.09 pp [+0.88, +3.29] pp

### Controls & Limitations

1. **Cross-Language Character N-Gram Control:** A logistic-regression classifier using character 3–5-gram TF-IDF features was fit separately on each source language and applied unchanged to every target language. Each source language and fold had its own training-only vocabulary and IDF values. Mean within-language accuracy was **81.42%**, but mean cross-language accuracy was **51.67%** (`artifacts/results/crosslingual_char_ngram_controls.json.gz`).

   | Train \ Test | en | es | de | zh | hi | mr |
   |---|---:|---:|---:|---:|---:|---:|
   | **en** | 83.62% | 60.95% | 51.90% | 50.00% | 50.00% | 50.00% |
   | **es** | 62.59% | 79.66% | 53.28% | 50.00% | 49.91% | 50.00% |
   | **de** | 50.00% | 50.86% | 80.78% | 50.00% | 50.00% | 50.00% |
   | **zh** | 49.66% | 50.09% | 48.97% | 79.66% | 50.00% | 50.00% |
   | **hi** | 50.00% | 50.00% | 50.00% | 50.00% | 83.28% | 59.31% |
   | **mr** | 50.52% | 50.26% | 50.60% | 50.00% | 61.21% | 81.55% |

   Direct surface transfer was strongest between English and Spanish and between Hindi and Marathi. The control does not test whether a multilingual semantic text representation can reproduce activation transfer.
2. **Earlier Surface-Control Artifact:** The earlier `text_surface_controls.json` pooled three incompatible controls into a misleading 61.48% average. Its saved fold assignments also do not match the current documented `GroupKFold` procedure (902 of 1,160 English assignments differ), so its 81.91% character n-gram result is not used as the corrected baseline.
3. **1D Vector Magnitude Controls:** Logistic regression trained strictly on scalar log vector length (`log(||h||)`) scored **49.8% to 50.7%** off-diagonal accuracy across all models and conditions. Scalar vector length alone was near chance in this evaluation.
4. **Batch-Size Parity:** Qwen Layer 12 `current_raw` extracted with batch size 8 reproduced the historical Layer 12 matrix exactly in all 36 cells. Batch-size-16 extraction changed 580 of 41,760 predictions and had a maximum cell difference of 1.29 pp. This identifies BF16 extraction batch size as the source of the Qwen parity discrepancy. The earlier parity discrepancies for OLMo, Gemma, and Ministral remain unresolved.

---

## Multi-Panel Visualizations

- **Width-8 one-hidden-layer MLP 4-Panel Comparison:** `artifacts/plots/rq2_mlp_cross_lingual_heatmaps.png`
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

The multi-peak matrices above are linear-probe observations at each selected layer. They should be kept separate from the corrected width-8 one-hidden-layer MLP result and the 8-condition extraction diagnostic, each evaluated at one selected layer per model.

1. **Gemma 2 & Qwen 3.5 Exhibit a Raw Final-Token Advantage:** Among the raw extraction methods, the final non-padding token produces the highest cross-lingual transfer in Gemma and Qwen (84.67% and 80.68%). Mean pooling reduces transfer by 8.66 pp in Gemma and 12.05 pp in Qwen. Stripping terminal punctuation reduces it by 4.39 pp and 9.20 pp, respectively. These results apply to this dataset, extraction procedure, and selected layers.
2. **OLMo 3 Benefits from Normalization and Punctuation Stripping:** Unlike Gemma and Qwen, OLMo's transfer improves when terminal punctuation is stripped (`stripped_raw` +1.42 pp) and under L2 normalization (`current_l2` +1.86 pp, `stripped_l2` +2.64 pp). Mean pooling decreases transfer (−2.14 pp).
3. **Ministral 8B Transfer is Condition-Specific:** While raw final L2 normalization causes transfer to drop (−5.91 pp), terminal-content and stripped L2 representations improve transfer (+2.17 pp and +2.09 pp).
4. **Scalar Vector Length Was Near Chance:** 1D vector magnitude controls (`norm_only`) yielded approximately 50.0% accuracy across all models and conditions in this evaluation.
5. **Scope & Limitations:** These findings describe out-of-fold linear and width-8 probe transfer on 580 parallel question pairs across 6 languages. They do not establish general ideological representation, causal mechanism, or universal coordinate alignment. Earlier roadmap mentions of CKA were scoped out in favor of direct cross-lingual transfer matrices, weight cosine similarities, and extraction ablations.
