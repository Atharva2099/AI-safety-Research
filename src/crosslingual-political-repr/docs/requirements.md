# Project Requirements & Architecture

Last updated: 2026-08-20

Simple reference for dependencies, candidate models, datasets, and research goals for cross-lingual political representations.

---

## 1. Libraries & Dependencies

- **torch & transformers:** Hugging Face model loading and forward-pass execution.
- **accelerate:** Memory-efficient model loading and multi-GPU execution.
- **scikit-learn:** Linear probing (LogisticRegression, Ridge, CKA, PCA).
- **datasets:** Loading and managing Hugging Face and local JSONL datasets.
- **pandas & numpy:** Hidden state array processing and metric evaluation.
- **matplotlib & seaborn:** Plotting layer-wise accuracy, transfer matrices, and steering curves.
- **deepl / google-trans:** Automated translation validation for parallel prompts.

---

## 2. Candidate Models (7B–15B Range)

- **Qwen 3.5 (9B):** Primary target for broad multilingual support (>100 languages).
- **Gemma 4 (12B):** Strong open-weight candidate with wide multilingual pre-training.
- **Ministral 3 (8B) / NeMo (7B–12B):** Representative European model family.
- **OLMo 3 (7B):** Fully open pre-training architecture for comparison.

---

## 3. Target Languages

- **English (en):** Primary reference language.
- **Spanish (es) & German (de):** Western high-resource target languages.
- **Hindi (hi):** South Asian non-Western target language.
- **Marathi (mr):** South Asian non-Western target language.
- **Mandarin Chinese (zh):** East Asian non-Western target language.
- **Japanese (ja):** East Asian non-Western target language.

---

## 4. Datasets & Formats

- **US Policy & Lawmaker Data:** DW-NOMINATE policy statements / roll-call records (US politics).
- **Voting Advice Applications (VAA):** EU election questionnaire items (e.g., Wahl-O-Mat, EU Manifesto Project).
- **Global Opinion QA:** Pew Research and World Values Survey cross-national policy questions.
- **Factorial Geopolitical Dataset:** `(Prompt Language) × (Referenced Country) × (Policy Stance)` 3-way parallel items.

---

## 5. Core Ideas (1-Line Summaries)

- **RQ1 (Layerwise Emergence):** Measure at which transformer layer political stance becomes linearly readable across languages.
- **RQ2 (Cross-Lingual Geometry):** Test whether an English-trained linear probe predicts political stance in Hindi, Chinese, etc.
- **RQ3 (Causal Steering):** Inject internal political directions at inference time to causally shift generated stances across languages.
- **RQ4 (Language vs. Context):** Separate whether internal shifts come from query language or the country/entity being discussed.
