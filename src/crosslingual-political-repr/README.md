# Cross-Lingual Political Representations

Last updated: 2026-08-20

Investigating where political information emerges across transformer layers in multilingual LLMs, whether representational geometry is shared across languages, whether directions causally steer generation, and how language effects separate from geopolitical context.

---

## Candidate Models (7B–15B Range)

We will start with **one model** first before running cross-model comparisons:

- **Qwen 3.5 (9B):** Primary target for broad multilingual support (>100 languages).
- **Gemma 4 (12B):** Strong open-weight candidate with wide multilingual pre-training.
- **Ministral 3 (8B) / NeMo (7B–12B):** Representative European model family.
- **OLMo 3 (7B):** Fully open pre-training architecture for comparison.

---

## Target Languages

- English (en)
- Spanish (es)
- German (de)
- Hindi (hi)
- Mandarin Chinese (zh)
- Marathi (mr)
- Japanese (ja)

---

## Research Plan: Four Complementary Ideas

### RQ1: Layerwise Emergence

- **Goal:** Map layer depth vs. linear decodability of political labels across languages.
- **Method:** Extract hidden states layer-by-layer for semantically matched policy prompts; train linear probes per layer and per language.
- **Controls:** Benchmark against non-political controls (topic, sentiment, lexical cues, country entities).

### RQ2: Cross-Lingual Geometry

- **Goal:** Determine if political directions are shared across languages or language-specific.
- **Method:** Perform cross-language probe transfer (e.g., train on English, test on Hindi/Chinese) and measure representational similarity (cosine similarity, CKA).

### RQ3: Causal Steering Across Languages

- **Goal:** Test if internal political directions causally shift generated stances cross-lingually.
- **Method:** Extract steering vectors in source language (e.g., English) and inject/patch during inference in target language (e.g., Hindi, Chinese). Measure stance shifts vs. neutral/control directions.

### RQ4: Language vs. Geopolitical Context

- **Goal:** Disentangle prompt language from target country/geopolitical priors.
- **Method:** Factorial prompt design independently varying `(Prompt Language) × (Geopolitical Entity) × (Policy Issue)` to isolate linguistic conditioning vs. country-specific knowledge.

---

## Execution Phasing

1. **Dataset Preparation:** Build/curate parallel policy statements with ground-truth policy labels across all 6 languages.
2. **Phase 1 (RQ1 + RQ2):** Single model activation extraction → layerwise linear probing → cross-language probe transfer.
3. **Phase 2 (RQ4):** Factorial language × country context probes.
4. **Phase 3 (RQ3):** Causal activation steering and cross-lingual intervention matrix.
