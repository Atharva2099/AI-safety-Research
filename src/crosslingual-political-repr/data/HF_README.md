---
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
- translation
language:
- en
- es
- de
- zh
- hi
- mr
multilingual: true
size_categories:
- 1K<n<10K
source: Anthropic/llm_global_opinions
---

# Multilingual Political Representations Dataset

Last updated: 2026-08-20

This dataset contains 1,246 parallel policy statements across six languages
(English, Spanish, German, Chinese, Hindi, and Marathi), with polarities in
`{-1, 0, 1}`.

## Attribution

The dataset is derived from `Anthropic/llm_global_opinions`, which combines
items from the Pew Global Attitudes Survey and the World Values Survey.

Statements and translations were generated and structured using
`gemini-3.5-flash-lite` via Google Cloud Vertex AI.

## License

This dataset is licensed under CC BY-NC-SA 4.0.
