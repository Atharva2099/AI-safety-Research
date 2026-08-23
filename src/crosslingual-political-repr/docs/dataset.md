# Global Opinion QA Dataset Documentation

Last updated: 2026-08-20

## Source and licensing

- **Upstream source:** `Anthropic/llm_global_opinions`, combining 2,203 Pew Global Attitudes Survey items and 353 World Values Survey items.
- **Pinned commit:** `cb2880488749218abb81802a94c2c62ebfde2f35`
- **Upstream timestamp:** 2023-06-29T00:46:48Z
- **Licensing:** CC BY-NC-SA 4.0

## Summary metrics

- **Total questions processed:** 2,556
- **Suitable for probing:** 580 (494 paired and 86 ternary)
- **Filtered out:** 1,976 non-policy or otherwise unsuitable questions
- **Total statements:** 1,246
- **Multilingual statements:** `data/multilingual_statements.json` contains 1,246 statements in six languages: English (`en`), Spanish (`es`), German (`de`), Mandarin Chinese (`zh`), Hindi (`hi`), and Marathi (`mr`).
- **Polarity counts:** 580 positive (`+1`), 580 negative (`-1`), and 86 neutral (`0`).

## Processing

Questions were processed with `gemini-3.5-flash-lite` on Vertex AI using ADC. Structured JSON generation was used, followed by polarity extraction with values in `{-1, 0, 1}`. Unsuitable questions were retained as filtered records without generated statements.

## Storage

- Raw upstream questions: `data/raw_global_opinions.json`
- Processed declarative statements: `data/declarative_statements.json`
- Translated statements: `data/multilingual_statements.json`
