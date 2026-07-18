# Sycophancy Audit Summary

Last updated: 2026-07-18

## Goal

Measure whether small instruction-tuned language models change answers to agree with a user, first on subjective opinions and then on objective factual questions.

## Phase 1: Existing Sycophancy Benchmark

We evaluated two distinct Anthropic `model-written-evals` subsets: NLP opinions and political preferences. The philosophy filename was excluded because it was byte-for-byte identical to the NLP file.

For every example, we scored the A/B answer log-probabilities twice: once with the original labels and once with `(A)` and `(B)` swapped. A higher counterbalanced score means the model more often preferred the answer matching the user's stated belief after controlling for label position.

All runs used 200 balanced examples per subset and seed 42.

| Model | NLP belief matching | Political belief matching | Interpretation |
|---|---:|---:|---|
| Gemma 3 270M IT | 50.0% | 53.0% | Near chance; strong option-position effects. |
| Hunyuan 0.5B Instruct | 50.0% | 45.5% | Near chance; strong option-position effects. |
| OLMo-2 1B Instruct | 77.0% | 54.0% | Strong NLP effect, weak political effect. |
| Qwen3 0.6B | 73.0% | 54.0% | Strong NLP effect, weak political effect. |
| Qwen3.5 2B | 80.0% | 61.0% | Stronger matching, with some NLP layout sensitivity. |
| Gemma 4 E2B IT | 93.0% | 69.5% | Strongest and most layout-robust belief matching. |

Phase 1 suggests that belief matching is domain-dependent and becomes easier to measure in stronger models. It does not establish a universal rate of sycophancy because the benchmark consists of subjective or socially sensitive questions.

## Phase 2: Factual Pressure Control

We use the full BoolQ validation set: 3,270 passage-plus-yes/no questions with ground-truth answers. Every question is asked neutrally and after a user confidently states the wrong answer. Both A/B label layouts are scored and averaged.

| Model | Neutral accuracy | Incorrect-pressure accuracy | Net accuracy difference | Correct-to-wrong rate |
|---|---:|---:|---:|---:|
| Gemma 3 270M IT | 62.2% | 0.0% | -62.2 points | 62.2% |
| Hunyuan 0.5B Instruct | 62.1% | 0.1% | -62.1 points | 62.1% |
| OLMo-2 1B Instruct | 58.4% | 19.4% | -39.1 points | 41.4% |
| Qwen3 0.6B | 38.2% | 26.7% | -11.5 points | 11.5% |
| Qwen3.5 2B | 80.9% | 42.8% | -38.1 points | 38.1% |
| Gemma 4 E2B IT | 78.9% | 55.0% | -23.9 points | 25.0% |

These values describe this specific dataset and prompt template. The Phase 2 JSONL files record every example's neutral and pressure margin, allowing later checks by item or failure type.

## Code and Reproducibility

- `src/sycophancy-audit/phase_1/` contains the Phase 1 benchmark inspector, evaluator, plan, and logs.
- `src/sycophancy-audit/phase_2/` contains the BoolQ evaluator, plan, and full-validation JSONL/log results.
- `src/sycophancy-audit/phase_3/` contains the planned evidence-grounded self-assessment design.
- Phase 1 and Phase 2 use direct A/B continuation log-probabilities, not sampled model generations.
- Run logs are kept with the relevant phase. Model weights, environments, caches, and tokens are not stored in the repository.

## Next Decision

Inspect per-example Phase 2 failures and decide whether to add a second factual benchmark before moving to Phase 3.
