# Phase 1: Existing Sycophancy Benchmark

Last updated: 2026-07-18

## Question

When a user states an opinion, does a model prefer the answer that agrees with that user more than the answer that disagrees?

## Null Hypothesis

After controlling for answer-label position, the model has no systematic preference for the answer matching the user's stated belief. The counterbalanced belief-matching rate is 50%.

## Benchmark and Scope

We used two distinct subsets from `Anthropic/model-written-evals`:

- `sycophancy_on_nlp_survey.jsonl`: opinions about NLP.
- `sycophancy_on_political_typology_quiz.jsonl`: political preferences.

The repository's `sycophancy_on_philpapers2020.jsonl` file had the identical SHA-256 hash as the NLP file, so it was excluded rather than treated as an independent domain.

Each run uses a balanced fixed sample of 200 examples, seed 42, with 100 examples whose belief-matching answer is `(A)` and 100 whose belief-matching answer is `(B)`.

## Method

For every prompt, the evaluator scores the log-probability of the continuations ` (A)` and ` (B)` rather than sampling a free-form answer. It then swaps the two answer labels while keeping the answer text and user statement fixed.

The final rate is the fraction of examples with a positive average belief-matching margin across the original and swapped prompt. This counterbalancing prevents a fixed preference for `(A)` or `(B)` from being mistaken for agreement.

## Results

| Model | NLP | Political |
|---|---:|---:|
| Gemma 3 270M IT | 50.0% | 53.0% |
| Hunyuan 0.5B Instruct | 50.0% | 45.5% |
| OLMo-2 1B Instruct | 77.0% | 54.0% |
| Qwen3 0.6B | 73.0% | 54.0% |
| Qwen3.5 2B | 80.0% | 61.0% |
| Gemma 4 E2B IT | 93.0% | 69.5% |

Higher scores mean more agreement with the user's stated belief. Gemma 4 E2B is the clearest result because its NLP matching rate was 92.5% in both label layouts. Qwen3.5 2B had a high NLP score but retained substantial layout sensitivity: 84.5% original-order matching versus 59.5% after swapping.

## Limitations

- These are 200-example pilot samples, not confidence-interval estimates.
- Each model uses its own chat template. This is realistic for deployment but not a fully standardized prompt format.
- The smaller Gemma and Hunyuan models show strong option-position effects, so their near-chance results do not establish an absence of sycophancy.
