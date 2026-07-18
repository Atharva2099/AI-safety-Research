# Bugs Squashed

Last updated: 2026-07-18

## 2026-07-18 - Duplicate Phase 1 benchmark file

**Issue:** `sycophancy_on_philpapers2020.jsonl` had the same SHA-256 hash as `sycophancy_on_nlp_survey.jsonl`.

**Impact:** Treating both files as independent domains would double-count the NLP examples.

**Fix:** Excluded the philosophy filename from Phase 1 results.

**Gain:** The benchmark now has two genuinely distinct domains: NLP opinions and political preferences.

## 2026-07-18 - Choice swapping assumed a `Choices:` marker

**Issue:** The initial Phase 1 swap function crashed on political prompts because they omit `Choices:`.

**Impact:** Political evaluation could not run.

**Fix:** Added a fallback that swaps the final A/B answer labels when the marker is absent.

**Gain:** The identical counterbalanced evaluator works for both Phase 1 prompt formats.

## 2026-07-18 - Choice swapping could alter biography text

**Issue:** One NLP prompt included `statement (A)` in the user's biography before the answer options.

**Impact:** Swapping every A/B marker would corrupt the user statement rather than only counterbalance answers.

**Fix:** When present, swapping is restricted to the text after `Choices:`.

**Gain:** Prompt semantics remain fixed while answer labels change.

## 2026-07-18 - L4 inference environment lacked Triton build prerequisites

**Issue:** Qwen inference initially failed because the L4 VM lacked a C compiler and Python development headers.

**Impact:** Triton could not compile its CUDA utility module.

**Fix:** Installed `build-essential` and `python3.10-dev`.

**Gain:** CUDA inference works in the existing `~/.venv`.

## 2026-07-18 - Gemma checkpoint access was unauthenticated

**Issue:** Gemma downloads returned Hugging Face gated-repository errors on the L4 VM.

**Impact:** Gemma evaluation could not start.

**Fix:** Authenticated with the Hugging Face CLI in the L4 virtual environment.

**Gain:** Both Gemma checkpoints load reproducibly on L4.

## 2026-07-18 - Phase 2 emitted aggregates only

**Issue:** The original BoolQ evaluator printed only aggregate results.

**Impact:** There was no record of selected questions, per-example margins, model versions, or correctness switches.

**Fix:** The evaluator now requires JSONL output containing metadata, dataset indices, question hashes, neutral and pressure margins, and a summary record.

**Gain:** Full-validation runs are auditable, reproducible, and diagnosable at the individual-example level.

## 2026-07-18 - Phase 2 did not validate A/B token parity

**Issue:** Unlike Phase 1, Phase 2 did not check whether A/B continuations had equal token length for each tokenizer.

**Impact:** Unequal candidate lengths could bias summed log-probabilities.

**Fix:** The evaluator records token IDs and stops if the candidate lengths differ.

**Gain:** The factual-pressure metric has the same candidate-length safeguard as Phase 1.

## 2026-07-18 - Full BoolQ evaluation was needlessly unbatched

**Issue:** Scoring each of the four prompt variants and two answer candidates separately would require 26,160 forward passes per model over full BoolQ.

**Impact:** The six-model evaluation would use unnecessary L4 time and make a full validation run impractical.

**Fix:** Batch the same padded prompt-plus-candidate sequences and recover each continuation log-probability from the final candidate-token positions.

**Verification:** The batched 10-item Gemma check must reproduce the earlier 70.0% neutral and 0.0% pressure accuracy before full runs begin.

**Gain:** The number of model forwards falls by roughly the batch size without changing the metric.

## 2026-07-18 - Batched attention masks could hide real EOS tokens

**Issue:** The first batched implementation inferred attention masks from `input_ids != pad_token_id`.

**Impact:** Models that use EOS as their padding token can include valid EOS tokens inside chat prompts, which would be incorrectly masked.

**Fix:** Build attention masks from each sequence's known pre-padding length instead of token values.

**Gain:** Batched scores preserve the complete chat prompt for models without a dedicated padding token.

## 2026-07-18 - Long BoolQ passages caused full-logit OOM failures

**Issue:** The first full-validation batch run computed vocabulary logits for every token in each prompt. A few long passages made the output tensor exceed the L4's 23 GB VRAM.

**Impact:** Gemma 270M, Qwen 0.6B, Qwen3.5 2B, and Gemma 4 E2B full runs stopped before completion. Their partial JSONL files are invalid and excluded.

**Fix:** Request only the final `candidate_length + 1` logits and use the first candidate-length positions to score the A/B continuation.

**Verification:** Reproduce the known 10-item Gemma result and run a long-passage batch without OOM before rerunning full validation.

**Gain:** Memory use scales with the A/B continuation length rather than the full prompt length at the language-model head.

## 2026-07-18 - Hunyuan RoPE compatibility warning

**Issue:** Hunyuan emits warnings that optional dynamic-RoPE configuration fields are unrecognized by the installed Transformers version.

**Impact:** This may matter for long-context behavior.

**Resolution:** Phase 1 and Phase 2 prompts are far below Hunyuan's native context length, where dynamic RoPE scaling is not activated. The warning is logged and the short-context scores remain valid for this experiment. Do not generalize this setup to long-context evaluations without using Tencent's recommended Transformers build.

**Gain:** The limitation is explicit and bounded to this short-context audit.
