# Bugs Squashed

Last updated: 2026-07-20

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

## 2026-07-18 - Phase 2 JSONL omitted layout-level diagnostics

**Issue:** The first full BoolQ JSONL files saved only counterbalanced neutral and pressure margins.

**Impact:** The combined metric could be reproduced, but original-versus-swapped accuracy and layout-specific pressure effects could not be reported.

**Fix:** Save both layout margins, both layout correctness flags, correct-label mappings, and layout-level aggregate accuracies. Preserve the earlier files as aggregate-only artifacts and rerun the full evaluation under a new layout-diagnostic result prefix.

**Verification:** A smoke test must contain the new per-example fields and summary fields before the full rerun starts.

**Gain:** The Phase 2 report can show both the counterbalanced result and any residual layout asymmetry.

## 2026-07-19 - One raw JD source returned HTTP 410

**Issue:** The selected S07 frontend internship URL returned HTTP 410 Gone during raw collection.

**Impact:** The initial raw collection contains 39 successful pages rather than the planned 40, with the software/frontend slot incomplete.

**Resolution:** Replaced the dead source with a public Heidi Systems frontend listing and reran the raw collector.

**Verification:** The final metadata file contains 40 records with 40 successful HTTP responses and no errors; no failed page is treated as a collected JD.

**Gain:** The collection count and missing source are explicit rather than silently substituting an unrecorded page.

## 2026-07-19 - Official JD pages lacked JobPosting structured data

**Issue:** The first 10 recognizable-company pages from Apple returned official HTML shells without the rendered requirements or `JobPosting` JSON-LD in the raw HTTP response.

**Impact:** HTTP success alone did not mean that a usable JD had been collected.

**Fix:** Replaced those pages with official Stripe, Anthropic, Figma, and GitLab career pages that expose substantive text in fetched HTML. The parser also records visible-text fallback pages with `structured_data_available: false` instead of discarding them.

**Verification:** The final 50-page collection has 50 HTTP-success records, 50 parsed records, and non-empty text for every record. Ten tier-A records use the visible-text fallback; forty tier-B records expose JobPosting structured data.

**Gain:** Source tier and extraction method are explicit before normalization; no missing structured field is filled from inference.

## 2026-07-18 - Hunyuan RoPE compatibility warning

**Issue:** Hunyuan emits warnings that optional dynamic-RoPE configuration fields are unrecognized by the installed Transformers version.

**Impact:** This may matter for long-context behavior.

**Resolution:** Phase 1 and Phase 2 prompts are far below Hunyuan's native context length, where dynamic RoPE scaling is not activated. The warning is logged and the short-context scores remain valid for this experiment. Do not generalize this setup to long-context evaluations without using Tencent's recommended Transformers build.

**Gain:** The limitation is explicit and bounded to this short-context audit.

## 2026-07-19 - JD section headings used unrecognized variants

**Issue:** The four-record parser pilot did not cover heading variants used elsewhere in the 50-page collection, including curly apostrophes, German labels, Figma-specific wording, and labels rendered as ordinary paragraphs or list items.

**Impact:** The initial full run found responsibility evidence in 25/50 records, required-qualification evidence in 28/50, and preferred-qualification evidence in 5/50 despite explicit sections in several misses.

**Fix:** Added only observed heading variants and explicit plain-text boundaries such as `What You’ll Do`, `What You’ll Bring`, `Must Have`, `Muss`, `About You`, and `KEY RESPONSIBILITIES`. Unsectioned prose remains unresolved rather than being semantically guessed.

**Verification:** Reran all 50 records and manually inspected missing-field and high-count outliers. Final coverage is 36/50 for responsibilities, 38/50 for required qualifications, and 11/50 for preferred qualifications.

**Gain:** Explicit source sections are recovered across more page formats while absent or unsectioned fields remain distinguishable from extracted facts.

## 2026-07-19 - JD heading substring match leaked later sections

**Issue:** Broad substring matching treated `Federal Contractor` as an employment type, generic `Office` text as location evidence, and `about your` inside an application-process heading as the `About You` qualification section.

**Impact:** Footer and application-process text could enter employment, location, or candidate-qualification evidence.

**Fix:** Narrowed employment and location patterns, added exact-heading matching for `About You`, and recognized explicit transition boundaries before application and benefits text.

**Verification:** All 50 records pass checks that candidate evidence excludes voluntary-identification, federal-contractor, and OFCCP text. The largest final section counts are 21 responsibilities, 19 required qualifications, and 12 preferred qualifications after manual outlier review.

**Gain:** Extracted evidence remains traceable to job-content sections rather than compliance footers or downstream application text.

## 2026-07-20 - L4 capacity unavailable for the first prompt test

**Issue:** Starting the primary GPU VM in its configured zone failed with `ZONE_RESOURCE_POOL_EXHAUSTED` for one `nvidia-l4` on `g2-standard-4`.

**Impact:** The planned factual-pressure prompt test did not start. No model outputs or metrics were produced.

**Resolution:** A later GUI start request succeeded after capacity changed. No duplicate CLI start command, migration, or configuration change was attempted.

**Verification:** The completed start operation reports the capacity error; follow-up inspection confirmed the VM is `TERMINATED` and the boot disk is `READY`.

**Gain:** The failed request was recorded without treating the stopped VM as broken. Future start attempts must still inspect live capacity and recent operations first.

## 2026-07-20 - Initial prompt-variant sample conditioned on prior pressure outcome

**Issue:** The first 100-item prompt-variant run selected 50 items that previously flipped under the original confirmation prompt and 50 that previously remained correct under it.

**Impact:** The selection rule uses the original confirmation outcome. It can bias the result for that wording and cannot cleanly compare all five prompt versions.

**Fix:** Select the next 100-item sample only from examples that were correct under the neutral Phase 2 prompt. Do not use any prior pressure outcome in the selection rule. Also replace the deprecated `torch_dtype` argument in the new evaluator with `dtype`.

**Verification:** The corrected run must save `selection: {"neutral_correct": 100}` in metadata and complete without the deprecation warning.

**Gain:** Each pressure wording is measured on the same sample selected independently of prior pressure behavior.

## 2026-07-20 - Remote evaluator lacked the source selection file

**Issue:** The VM contained the earlier prompt-variant outputs but not the Phase 2 JSONL file needed to select the corrected neutral-only sample.

**Impact:** The corrected evaluation could not start until the source result was transferred to the VM. No partial evaluation output was treated as a completed run.

**Fix:** Transferred `phase2_layout_google_gemma-4-E2B-it_boolq_validation_seed42.jsonl` to the VM before rerunning the evaluator.

**Verification:** A two-item smoke run completed and recorded `selection: {"neutral_correct": 2}` before the 100-item and 300-item runs.

**Gain:** The corrected runs used a source file that was present and independently checked on the VM.

## 2026-07-20 - L4 capacity unavailable on the retry start

**Issue:** A retry start for the primary GPU VM failed with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` and GPU availability root cause. GCP reported `STOCKOUT` for one `nvidia-l4` on `g2-standard-4` in the configured zone.

**Impact:** The VM remained `TERMINATED`; no GPU or VM compute billing began from this attempt.

**Fix:** No retry was issued. The failed operation was inspected and the VM and boot disk were verified unchanged.

**Verification:** The completed start operation ended at `2026-07-20T23:00:53.369Z`; live instance state was `TERMINATED` and disk state was `READY`.

**Gain:** The capacity failure is recorded without treating the stopped VM or its disk as missing or damaged.

## 2026-07-20 - L4 capacity unavailable on the second retry start

**Issue:** An explicit second retry start failed with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`. GCP reported `STOCKOUT` for one `nvidia-l4` on `g2-standard-4` in the configured zone.

**Impact:** The VM remained `TERMINATED`; no GPU or VM compute billing began from this attempt.

**Fix:** No further retry was issued.

**Verification:** The start command failed and a follow-up instance description returned `TERMINATED`.

**Gain:** The retry result and unchanged VM state are recorded without treating the capacity error as a VM or disk failure.
