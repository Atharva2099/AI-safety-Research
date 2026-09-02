# Bugs Squashed

Last updated: 2026-09-01

## 2026-09-01 - Surface-text controls were pooled and interpreted incorrectly

**Issue:** The earlier documentation pooled character n-grams, length/tokenization/punctuation features, and terminal token ID into one 61.48% accuracy. These controls measure different hypotheses and should not be averaged. The saved artifact's fold assignments also do not match the current documented `GroupKFold` procedure: 902 of 1,160 English character-n-gram assignments differ.

**Impact:** The documentation incorrectly claimed that the pooled result established that cross-language activation transfer was not decoding surface phrasing. The within-language control could not support that claim, and its fold provenance was not reproducible from the current script.

**Fix:** Added a source-fit character 3–5-gram TF-IDF control using separate training-only vocabularies for every source language and fold. Each fitted classifier was applied unchanged to all six target languages. The artifact saves all predictions, vocabularies, IDF values, coefficients, intercepts, question-ID splits, software version, dataset hash, and target overlap rates.

**Verification:** With scikit-learn 1.7.2 and data hash `95acbb8b293d1e22c054b18fd0b6d058e7420d0a4028a93c89e93fb68c63d170`, all 30 fits converged, all train/test question sets were disjoint, all vocabulary/IDF/coefficient arrays aligned, and every matrix cell contained 1,160 predictions. Mean within-language accuracy was 81.42%; mean cross-language accuracy was 51.67%.

**Gain:** The corrected control distinguishes within-language textual shortcuts from direct cross-language character overlap. It does not claim to test multilingual semantic alignment.

## 2026-09-01 - Qwen diagnostic parity depended on BF16 extraction batch size

**Issue:** The Qwen Layer 12 batch-size-16 diagnostic did not satisfy the strict historical matrix parity gate, despite using the same model revision, data, layer, folds, and probe settings.

**Impact:** The mismatch could have been mistaken for a layer-selection, model-revision, or probe implementation error.

**Fix:** Repeated only the Qwen Layer 12 `current_raw` legacy lane using the historical extraction batch size of 8.

**Verification:** The batch-size-8 run reproduced all 36 historical matrix cells exactly, with zero maximum and mean difference. Comparing batch sizes 8 and 16 changed 580 of 41,760 predictions. All 30 parity probes converged, the artifact hash was verified after synchronization, and the VM was stopped and verified `TERMINATED`.

**Gain:** Qwen's parity discrepancy is explained by numerical sensitivity to BF16 extraction batching. Full eight-condition diagnostics remain at batch size 16 for consistency across all four models.

## 2026-08-24 - Earlier MLP transfer evaluation used an oversized, incorrectly described setup

**Issue:** The previous MLP transfer artifacts and findings described a width-128, LayerNorm/Dropout probe, but the reviewed corrected evaluation instead specifies a frozen width-8 ReLU MLP with source-only preprocessing, inner source validation, fresh outer-training refits, and explicit shuffled-label controls. The earlier evaluation was not used as evidence for the corrected result.

**Impact:** The old width-128 matrices and conclusions were not directly comparable to the reviewed method and could overstate what the experiment measured. Those conclusions are retracted in `src/crosslingual-political-repr/docs/rq2_findings.md`.

**Fix:** Ran the four reviewed model/layer pairs sequentially on the existing single L4 VM with five primary initialization seeds and a bounded label-shuffle control using seed 1729. The corrected artifacts use schema v4, width 8, 6×6 matrices, 150 primary fits, and 30 control fits.

**Verification:** All four JSON artifacts passed schema, finite-value, matrix-shape, fold/seed-count, source-metadata, and control checks. Source-language diagonal comparisons were made against the existing linear baseline before interpreting transfer. The corrected heatmap was regenerated with labels identifying the width-8 method. The VM was synchronized and gracefully stopped; its live status was verified `TERMINATED`. Actual billing remains unreconciled.

**Gain:** The documented MLP comparison now reports only the audited width-8 procedure, its variability and shuffled controls, and model-specific differences from the linear baseline. No claim is made about larger MLPs or representation-level conceptual absence.

## 2026-08-22 - Left-padding corrupted Gemma cross-lingual transfer matrix

**Issue:** `compute_6x6_matrix.py` batched tokenization with the tokenizer's default padding side. `google/gemma-2-9b-it` ships `padding_side="left"`, so in a padded batch real tokens receive RoPE positions shifted by each sample's pad count. The resulting activations were position-scrambled, and every cell of the Gemma 6x6 transfer matrix collapsed to ~52-64% (e.g. en->en at Block 23 read 63.1% instead of the true 86.03% recorded by the unpadded layerwise run). OLMo 3, Qwen 3.5, and Ministral ship `padding_side="right"` and were unaffected.

**Impact:** The annotated heatmap and findings table presented Gemma 2 as near-chance when it is actually among the strongest models; any downstream conclusion built on that matrix was invalid.

**Fix:** Forced `tokenizer.padding_side = "right"` in `compute_6x6_matrix.py` before extraction (with right padding and causal attention, real-token positions match single-sequence inference). Verified with a single-cell sanity check: en->en at Block 23 returned exactly 86.03%, matching the layerwise artifact.

**Verification:** Recomputed the full Gemma matrix on the L4 VM after the fix; diagonal values now agree with `multilingual_probe_google_gemma-2-9b-it_*.jsonl` within fold noise. Heatmap, 3D explorer, and `docs/rq2_findings.md` regenerated from the corrected matrix.

**Gain:** All four model matrices are now directly comparable; the extraction path no longer depends on tokenizer-specific padding defaults.

## 2026-08-19 - GPU capacity failures and migration records were incomplete

**Issue:** The 2026-08-18/19 workspace records did not consistently summarize the Iowa and Virginia L4 stockouts, the successful Oregon migration, the atomic-create behavior, the redundant-asset cleanup, the OLMo 3 layerwise probing activity, and the final graceful termination in one reconciled account.

**Impact:** The current resource state was accurate, but the historical record could incorrectly suggest that no migration or research workload occurred.

**Fix:** Reconciled `workspace_state.md`, the session and cost ledgers, and this entry against the append-only event log and the locally present OLMo artifact. Recorded Iowa (`us-central1`) and Virginia (`us-east4-a`) `STOCKOUT` failures, the Oregon (`us-west1-a`) destination, the verified deletion of redundant resources, the 8.273523-hour L4 session, and the final `TERMINATED` state. The OLMo artifact is recorded as 38,280 records for 580 question IDs across the embedding layer and Blocks 0–31 with five folds; its exact invocation timestamp and actual billing remain unknown.

**Verification:** The event log contains the atomic-create failures, Oregon creation, cleanup, workload session, and graceful stop. Final retained resources are one 100 GB `pd-balanced` boot disk and the `READY` machine image; no snapshots, static IPs, or reservations remain. Actual billing is still unreconciled.

**Gain:** The ledgers now distinguish verified infrastructure facts from missing experiment metadata and do not treat capacity stockouts as VM or disk failures.

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

## 2026-07-28 - Activation-patching driver code existed only on the GPU VM

**Issue:** The Python scripts that produced `results/experiment_a_seed42.jsonl` and `results/full_experiment_seed42.jsonl` in `src/sycophancy-audit/phase_2/activation_patching/` were not present in the git repository. They existed only under `~/sycophancy-audit` on the GPU workspace VM, which was never a git repository.

**Impact:** The two results files could not be reproduced, audited, or debugged from the repository. The file of the same purpose already in the repository (`activation_patching.py`) is a separate, later, non-functional consolidation attempt that does not import correctly and was never run.

**Fix:** Recovered the driver code, chunk scripts, and a pytest test file from the VM over SSH and added them to the repository under `src/sycophancy-audit/phase_2/activation_patching/`, with a provenance note (`RECOVERY_NOTE.md`) in the same directory.

**Verification:** File contents were copied byte-for-byte and diffed against the VM originals before commit; no secret files were included.

**Gain:** The code behind the original activation-patching results is now version-controlled and available for audit.

## 2026-07-28 - Shuffled-source negative control was a silent no-op

**Issue:** In the recovered `full_experiment.py`, the patch span was computed as `patch_end = min(pos + patch_span, len(source), len(target))`. When the randomly chosen "shuffled" source prompt was shorter than the target's patch position, the resulting slice `pos:patch_end` was empty, so the patch wrote nothing.

**Impact:** The negative control - intended to show that patching in an unrelated example's activation does nothing - always reported a margin identical to baseline, regardless of whether the patching mechanism worked. It provided no evidence about whether the reported recovery values (for example, a mean recovery of 0.708 in `full_experiment_seed42.jsonl`) were distinguishable from noise.

**Fix:** Reproduced the bug directly (regression test: `pos=124, patch_end=124, slice_len=0`, patched margin bit-identical to baseline). Subsequent corrected scripts assert the patch slice is non-empty before proceeding.

**Verification:** The regression test confirms the original code produces an empty slice under the documented conditions.

**Gain:** The empty-slice condition is now an explicit, checked failure mode rather than a silent no-op that reads as a passing control.

## 2026-07-28 - Candidate-token log-probabilities computed in bfloat16

**Issue:** `full_experiment.py` and an early diagnostic script (`controls_v2.py`) computed `log_softmax` directly on bfloat16 logits before comparing margins against a tolerance of `1e-3`.

**Impact:** bfloat16 has roughly 0.03 resolution near the log-probabilities used in this experiment, about 30 times coarser than the comparison tolerance. Small real effects and tolerance-based pass/fail judgments could not be distinguished from rounding.

**Fix:** Upcast logits to float32 before `log_softmax` in all subsequent scripts (`a1_discriminator.py` onward), and derive the comparison tolerance from an empirically measured repeat-run spread rather than a fixed constant.

**Verification:** Five repeated unpatched forward passes on the same input produced a bit-identical margin, confirming float32 scoring has no run-to-run noise at the precision used.

**Gain:** Margin comparisons are no longer confounded by scoring precision.

## 2026-07-28 - Discriminating-token score diluted by two shared tokens

**Issue:** The candidate strings `" (A)"` and `" (B)"` tokenize to three tokens each - a shared `"("`, a discriminating letter token, and a shared `")"`. The original scorer summed log-probability over all three tokens per candidate.

**Impact:** The shared tokens contribute equally to both candidates and cancel in the margin, but their presence meant that patches affecting only the shared-token position could appear in raw per-token diagnostics as a large effect, while the single position that actually decides the answer (the letter token) was not scored separately. Combined with the KV-sharing bug described below, this made deep-layer patches (see next entry) impossible to interpret correctly from the summed score alone.

**Fix:** Rewrote scoring to read log-probability only at the discriminating letter token's position, confirmed by tokenizing `" (A)"` and `" (B)"` and checking the token ids differ at exactly index 1.

**Verification:** Token-id check for `google/gemma-4-E2B-it`: `" (A)" -> [568, 236776, 236768]`, `" (B)" -> [568, 236799, 236768]`; index 1 is the sole differing position.

**Gain:** The recorded margin now reflects only the token that determines the model's answer.

## 2026-07-28 - Patch position for deep-layer reachability test was one token before the scored token

**Issue:** An interim diagnostic (and an early draft of a corrected experiment script) defined a "readout" patch position at `len(prompt_ids) + 1` inside the concatenated prompt-plus-candidate sequence, intending to test whether patches at deep layers could ever reach the position that determines the answer. Under causal attention, the discriminating logit is produced from the hidden state at position `len(prompt_ids)`, one token earlier; a position can only be influenced by patches at itself or earlier positions in the same forward pass, never a later one.

**Impact:** Patching at `len(prompt_ids) + 1` could not possibly affect the scored logit, by construction, independent of model architecture. An early result reporting exactly zero recovery at layers 15 and above, and an "extreme perturbation still produces zero effect" diagnostic that appeared to confirm it, were both confounded by this same off-by-one and did not establish anything about deep-layer reachability.

**Fix:** Corrected the patch position to `len(prompt_ids)` (matching the scored position) in `full_experiment_v4.py`, verified by two independent static code reviews before the corrected script was run.

**Verification:** The corrected run (`full_experiment_v4_seed42.jsonl`) shows non-zero, non-null recovery at layers 15, 17, and 20 for both the discovery and heldout example sets - see `docs/sycophancy-audit-summary.md`, Extension A.

**Gain:** The deep-layer reachability question can now be measured; the prior "layers 15+ have zero causal effect" conclusion is retracted as an artifact of the wrong patch position, not a finding about the model.

## 2026-07-28 - Sycophancy patching prompts did not contain a genuine false claim

**Issue:** An interim corrected script (`full_experiment_v2.py`) built the "wrong claim" prompt by re-deriving which letter denotes True/False from the same label used for the claim itself. This meant the True/False lettering changed together with the claim, so the claim was always true relative to its own, possibly relabeled, lettering.

**Impact:** Neither the "clean" nor the "wrong claim" prompt used in that script actually contained a false statement. Results from that script could not be interpreted as measuring sycophancy.

**Fix:** In `full_experiment_v4.py`, the True/False lettering is fixed once from ground truth and held identical between the source and target prompts; only the claimed letter changes between them.

**Verification:** A mechanical check confirms the Choices block is byte-identical between source and target prompts, and the only character that differs anywhere in either prompt is the claimed letter.

**Gain:** The source/target prompt pair now differs only in whether the stated claim is true or false, which is the condition the experiment is meant to test.

## 2026-07-28 - Negative control did not isolate content from token identity

**Issue:** An interim negative control (`full_experiment_v3.py`) patched in a donor activation from an unrelated example's own correct-claim ("source role") prompt. At shallow layers this activation is dominated by the literal claimed-letter token, which is nearly the same token being patched in the real condition.

**Impact:** The control's recovery values tracked the real condition's recovery closely at shallow layers (for example, 0.668 vs. 0.667 at layer 0), so the shallow-layer result could not be attributed to claim-specific content rather than to the act of overwriting the claimed-letter token with any letter token.

**Fix:** In `full_experiment_v4.py`, the negative-control donor is drawn from an unrelated example's own wrong-claim ("target role") prompt, holding the claimed-letter token identity constant between the real and control conditions so only the surrounding content varies.

**Verification:** In the corrected run, the negative control's mean recovery at the shallow layers (0.018, restricted to the position type it is valid for) is clearly separated from the real discovery-flip result at the same position and layers (about 0.23); see `docs/sycophancy-audit-summary.md`, Extension A.

**Gain:** The shallow-layer recovery result can now be attributed to claim-specific content rather than to token overwriting alone. This control has not yet been extended to the deep-layer position; see the open item in the Extension A writeup.

## 2026-07-28 - Two crash-causing dangling references found before any GPU run

**Issue:** During a repair cycle on `full_experiment_v4.py`, static code review (performed before any execution, per this repo's plan-review-run-review discipline) found two places where a dictionary key was read without being guaranteed to exist on every code path: an unused leftover field reference from a prior fix, and a missing check for a "skip" result inside the negative-control loop that would only be reached when a specific rare tokenization condition occurred.

**Impact:** Both would have raised an uncaught exception partway through a run lasting up to roughly 45 minutes of GPU time. The second one specifically would only trigger on a length-mismatched example, making it likely to surface unpredictably deep into a run rather than at the start.

**Fix:** Both dangling references were corrected before the script was executed. Output writing was also changed from a single write at the end of the run to an incremental, flushed write after every record, so a crash from any other unforeseen cause would not lose already-computed results.

**Verification:** A second static code review confirmed both fixes and found no further instance of the same class of bug; the corrected script then ran to completion with zero errors and an exact, independently verified output record count.

**Gain:** GPU time is no longer spent running code that has not been checked for this class of defect, and a crash partway through a long run no longer discards completed work.
