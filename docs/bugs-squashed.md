# Bugs Squashed

Last updated: 2026-09-30

## 2026-09-30 - Exact model-cache lookup rejected unused repository files (fixed)

**Issue (FACT):** The source-matrix replay's exact-pinned `snapshot_download(local_files_only=True)`
required a complete model repository, even though the cached Transformers weights and configuration
had loaded in the earlier run. The Qwen3.5 attempt raised `IncompleteSnapshotError` for five absent
repository files before model loading. Read-only checks also found missing alternative weights for
Ministral and bundled package files for Gemma. Evidence: ignored replay diagnostic logs under
`gcp-workspace/`; resolver code in
`src/crosslingual-political-repr/src/multilingual_layerwise_probe.py`.

**Impact (FACT):** The resumed Qwen attempt aborted before extraction and produced no scored transfer
matrix. The guarded wrapper shut down the GPU; a brief diagnostic retrieved the error. Completed OLMo
results remain preserved. No alternate weights or bundled packages were downloaded.

**Fix (FACT):** Resolve `config.json` from the local cache at the exact requested revision, require
its file to exist, and verify that its parent is the matching `snapshots/<revision>` directory. Load the
tokenizer and model from that directory with `local_files_only=True`, preserving the conflicting-config
revision check. No network or branch fallback is introduced. This avoids demanding unused repository
files while retaining the exact snapshot boundary.

**Verification and gain (FACT):** Focused checks passed for an existing, missing and sentinel cache
result; exact, malformed and mismatched snapshot paths; and absent, matching and conflicting config
revision metadata. Syntax and whitespace checks passed, and independent review found no material defect.
The local-only loaders remain in place. Luna verified Qwen3.5 `replay_model_loaded` at
`2026-09-30T08:43:32Z`, after 114.61 seconds, using the exact pinned revision and 33 hidden states.
The process remained active with 17,356 MiB GPU memory and no observed error. This confirms that the
cache-completeness failure is resolved for real model loading. On 2026-09-30, the completed Qwen3.5, Ministral and Gemma artifacts were retrieved and verified:
each has 181,584 prediction rows, 30 source refits, 90 probe arrays, zero convergence warnings and
zero original-score parity mismatches. The driver exited successfully for all three. Local/remote
checksums and input/selection provenance match. Evidence: ignored
`gcp-workspace/rile-v2-source-matrix-resume-20260930T0840Z-retrieval/` model summaries, compressed
predictions, probe archives and driver status. Together with the earlier OLMo result, all four source
matrices are complete. This resolves the runtime failure; no scientific performance gain is claimed.

## 2026-09-29 - CPU-helper SSH blocked pilot artifact retrieval (OPEN)

**Issue:** The existing CPU helper started with the repaired L4 boot clone attached read-only, but its
first IAP SSH attempt returned `4003: failed to connect to backend` before any mount or file read. The
attempt was about 10–15 seconds after the helper start returned. The exact guest-side cause is unknown;
early SSH service readiness is a hypothesis, not a confirmed diagnosis. Earlier IAP access to the same
helper worked, and the effective firewall still permits TCP/22. Evidence: ignored
`gcp-workspace/workspace_log.jsonl` and `gcp-workspace/workspace_sessions.csv`.

**Impact:** The completed pilot's result JSON, layer agreement, and sampled memory measurements remain on
the stopped disk and unverified locally. The helper retrieval session used 0.048571 CPU VM-hours and no
GPU time. No RQ1/RQ2 scores were produced.

**Fix and verification (OPEN):** The helper was stopped, and the clone was restored to the stopped L4 as
its boot disk with `autoDelete=false`; the original disk remains untouched. A bounded retry after the
helper has had time to boot, with live serial inspection, is pending fresh approval under the ignored
`gcp-workspace/SKILL.md`. Both VMs were verified off; the pilot artifacts were not retrieved.

## 2026-09-28 - Worker pilot stopped before SSH or model work (fixed 2026-09-29)

**Issue:** An approved two-hour worker pilot first met a zone GPU stockout. A later start of the existing
VM ended in a guest-initiated shutdown before SSH or model work. At discovery, the cause was unknown; an
expired persistent timer was only a possible cause. See the ignored `gcp-workspace/` operation log and
session and cost ledgers for the full infrastructure record.

**Impact:** The worker pilot produced no model timing or scored RQ1/RQ2 evaluation. Its closed session
recorded 0.033195 VM/GPU-hours. The VM was verified terminated; retained storage continues to bill.

**Initial fix plan (SUPERSEDED):** Inspect the guest shutdown and timer state before another worker pilot.
No repair resources were created in this attempt; the cause had not yet been established.

**Verification:** The operation record shows the initial stockout and later guest-initiated shutdown;
the session ledger records the elapsed VM/GPU-hours and terminated end state. There was no SSH or model
work in the worker pilot.

**Gain:** Recording the interrupted session separates its cost and missing timing evidence from the four
completed diagnostic convergence pilots documented in `src/crosslingual-political-repr/docs/requirements.md`.

**2026-09-29 diagnosis (FACT; fix OPEN):** Read-only inspection of an independent boot-disk clone found two
enabled `Persistent=true` hard-stop timers with deadlines before the failed start. The guest journal records
both timers and their power-off services starting before systemd shut down and GCE logged `guestTerminate`.
This establishes the expired timers as the shutdown cause. The original VM and disk were not changed; the
clone was not repaired. Evidence: ignored `gcp-workspace/rile-v2-offline-diagnosis-20260928/diagnostics.txt`
and the closed helper session in `gcp-workspace/workspace_sessions.csv`.

**2026-09-29 repair and verification (FACT):** On the clone, the two expired timer units were masked and
the clone replaced the stopped VM's boot disk. The original disk remains untouched and unattached for
rollback. A guest check found the timers masked and inactive; CUDA worked. The bounded English-only
Ministral worker pilot then completed all 244 classifier fits with no convergence warnings. The L4 was
stopped after the pilot, before its GCE-side two-hour STOP cap. The result JSON is still on the stopped
boot disk, so layer agreement and memory measurements remain to be retrieved and checked. This repaired
the premature shutdown; it did not produce RQ1/RQ2 evaluation scores. Evidence: ignored
`gcp-workspace/workspace_log.jsonl`, `gcp-workspace/workspace_sessions.csv`, and the pilot's remote
progress log recorded there.

## 2026-09-27 - GPU driver unavailable after existing VM restart (fixed 2026-09-28)

**Issue:** The existing GPU VM booted into a kernel without a matching NVIDIA module. The previously installed kernel has a matching driver module. See the ignored `gcp-workspace/` diagnostic record and session ledger from the bounded RILE-v2 pilot attempt.

**Impact:** The corrected RILE-v2 convergence and timing pilot stopped before model loading or data processing. No GPU runtime or convergence result was produced.

**Initial fix status:** Open at discovery. A one-time boot into an older installed kernel was the first repair candidate; the later verified fix is recorded below.

**Verification:** The guest reported no NVIDIA module for its current kernel and `nvidia-smi` could not contact the driver. The VM was verified terminated after the diagnostic session. A separate CPU-only synthetic check exercised four grouped folds and four logistic fits; it does not measure the RILE data or GPU speed.

**Gain:** The pilot cannot silently proceed without a working GPU, and its failed session is recorded in the infrastructure ledger.

**2026-09-27 update — approved recovery failed:** One approved attempt to boot the already installed kernel with a matching driver used a one-time GRUB entry. Two guest reboots returned to the driverless kernel; the GPU checks still failed. The VM was stopped without transferring data or running a model. The default boot setting remained unchanged, but a pending one-time GRUB entry could not be cleared after SSH stopped responding. The next VM session must inspect that entry before another boot repair. See the ignored recovery diagnostics and closed infrastructure session ledger. The issue remains open.

**2026-09-27 update — exact module has broader dependencies:** The pending one-time boot entry was clear on the next start. A read-only package-manager simulation found a module matching the active kernel, but installing it would also upgrade 16 existing NVIDIA packages and add a firmware package (18 package changes total, no removals). No package was installed because that transaction exceeded the approved narrow repair. The VM was verified terminated and the session closed in the ignored infrastructure ledger. The RILE-v2 GPU pilot remains unrun.

**2026-09-28 fix and verification:** After a READY pre-change disk snapshot was created, the guest's unattended updater installed the matching NVIDIA driver stack. The agent did not invoke that package transaction. A reboot then made the L4 visible to `nvidia-smi`, and PyTorch reported CUDA available. Four bounded RILE-v2 convergence pilots subsequently completed on that GPU; see `src/crosslingual-political-repr/docs/requirements.md` for their diagnostic settings and results. The VM was stopped after automatic approval review rejected a longer timer for the separate full evaluation. The driver issue is fixed; the four-model evaluation remains pending.

## 2026-09-27 - GPU capacity stockout delayed the recovery attempt (open)

**Issue:** A start of the existing GPU VM failed because the zone had no available L4 capacity. See the ignored infrastructure operation log.

**Impact:** The failure delayed the approved recovery and pilot; it did not run a VM or process data.

**Fix:** A single retry after a ten-minute wait started the same VM. Future starts may still face a stockout, so capacity remains an open operational risk.

**Verification:** The failed operation reported a stockout and the VM remained terminated; the later start reported the same VM running. The recovery session then ended with the VM verified terminated.

**Gain:** No second VM or zone was used, and the failed start was recorded as zero running hours.

## 2026-09-26 - RILE builder admitted excluded subcategories and duplicated texts (open)

**2026-09-27 update — dataset candidate built:** `src/finalize_rile_translations.py --v2` produced
`data/rile_v2/` from the finalized original file. The builder and sampler now reject exact CMP 202.2,
605.2, and 703.2 before mapping to parent categories. The new candidate removes 30 invalid-code rows,
both members of one conflicting duplicate pair, and one redundant member from each of four consistent
pairs: 6,131 items remain (5,044 development, 1,087 category holdout). See the local manifest and
`src/crosslingual-political-repr/docs/dataset_manifesto_rile.md` §13.2 for provenance. **Open:** new
probe scores, convergence, tone and lexical controls, and category-versus-document evaluation design.

**Issue:** The current RILE file has 30 items drawn from CMP subcategories 202.2 (one left item) and 605.2
(29 right items), which the MPDS2024a RILE definition excludes. The builder at
`src/crosslingual-political-repr/src/build_rile_set.py:49-55` strips the subcategory suffix before checking
membership, while the sampler at `src/crosslingual-political-repr/src/sample_corpus.py:37-50` can keep those rows. A
normalized exact-text audit also found duplicate groups across languages and development/heldout boundaries,
including one English text with conflicting CMP 407/107 labels. Evidence: the final local dataset audit;
the authoritative definition is the [MPDS2024a codebook](https://manifesto-project.wzb.eu/down/data/2024a/codebooks/codebook_MPDataset_MPDS2024a.pdf),
pp. 10 and 30.

**Impact:** The current 6,167-row file, its tone check, and its activation scores are provisional. Mapping-only
exclusion implies 6,137 rows, but duplicate handling may change that count. The category-heldout split shares
manifestos with development, so it cannot support an unseen-manifesto claim. Development CV does exclude
manifesto IDs across folds, but nine parties recur across separate manifestos in different folds, so it does
not establish unseen-party generalization. Duplicate groups also cross development folds and must be resolved
before interpreting CV as leakage-free. Previously inspected heldout scores are exploratory.

**Proposed fix:** Build a separately versioned dataset excluding the invalid subcategories, preserving valid
translations and prior approved drops, and resolve exact duplicates as groups before assigning evaluation
folds. Recompute counts and tone diagnostics. Use nested manifesto-grouped development evaluation for layer
selection, then a clearly labeled category-heldout robustness check. A genuinely untouched confirmation would
require reserved source documents before model and analysis choices are frozen.

**Verification:** The corrected dataset candidate is built and checked against its source: 6,131 retained
rows, six nonempty aligned texts per row, metadata and texts unchanged for retained IDs, no excluded
subcodes or normalized duplicate texts in any language, and both prior drops absent. RILE-v1 source remains
unchanged. No corrected evaluation has run; its score and tone issues remain OPEN. The 6,137 figure is only
the mapping-only intermediate count.

**Gain:** Recording the mapping and split limitations prevents current diagnostic scores from being described
as clean confirmatory evidence.

## 2026-09-16 - Spanish tone-matching passed only after changing the matching model (open caveat)

**Issue:** The Manifesto Spanish RILE set, tone-matched on `cardiffnlp/twitter-xlm-roberta-base-sentiment` like English and German, failed the pre-declared validator gate: worst independent validator 0.051 against a 0.05 threshold, with `nlptown` at 0.551 and `pysentimiento/robertuito-sentiment-analysis` at 0.537, both above chance in the same direction. Rebuilding matched on robertuito instead passes at 0.047.

**Impact:** Two consequences, both reportable. The pipeline is no longer uniform across languages: English and German match on cardiff, Spanish on robertuito. And the passing margin is thin (0.047 against 0.05), on the smallest set (2,153 per side).

**Fix:** Added `--match-model` to `src/build_rile_set.py` so the matching model can be one competent in the target language. Spanish is built with `--match-model spanish`.

**Verification:** Gate after rebuild: robertuito 0.499 (matched on), cardiff 0.453, nlptown 0.517, xlmr_multi 0.464; worst independent deviation 0.047. English (0.019) and German (0.006) pass matched on cardiff.

**Gain:** None claimed. This is recorded as an open caveat. Both the failing cardiff-matched result and the passing robertuito-matched result belong in any writeup; reporting only the second would be selection.

## 2026-09-16 - Tone scoring truncated every model at 512 tokens

**Issue:** `src/tone.py::score_texts` passed a fixed `max_length=512`. `pysentimiento/robertuito-sentiment-analysis` has `max_position_embeddings` 130, so scoring raised `RuntimeError: index 130 is out of bounds for dimension 1 with size 130`.

**Impact:** The Spanish validator gate aborted partway, after scoring two of four models.

**Fix:** Cap per model at `min(512, max_position_embeddings - 2)`, leaving room for the special tokens that RoBERTa-family position offsets consume.

**Verification:** The Spanish gate then completed across all four models. No English or German number changed, since those models have 512 positions.

**Gain:** Tone models with short contexts work without special-casing.

## 2026-09-16 - Country is not language: a third of the Spanish set was Catalan

**Issue:** Manifesto country codes select countries, not languages. Country 33 (Spain) includes Catalan, Valencian and Galician parties. `langdetect` over the built Spanish set found 1,906 of 6,010 sentences (31.7%) were Catalan, mostly from Catalan Republican Left (707, 0% Spanish), In Common We Can (714, 0%) and Together for Catalonia (492, 1%), plus 55 Portuguese/Galician. The English set contained 44 French sentences from Bloc Quebecois (0.8%); German was 99.8% clean.

**Impact:** Language is the variable under test in a cross-lingual transfer experiment, so a third of one language's data being a different language would have confounded the result.

**Fix:** Added a seeded per-sentence `langdetect` filter to `src/build_rile_set.py`, applied before capping and tone-matching so class balance still holds afterwards.

**Verification:** Filter kept 37,576 of 37,883 (en), 18,149 of 18,178 (de), 7,136 of 10,463 (es). Rebuilt matched sets: en 6,197 per side, de 5,339, es 2,153.

**Gain:** Each language set now contains one language. Spanish is a third smaller, which is the honest size of the usable Spanish data.

## 2026-09-16 - Left/right keying was invented before being checked against the codebook

**Issue:** The first Manifesto design keyed left versus right from opposing category pairs (601/602, 603/604, 203/204, 701/702) on recalled knowledge of the RILE index. The Manifesto Project Dataset codebook (MPDS2024a, s3.6 "Programmatic dimensions", p.30) defines RILE as an additive index over specific categories, and 602, 604, 204 and 702 appear in neither the right nor the left list.

**Impact:** Four of the six planned issue pairs would have been keyed on a scheme of our own invention while being described as standard. Caught before any probe was trained.

**Fix:** Adopted the verified formula, quoted in the `src/build_rile_set.py` docstring, and dropped categories outside it.

**Verification:** Category counts recomputed under the verified lists: 16,866 right and 21,017 left sentences across 40 English manifestos, with 29,705 usable sentences falling outside RILE entirely.

**Gain:** The axis matches a published, externally defined index instead of a keying we chose.

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


## 2026-09-30 - RILE text-control compatibility and stdout logging

**FACT — Issue:** The older text-surface-control script targets an earlier survey cohort and split procedure. Its saved results and accuracy metric do not supply the matched RILE v2 manifesto-fold comparison. This is a compatibility issue for the new dataset, not a finding that the older survey split was incorrect. Sources: [older runner](../src/crosslingual-political-repr/src/run_text_surface_controls.py); [RILE control runner](../src/crosslingual-political-repr/src/run_rile_text_controls.py).

**FACT — Impact and fix:** Added a separate small RILE runner using the exact saved OLMo item-to-fold map, source-training-only TF-IDF/scaling, and pooled balanced accuracy; the old study script remains unchanged. The fixed run contains character 3–5-grams, word 1–2-grams with language-specific preprocessing, and 11 length/final-punctuation features. Sources: [run summary](../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json); [dataset report §15](../src/crosslingual-political-repr/docs/dataset_manifesto_rile.md).

**FACT — Verification and gain:** The 90 fits completed with no convergence warnings. The 108 cells each contain 5,044 unique aligned items from 66 manifesto groups; bootstrap cell point checks match summaries to 1e-12. Saved predictions provide matched lexical/surface measurements on the existing RILE folds. Sources: [run summary](../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json); [bootstrap runner](../src/crosslingual-political-repr/src/bootstrap_rile_text_controls.py); [bootstrap results](../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json).

**OPEN — Terminal-token identity:** The older scalar numeric token-ID control treats token identifiers as ordered quantities. It was excluded from this RILE run; categorical token identity with separately defined tokenization has not been implemented or evaluated. Source: [older runner](../src/crosslingual-political-repr/src/run_text_surface_controls.py).

**LIMITATION — Stdout log:** A full-run `tee` log attempt targeted a missing parent directory and did not save a persistent stdout log. Live completion output was observed; the completed summary retains per-fit records, iterations, warnings, runtime, and prediction hashes. No rerun was performed solely for logging. **OPEN mitigation:** Create the destination parent before a future `tee` invocation. This mitigation was recorded, not retroactively tested. Source: [run summary](../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json).

## 2026-09-30 - Category-holdout GPU transfer approval (open)

**OPEN — Issue (2026-09-30):** Automatic approval review rejected the proposed SCP export of nonpublic code, translated statements, predictions, and fitted probes because the existing task authorization did not explicitly authorize that export. Source: automatic approval review response in the current work session (2026-09-30).

**INCOMPLETE — Impact (2026-09-30):** The four GPU model evaluations have not launched. No bundle transfer or workaround occurred; the category-holdout campaign remains partial, with only the three CPU controls completed. Sources: current work-session execution record (2026-09-30); [partial bootstrap JSON](../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`status=PARTIAL_CONTROLS_ONLY`, `families`).

**OPEN — Mitigation (2026-09-30):** The owner subsequently explicitly authorized the private transfer and evaluation after the push to main. The prepared bundle is retained for that sequence; execution remains unverified, so this issue is not recorded as fixed. Source: current work-session approval request and subsequent owner authorization (2026-09-30).

**FACT — Completed verification and gain (2026-09-30):** The CPU evaluation finished in 36.351 seconds with 90 frozen-fit hash matches and zero fit warnings. Each control saved 39,132 predictions, with 1,087 unique aligned items per cell. The partial analysis checked 108 holdout and 108 full-development cell points against summaries to 1e-12, then recomputed development on the same 58 documents (4,901 items) for descriptive paired-bootstrap differences. These saved controls provide an auditable partial result while GPU execution is blocked. Sources: [CPU summary](../gcp-workspace/rile_v2_category_holdout/full-20260930-luna/controls/summary.json); [partial bootstrap JSON](../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json); [dataset report §16](../src/crosslingual-political-repr/docs/dataset_manifesto_rile.md).

**FACT — Transfer approval resolved (2026-09-30):** Following explicit owner authorization, the private transfer completed with all 23 remote hashes matching the local transfer manifest. The transfer-approval issue above is resolved; the campaign remains incomplete because a subsequent OLMo smoke failed development parity. Sources: [local manifest](../gcp-workspace/rile_v2_category_holdout/bundle-20260930-luna/local_relative_sha256.txt); [returned manifest](../gcp-workspace/rile_v2_category_holdout/bundle-20260930-luna/remote_sha256.txt); [failed smoke summary](../gcp-workspace/rile_v2_category_holdout/diagnostic-20260930-luna/summary.json); current work-session owner authorization (2026-09-30).

## 2026-09-30 - Category-holdout development parity and batch size (open)

**FACT — Issue and impact (2026-09-30):** The OLMo GPU smoke stopped with `needs_review` after 101.080 seconds on its first reported development-margin parity failure: absolute error 0.0211628636. Its metadata cohort had 20 holdout smoke items, but no accepted model holdout result was saved. Full OLMo and later model evaluations were not launched. Sources: [failed smoke summary](../gcp-workspace/rile_v2_category_holdout/diagnostic-20260930-luna/summary.json); current work-session execution record (2026-09-30).

**FACT — Setting mismatch and repair (2026-09-30):** The failed evaluator hardcoded extraction batch size 4. All four original selection summaries record 8, and the source-replay validator requires that recorded setting. The repaired evaluator verifies each original selection-summary hash and configuration, derives batch size 8, records its provenance, and preserves the first eight development rows as one original batch. Sources: [dataset report §16.6 with the four original summaries](../src/crosslingual-political-repr/docs/dataset_manifesto_rile.md#166-olmo-smoke-failure-and-recorded-batch-size-correction-2026-09-30); [source replay helper](../src/crosslingual-political-repr/src/multilingual_layerwise_probe.py) (`replay_source_layers`); [repaired evaluator](../src/crosslingual-political-repr/src/evaluate_rile_category_holdout.py) (`selection_config`, `model_run`).

**HYPOTHESIS / OPEN — Cause (2026-09-30):** The mismatch may explain the parity failure. GPU verification is pending; the issue is not recorded as fixed. No selected layers, fitted probes, labels, folds, thresholds, feature precision, or tolerances were changed. Source: [repaired evaluator](../src/crosslingual-political-repr/src/evaluate_rile_category_holdout.py); current work-session repair decision (2026-09-30).

**FACT — Local verification and next check (2026-09-30):** All four original selection-summary hashes and batch-size values matched the frozen replay records. Model/count/hash mismatch fixtures were rejected; syntax passed; importing CPU code did not load Torch. The next check is a new GPU smoke using the recorded batch size and unchanged parity gate, before full model evaluation. The gain so far is correction of an evidenced configuration mismatch, not confirmation that the GPU failure is resolved. Sources: [repaired evaluator](../src/crosslingual-political-repr/src/evaluate_rile_category_holdout.py); current work-session local-check record (2026-09-30).

**FACT — Parity repair verified; current issue status resolved (2026-09-30):** All four corrected GPU smokes and full runs completed using the hash-verified recorded batch size 8. Every smoke and full summary records 288 development-parity checks, exact predictions, and maximum absolute margin error 0.0, with the unchanged `atol=0.001`, `rtol=0.0001` gate. This supersedes the pending-verification status above. The earlier OLMo failure and its diagnostic logs remain preserved; successful verification does not establish a causal explanation for the earlier margin difference. Sources: [OLMo smoke](../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/OLMo/summary.json), [Qwen3.5 smoke](../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/Qwen3.5/summary.json), [Ministral smoke](../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/Ministral/summary.json), [Gemma smoke](../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/Gemma/summary.json); [OLMo full](../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/OLMo/summary.json), [Qwen3.5 full](../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Qwen3.5/summary.json), [Ministral full](../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Ministral/summary.json), [Gemma full](../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Gemma/summary.json) (`status`, `batch_size`, `dev_parity`); [preserved failed summary](../gcp-workspace/rile_v2_category_holdout/diagnostic-20260930-luna/summary.json).

**FACT — Campaign completion update (2026-09-30):** The transfer-approval issue was resolved by explicit authorization and verified transfer, as recorded above. The four model evaluations and seven-family bootstrap are now complete; earlier partial and unlaunched snapshots are superseded by this update. The completed report preserves all 252 held-out source–target measurements and descriptive comparisons. Sources: [corrected driver status](../gcp-workspace/rile_v2_category_holdout/diagnostic-batch8-20261001-luna/driver_status_batch8_v4.tsv); [complete bootstrap](../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json); [dataset report §16.7](../src/crosslingual-political-repr/docs/dataset_manifesto_rile.md#167-completed-seven-family-category-holdout-2026-09-30).

## 2026-09-30 - Category-holdout wrapper selection-hash typo (resolved)

**FACT — Issue and impact (2026-09-30):** The GPU wrapper contained a one-character typo in the expected Ministral original-selection-summary hash. The preflight check rejected it before model evaluation. This was a wrapper constant mismatch; the selection artifact and frozen evaluator were unchanged. Sources: [earlier wrapper](../gcp-workspace/rile_v2_category_holdout/batch8-selection-metadata-20260930-luna/run_driver_batch8.sh); [earlier preflight wrapper](../gcp-workspace/rile_v2_category_holdout/batch8-selection-metadata-20260930-luna/run_driver_batch8_v3_preflight.sh); current work-session preflight execution record (2026-09-30).

**FACT — Fix and verification (2026-09-30):** Corrected the expected hash to the artifact's recorded SHA-256 and verified preflight before launching the same frozen evaluator and helper. The corrected driver checked all four selection-summary hashes and batch-size values, then recorded completion of all four smokes and full evaluations. No scoring, probes, layers, labels, or parity tolerances changed for this wrapper repair. Sources: [corrected preflight wrapper](../gcp-workspace/rile_v2_category_holdout/batch8-selection-metadata-20260930-luna/run_driver_batch8_v4_preflight.sh); [corrected driver](../gcp-workspace/rile_v2_category_holdout/diagnostic-batch8-20261001-luna/run_driver_batch8_v4.sh); [driver status](../gcp-workspace/rile_v2_category_holdout/diagnostic-batch8-20261001-luna/driver_status_batch8_v4.tsv); [completed model/analysis provenance](../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json).

**FACT — Gain (2026-09-30):** The intended preflight integrity check passed and the completed evaluation artifacts were retained. Failed diagnostics remain preserved; this entry records a verified wrapper repair rather than a numerical-model finding. Sources: corrected driver status and complete bootstrap cited above; [earlier failed diagnostic](../gcp-workspace/rile_v2_category_holdout/diagnostic-20260930-luna/summary.json).

## 2026-09-30 - Category error-analysis draw masking and partition detail (fixed)

**FACT — Issue and impact:** The category confidence-interval code allocated space for every shared bootstrap draw before assigning only draws with category items. The frozen four-category data had all 10,000 draws valid, so current point estimates and intervals were unaffected; a missing-category draw would have caused a shape error instead of being excluded. The correctness partition also reported only category-wide counts over all 36 cells, omitting the requested cellwise, diagonal, and cross-language counts.

**FACT — Fix and verification:** Allocate category draw matrices for valid draws only; a three-draw in-memory fixture confirms an absent-category draw is excluded from the interval. Add one 6×6 matrix for each of four correctness partitions per category, plus 36-cell, diagonal-6-cell, and cross-language-30-cell totals. Assertions verify each cell sums to its category item count, the diagonal and cross-language totals sum to six and 30 times that count, respectively, and each matrix reproduces its prior category-wide total. Recomputed all category and pairwise point estimates and intervals matched the preserved first pass within 1e-12; all 34,514 decompressed candidate rows were byte-identical. Sources: [analysis runner](../src/crosslingual-political-repr/src/analyze_rile_holdout_errors.py); [repaired JSON](../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/error_analysis.json); [repaired HTML](../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/tables.html); preserved first-pass JSON in the sibling `error-analysis-20260930-luna/` folder.

**FACT — Gain:** The revised analysis handles categories absent from some draws and retains the requested partition counts by cell and language grouping. The repaired run completed in 2.02 seconds. No predictions, labels, categories, bootstrap weights, metrics, or model fits changed.
