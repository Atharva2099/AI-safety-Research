# Project Requirements & Architecture

Last updated: 2026-09-29

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

## 2. Historical Candidate Models (not the current experiment roster)

This list is retained as an earlier planning snapshot, not a specification of the current run. The RILE
v1 experiment uses the four model checkpoints documented in `docs/dataset_manifesto_rile.md` and the
runner configuration. The earlier names below include outdated model generations and must not be used to
identify current results.

- **Qwen 3.5 (9B):** Primary target for broad multilingual support (>100 languages).
- **Gemma 4 (12B):** Strong open-weight candidate with wide multilingual pre-training.
- **Ministral 3 (8B) / NeMo (7B–12B):** Representative European model family.
- **OLMo 3 (7B):** Fully open pre-training architecture for comparison.

The current RILE v1 roster in `src/multilingual_layerwise_probe.py` is `allenai/Olmo-3-7B-Instruct`,
`mistralai/Ministral-8B-Instruct-2410`, `google/gemma-2-9b-it`, and `Qwen/Qwen3.5-9B`. The fixed old-layer
reference uses Blocks 17, 31, 15, and 14 respectively. Record each resolved model revision with results;
the repository name alone may point to changing weights.

---

## 3. Target Languages

- **English (en):** Primary reference language.
- **Spanish (es) & German (de):** Western high-resource target languages.
- **Hindi (hi):** South Asian non-Western target language.
- **Marathi (mr):** South Asian non-Western target language.
- **Mandarin Chinese (zh):** East Asian non-Western target language.
- **Japanese (ja):** East Asian non-Western target language.

Japanese is a historical candidate only. The current RILE v1 design has six languages: en, de, es, zh,
hi, and mr.

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

---

## 6. RILE v1 design audit and proposed analysis (2026-09-26)

This is a design record, not evidence that the analysis is complete, preregistered, or ready to publish.
It supersedes the older synthetic-question framing above for the current RILE campaign. The target is
whether a linear classifier can read **Manifesto RILE-category-linked text labels** from frozen model
representations across languages. It does not measure model beliefs, a universal left/right ideology, or
the causal effect of language. Comparisons with the earlier survey-derived corpus are descriptive because
the text, labels, sampling, and language construction changed.

**RILE-v1 dataset blocker — resolved in the RILE-v2 candidate.** The original 6,167-item file included 30
items through invalid RILE subcategories: one CMP 202.2 item and 29 CMP 605.2 items, all in development.
The builder strips the suffix before checking membership (`src/build_rile_set.py:49-55`), while the sampling
script can retain those subcategories (`src/sample_corpus.py:37-50`). The Manifesto codebook defines the
RILE aggregate from the parent categories and excludes 202_2 and 605_2 from its component lists (MPDS2024a,
pp. 10, 30: [codebook](https://manifesto-project.wzb.eu/down/data/2024a/codebooks/codebook_MPDataset_MPDS2024a.pdf)).
RILE-v2 removes those items and resolves the identified exact-text duplicate groups, preserving valid
translations and earlier approved drops. It has 6,131 items (5,044 development; 1,087 category-heldout).
Its manifest and dataset note document the filtering and duplicate rules. Four bounded RILE-v2
convergence pilots have run (§6); no full RQ1/RQ2 outer-test evaluation has run.

**Other dataset checks — OPEN.** The normalized exact-text audit (`NFKC`, strip, casefold) found duplicate
groups in RILE-v1: English (two groups/four rows) and one group/two rows in each of German, Chinese, Hindi,
and Marathi. One English duplicate crossed development and category-heldout and had conflicting CMP 407/107
labels within one manifesto; English and Hindi also had duplicate groups split across original development
CV folds. RILE-v2 removes or resolves those groups according to its recorded rule. The corrected corpus still
requires substantive review of labels, category coverage, translations, and potential lexical/style shortcuts.
The category-heldout set contains four categories across the same 58 manifestos as development, so it is a
new-category robustness check, not an unseen-manifesto test. Nine parties have different manifestos in two
development CV folds (18 manifestos); do not claim unseen-party generalization. Source counts are uneven
(en 2,113; de 2,056; es 1,998 native, with zh/hi/mr translated). The existing native-versus-translated
contrast is confounded by source country, party, topic, and translation; describe it only as exploratory.
The six-language final file and sampling/build code are the evidence sources; do not reproduce raw text or
row identifiers in public documentation.

**RQ1 and layer selection.** Freeze each base model and extract the same documented final-token residual
representation from each layer. Fit a logistic regression probe separately by language and layer. For the
corrected primary analysis, keep raw features and `C=1` fixed; do not tune regularization in the primary
evaluation. First run a bounded development-only optimization check on the same examples, folds, features,
preprocessing, objective, and `C`, increasing only solver effort. Record convergence and compare prediction
and score stability. Select the highest inner-validation balanced accuracy, with earliest layer breaking an
exact tie. Report per-language inner-validation layer curves as descriptive selection evidence; only
the selected layer receives an outer-test score. The corrected performance estimate uses nested
manifesto-grouped CV: five outer folds; within each outer-training partition, use grouped inner validation to
select a layer separately for each model and language; refit that choice on the full outer-training partition;
score once on the excluded outer-test manifestos. Use identical outer manifesto groups across languages.
This estimates the layer-selection procedure on new documents; it is not an independent locked test after
choices informed by the same campaign. Report accuracy, F1, and label counts as supporting metrics.

**RQ2 and old results.** Primary transfer is English-to-each-of-five-target-languages at the English layer
selected independently inside each outer training fold by nested RQ1. Apply the same selected English layer
to every target language in that fold and use aligned outer-test item IDs. Fit a target-language comparator
on the target versions of those same training items, at that selected layer, then evaluate on the same target
test rows. The primary summary is the equal-weighted mean of five target balanced accuracies. Use paired,
manifesto-level uncertainty intervals, resampling aligned translations together; also report party-cluster
sensitivity because parties recur across manifestos. The full 6×6 directed matrix is secondary and must use
the source language's inner-selected layer for each outer fold, applied to all targets. Reuse each fitted
source probe across targets. Historical English-selected layers (OLMo 17, Ministral 31, Gemma 15, Qwen 14)
are a separately named secondary diagnostic. Do not pool old and corrected-corpus scores. Report model
versions, extraction settings, folds, solver `n_iter_` and convergence, split summaries, and row predictions or
per-manifesto confusion counts sufficient to recompute the metrics.

**Split and layer interpretation (2026-09-27; agreed future analysis, not run).** Every sampled sentence
has a source `manifesto_id`, a Manifesto category code, a category-derived left/right label, an original
source language, and aligned text in six languages. The manifesto ID identifies the source document; it is
not a prediction label. Current five-fold development evaluation separates manifesto documents, while
the same 22 categories occur in both training and validation in every fold. Because these validation scores
also identify the reported peak layer, the best-layer score is descriptive rather than an independent
new-document result. The separate four-category evaluation excludes those categories from training, but
its sentences come from manifestos also represented in development; it tests category transfer within
known documents. Category-derived labels make topic recognition a plausible shortcut. Report these as two
different questions, with denominators and category counts, and obtain an independent new-document
evaluation of the layer-selection procedure before claiming new-document performance.

RQ1 gives a development peak for each model and language. For English-to-other-language RQ2, selecting
the English source layer using training-side data is legitimate; selecting it using the evaluation fold
would bias the reported score. The current RQ2 instead uses English peaks chosen on the earlier survey
corpus, so it does not yet measure transfer at newly selected manifesto-data peaks. In the corrected run,
select an English layer inside each outer training fold and evaluate that choice on its excluded manifestos;
report each language's RQ1 peak separately. If reporting transfer from another source language, select that
source language's layer inside its training partition and apply the same layer to every target. Keep the
historical fixed-layer transfer as a clearly named secondary comparison. Selecting a target language's peak
from its evaluation labels would answer a different, target-informed question. Previously inspected
category-heldout scores remain exploratory.

**Controls and interpretation.** Before interpreting activation scores, compare a majority-class baseline,
a source-fit lexical character/word baseline applied unchanged to every target, and simple style features
(length, punctuation, negation, plus suitable independent sentiment scores). Include an item-aligned label
shuffle as a pipeline sanity check, not as proof that confounds are absent. Use the existing opposite-category
issue pairs (military 104/105, welfare 504/505, protectionism 406/407) only as within-issue diagnostics;
their labels still proxy category membership, and Spanish welfare is highly imbalanced (4 right, 129 left).
Audit examples before naming these comparisons as stance tests. Previous notes report potentially problematic
military examples; inspect the current sample rather than assuming every pair is invalid. A bilingual human
review of stratified source text and translations is needed before strong claims about political stance or
translation fidelity. Existing model-only tone AUCs are limited diagnostics, not evidence that tone is absent.

**Publication boundary and next step.** After fixing data, first repeat a bounded development-only check while
holding representations, preprocessing, regularization `C`, objective, examples, and folds fixed. Increase only
optimization effort; record convergence status and iteration counts, and check whether predictions and scores
stabilize. This diagnoses optimizer truncation without changing the measurement. If scaling or another solver
is considered, evaluate it afterward as a separate method comparison on the same examples and folds; do not
silently change preprocessing or solver as a repair. Then run nested grouped development evaluation and one
category-heldout robustness check. The current heldout results have
already been inspected during this campaign, so label them exploratory; a new untouched confirmatory claim
would require reserving unseen source documents/manifests before choices are frozen. The Manifesto terms
[prohibit redistribution of the corpus](https://manifestoproject.wzb.eu/information/documents/terms_of_use)
without written permission. Confirm the permitted scope for derived labels, translations, and metadata before
sharing an archive; do not infer that text-free files are automatically exempt. These are design repairs and
open questions, not a claim of arXiv or blog readiness. Do not start another major run from this record alone.
Public manifesto documents may have appeared in a base model's pretraining; heldout probing does not establish
that the model itself has never seen the source text.

**Corrected evaluation status (2026-09-28).** RILE-v2 contains 6,131 items (5,044 development; 1,087
category holdout). The runner now implements `--protocol nested_source_peak` for the grouped RQ1/RQ2
development evaluation, with fit-level convergence and timing records, row predictions, and partial results
after each outer fold. `--protocol fixed_survey_en_layer` remains a separate historical-layer diagnostic;
the previous RILE-v1 path and filenames remain unchanged. The nested primary has **not run**.

**Budgeted selector (2026-09-28; FACT, implementation only).** `--protocol budgeted_source_peak`
uses the same five manifesto-grouped outer folds and aligned six-language RQ1/RQ2 scoring, but chooses
candidate layers with less inner training. Within each outer-training partition, it samples up to 1,250
items proportionally across manifesto × label × CMP category and scores every layer on one of five
grouped validation folds (about 1,000 training items), retaining the 12 highest scores. It independently samples up to 2,000 items from the
same outer-training partition, scores those 12 layers on two grouped folds, and retains four. It then
scores those four on all outer-training items with four grouped folds. Ties favor the earliest layer.
Sampling starts at seed 20260928 plus 1,000 per outer fold and advances deterministically if a sampled
grouped fold lacks either label. The selected layer is refit on all outer-training items and scored on
excluded outer-test manifestos; category-heldout items never enter selection or scoring. The output
records samples, stage scores, retained layers, folds, fit diagnostics, and row predictions under a
distinct protocol filename. This gives a valid outer score for the budgeted selection procedure, but
does not establish the exhaustive full-data peak layer. The exhaustive `nested_source_peak` protocol
remains available. For the four currently listed models, the known hidden-state counts are 33 (OLMo),
37 (Ministral), 43 (Gemma), and 33 (Qwen). The staged design therefore plans 9,400 logistic fits:
4,380 first-screen fits, 2,880 second-screen fits, 1,920 full-training selection fits, and 220 final
refits/comparators. This is a fit count, not a runtime measurement. No budgeted model evaluation has run.

**Parallel fit option (2026-09-29; FACT, bounded verification).** The runner now accepts `--fit-workers` for full
RILE-v2 `budgeted_source_peak` runs. Its default is one worker. Higher settings score independent
candidate layers concurrently within each fixed language, outer fold, and screening stage; sampling,
grouped folds, ranking, refits, and parent-written outputs remain the same in code. Independent source
review found no material protocol issue, and syntax/CLI checks passed. The local serial-versus-parallel
check could not run because the environment lacked `joblib` and dependency download failed on DNS. On
the L4, the saved English-only pilot JSON showed exact per-layer score and shortlist matches for one,
two, four, and six workers on the 1,250- and 2,000-item screens. Four workers took 148.489 seconds
across both screens, versus 244.550 seconds for one worker. Its largest sampled parent-plus-child RSS
was about 4.36 GiB; this sum can double-count shared pages and miss peaks between samples. A separate
80-item synthetic four-fold check found exact serial/four-worker scores, validation predictions,
rankings, iteration counts, and warning records. These are bounded checks, not a full-stage memory
measurement or RQ1/RQ2 result. Evidence: `src/multilingual_layerwise_probe.py` and the ignored
`gcp-workspace/worker-pilot-retrieval-20260929/` JSON/progress/stdout.

**Full four-model run (2026-09-29; INCOMPLETE).** The RILE-v2 `budgeted_source_peak` development run
started at 03:48:34 UTC with OLMo first, followed in the detached driver by Qwen, Ministral, and Gemma.
Each model has a separate directory under the cloud workspace's
`runs/rile-v2-budgeted-full-20260929/`. The command uses all six languages, five manifesto-grouped
outer folds, the recorded 20260928-plus-fold sampling seeds, `C=1`, `max_iter=3000`, batch size 8,
four CPU fit workers, and the corrected 5,044-item development split. The 1,087 category-holdout
items are excluded. This is a Compute Engine run; there is no Slurm ID or model-fitting random seed.
The active runner is the reviewed local `src/multilingual_layerwise_probe.py` copied to the VM before
launch; the local changes were not yet committed. The output is configured to retain per-fit timing,
iterations, warnings, layer selections, per-fold partial summaries, row predictions, and progress
events. A verified 48-hour Compute Engine STOP cap is in place, and the driver is configured to stop
on the first model error and shut down after completion. The last permitted launch check found the
driver and OLMo loading cached weights, with no scored result yet. See the ignored
`gcp-workspace/workspace_log.jsonl` and session/cost ledgers for operational provenance.

**Optimization pilots (2026-09-28; FACT, diagnostic only).** All four pilots used the corrected development
data, English text only, outer fold 0's 4,033 training items, four manifesto-grouped inner folds, raw
float32 final-token features, logistic `C=1`, and iteration caps 1,000 and 3,000. There was no random seed
or Slurm job. The code is `src/multilingual_layerwise_probe.py` with `--protocol nested_source_peak
--pilot-convergence`; separate ignored `gcp-workspace/rile-v2-nested-pilot-*` directories hold each pilot's
JSON, progress, and stdout. No values below are RQ1/RQ2 outer-test scores.

| Model | Hidden-state indices | Inner BA at 1,000 / 3,000 | Prediction changes | Convergence |
| --- | --- | --- | ---: | --- |
| OLMo | 18, 32 | 0.594248 / 0.594248; 0.646696 / 0.646696 | 0; 0 of 4,033 | All 16 fits converged below 1,000 |
| Ministral | 0, 32 | 0.463973 / 0.463973; 0.707880 / 0.709104 | 0; 39 of 4,033 | Deep layer: all four 1,000-cap fits warned; all four 3,000-cap fits converged at 1,259–1,769 iterations |
| Gemma | 16, 42 | 0.667911 / 0.667911; 0.713271 / 0.713271 | 0; 0 of 4,033 | All 16 fits converged below 1,000 |
| Qwen | 15, 32 | 0.645099 / 0.645099; 0.696096 / 0.696096 | 0; 0 of 4,033 | All 16 fits converged below 1,000 |

These sampled folds/layers support using a shared `max_iter=3000` cap without changing features, `C`,
objective, or preprocessing; all fits will still record their actual iterations and warnings. Other
layers/languages have not been tested for convergence. The four-model evaluation is authorized by the
owner but has not started. Automatic approval review rejected a 48-hour shutdown-timer extension during
the earlier session; that VM was stopped and the infrastructure ledger closed. A later two-hour worker
pilot was blocked first by a zone GPU stockout, then by a guest-initiated shutdown before SSH or model
work. At that point the shutdown cause was unknown. A 2026-09-29 read-only boot-disk-clone inspection
confirmed that two enabled, expired persistent timers started their power-off services during boot;
the guest journal records systemd shutdown before GCE's `guestTerminate` event. The failed pilot used
0.033195 VM/GPU-hours. On 2026-09-29 the two expired timer units were masked on a 100 GB disk clone;
the original disk remains untouched for rollback. The repaired clone booted, CUDA worked, and an
English-only Ministral worker pilot completed all 244 fits with no convergence warnings. Extracting
5,044 English development vectors took 113.53 seconds; fitting used only the 4,033 outer-fold-0
training items. The two screen stages took 244.550 seconds with one worker, 155.380 with two,
148.489 with four, and 151.379 with six. The result JSON, including layer agreement and sampled
process-tree RSS, remains on the stopped VM and has not yet been checked. The L4 was stopped before its
GCE-side two-hour cap; the helper is also stopped. The repair/pilot task used 0.425530 L4 VM/GPU-hours
across two starts and 0.077133 helper CPU-hours. The original disk, clone, helper disk, snapshot, and
machine image remain storage-billed; actual USD is unreconciled. See the ignored `gcp-workspace/`
session and cost ledgers and the dated entry in `docs/bugs-squashed.md`. These diagnostic fits do not
produce RQ1/RQ2 outer-test scores.

## 7. Study and result names (2026-09-26)

Use these identifiers in tables, notes, and messages. They are reporting names; they do not rename output
directories or files. Do not use “old run,” “new run,” or “corrected run” by themselves.

### Corpus and campaign names

- **SURVEY-v1**: the earlier GlobalOpinionQA-derived corpus, with 1,160 eligible items per language.
  Historical work on it used several different protocols; identify the protocol too.
- **RILE-native-gate**: the earlier, unversioned native English/German tone-gate data in
  `artifacts/results/rile_gate_legacy.json`. It is not SURVEY-v1 and is not the current six-language corpus.
- **RILE-v1**: the current original 6,167-item six-language diagnostic corpus (5,078 development and
  1,089 category-heldout items). The invalid-subcategory mapping, duplicate groups, and probe convergence
  issues remain open in §6. RILE-v1 scores are diagnostic and the category-heldout scores exploratory.
- **RILE-v2**: corrected six-language dataset candidate built 2026-09-27: 6,131 items (5,044 development,
  1,087 category holdout). It has four bounded convergence pilots but no scored outer-test evaluation.
  See `data/rile_v2/manifest.json` and [`dataset_manifesto_rile.md`](dataset_manifesto_rile.md) §13.2.
  The earlier 6,137 mapping-only count
  remains a hypothetical intermediate count, not this dataset.

For runner invocations, `--dataset rile_v2 --data-only` is a data-readiness report, not an evaluation.
Scoring RILE-v2 requires an explicit protocol. `--protocol fixed_survey_en_layer` identifies only a
secondary historical-layer diagnostic. `--protocol nested_source_peak` implements the outer/inner grouped
selection described in §6, but no completed model evaluation exists yet. Four bounded convergence pilots
are complete; the approved pre-run card and operational repair precede a full run. RILE-v2 outputs include
dataset and protocol in their filenames and records and refuse to overwrite existing outputs.

Name completed nested-analysis results **RILE-v2/NESTED-MANIFESTO-CV** for new-document
performance and **RILE-v2/NESTED-EN-TO-5** for the primary English transfer mean. Its unit of evaluation is
the outer held-out manifesto fold; report balanced accuracy per target language and the equal-weight mean
across the five targets, with paired manifesto-level intervals. The four-category result is
**RILE-v2/CATEGORY-HOLDOUT-EXPLORATORY** because those categories were inspected in the earlier campaign
and their manifestos overlap development. Historical-layer results use
**RILE-v2/FIXED-SURVEY-EN-LAYER-DIAGNOSTIC**. For `budgeted_source_peak`, use
**RILE-v2/BUDGETED-MANIFESTO-CV** and **RILE-v2/BUDGETED-EN-TO-5**. Its outer-test scores estimate the
budgeted layer-selection procedure, not the exhaustive full-data peak. These are proposed names and
estimands, not completed results.

For the current run, **DEV-MANIFESTO-CV** means 5-fold out-of-fold predictions grouped by manifesto across
5,078 development items. **CATEGORY-HOLDOUT** means 1,089 items from categories excluded from training;
their 58 manifestos also occur in development, so this does not hold out documents. The current probe is
raw-feature logistic regression with `C=1` and `max_iter=1000`.

### Result identifiers

- **RILE-v1/RQ1-EN-DEVPEAK/DEV-MANIFESTO-CV**: English development peak from the all-layer scan, selected
  by maximum balanced accuracy (earliest depth wins an exact tie). It is a derived view of the scan, not
  another experiment. Because the peak is selected on these development scores, report it as descriptive.
- **RILE-v1/RQ1-EN-DEVPEAK/CATEGORY-HOLDOUT**: category-heldout English balanced accuracy at the same
  development-selected layer. This is exploratory, not an untouched confirmation.
- **RILE-v1/RQ2-SURVEY-EN-LAYER/DEV-MANIFESTO-CV** and **…/CATEGORY-HOLDOUT**: current RILE-v1 scores at
  the historical English-selected layer, not SURVEY-v1 scores. The fixed blocks are OLMo 17, Ministral 31,
  Gemma 15, and Qwen 14. Report the equal-weight arithmetic mean of the five English-to-non-English target
  balanced accuracies; do not mix it with diagonal or full-matrix means.

For historical SURVEY-v1 results, keep these protocols distinct: **RQ1-ALL-LAYERS** (per-layer outputs
`probe_results_<model>.jsonl`); **RQ2-LINEAR-MATRIX** (`cross_lingual_matrix_<model>.json`);
**RQ2-LINEAR-ALL-PEAKS** (`cross_lingual_matrix_<model>_block<N>.json`);
**EXTRACTION-DIAGNOSTIC** (`extraction_diagnostics_<model>.json.gz`);
**RQ2-MLP** (`mlp_cross_lingual_matrix_<model>.json`); and the separate
**NEGATION-RESLICE**, **SENTIMENT-CHECK**, **TONE-CONTROL**, and **CLEAN-TONE-WRONG** checks. See
[`rq1_findings.md`](rq1_findings.md), [`rq2_findings.md`](rq2_findings.md), and
[`negation_preregistration.md`](negation_preregistration.md) for their historical descriptions. These
names separate workstreams; they do not imply identical data, metrics, or layer-selection rules.

### Reporting and provenance

Every reported number must name the corpus version, protocol, split/grouping, model, layer and its selection
rule, metric, aggregation, evaluation N, run/snapshot, and status (diagnostic, exploratory, or confirmatory).
The exact source summaries, formulas, selected blocks, and recomputed values behind the latest three-model
tables are recorded in the ignored local artifact at repository-root path
`gcp-workspace/rile-v1-four-model-20260926-0651-audit/diagnostic_table_provenance.json`. Qwen is omitted
there because this snapshot had no final development/heldout summary pair for it. Preserve physical
artifact names; use the identifiers above to label them in reporting.
