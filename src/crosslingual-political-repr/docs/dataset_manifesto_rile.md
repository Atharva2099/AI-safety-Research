# Manifesto Project RILE Dataset: Preprocessing Record

Last updated: 2026-09-30

This is a factual preprocessing record of the Manifesto Project data pipeline, organized by stage.
For the earlier survey-derived (GlobalOpinionQA) dataset, see `docs/dataset.md` — that background is
not repeated here.

Dates below come from `git log` and file modification times (`stat -f '%Sm'`), not from stage order.
Compute for every run in this document: local M1 (arm64) CPU. Slurm ID: N/A — not run on Slurm.

---

## 0. Motivation: why the survey-derived labels were replaced (FACT / NEGATIVE RESULT)

The GlobalOpinionQA-derived ±1 statements (`docs/dataset.md`) were checked for a negation/tone confound
before being trusted as a left/right signal. The check is pre-registered in
`docs/negation_preregistration.md` (Amendments 1–4), written and amended on 2026-09-15.

- **P1–P3** (pairwise cross-lingual transfer on the negation-clean subset): P1 true, P2 **false**, P3 true
  (Amendment 2, `artifacts/results/negation_reslice.json`). P2 predicted OLMo and Ministral would fall to
  ≤58% clean-subset pairwise accuracy; they did not, so the negation-only explanation for those two models'
  cross-lingual transfer is not supported by this test.
- **P4** (tone-matched subset): true (`artifacts/results/sentiment_check.json`), but the pre-registration
  itself notes (Amendment 2, point 4) that P4's 58% threshold was set relative to chance, not relative to a
  tone-only baseline, and that a tone-only pairwise baseline scores 0.679 on the same subset — so P4 being
  true does not show the probes beat tone.
- **Amendment 3 (tone-controlled, primary check):** `artifacts/results/tone_control.json`. Verdict "probe
  carries information beyond tone" (per the pre-declared 3-part decision rule) holds only for **gemma**;
  olmo, ministral and qwen are "not established."
- **Amendment 4 (exploratory, not pre-registered as a prediction):** `artifacts/results/clean_tone_wrong.json`.
  Restricting gemma's tone-wrong check to negation-clean questions drops its lower bound to **0.474**
  (mean 0.611, ci95 [0.536, 0.687] on all tone-wrong; mean 0.565, ci95 [0.474, 0.658] on tone-wrong-and-clean,
  n=45), i.e. **at or below 0.50**. Per the pre-registration's own reading rule, this means gemma's Amendment-3
  "beyond tone" result is reported as possibly explained by a negation preference rather than a stance signal,
  not confirmed as a clean stance effect. This is an exploratory check, not a pre-registered prediction.

Net: the survey-derived labels' cross-lingual transfer signal could not be cleanly separated from tone and
negation confounds by these checks, which motivated moving to the Manifesto Project's expert-coded RILE
labels as an independent label source not derived from an LLM.

---

## 1. Source & access (FACT)

- **Corpus:** Manifesto Project (MARPOR), dataset id `MPDS2026a`, corpus version `2026-1`, accessed via the
  official API (registration + personal API key, read from `MANIFESTO_API_KEY`, never printed or written to
  disk — verified by reading `src/fetch_manifesto.py`, which reads the key from the environment only).
- **Licence:** Manifesto Project Terms of Use (https://manifesto-project.wzb.eu/information/documents/terms_of_use)
  forbid redistributing the corpus, including derived translations. API documented at
  https://manifesto-project.wzb.eu/information/documents/api.
- **Code:** `src/fetch_manifesto.py`. Untracked as of 2026-09-17 (`git status --short` shows `??`; `git log
  --oneline -- src/fetch_manifesto.py` returns nothing), so it has no commit hash to cite yet.

**Fetch bugs fixed during development** (observed in-session; the original failing states were not saved as
artifacts or error logs, so these are recorded as FACT only insofar as the *current* code reflects the fix,
not as independently reproducible evidence of the original failures):

- **Dataset id prefix 404:** an earlier core-dataset key without the `MPDS` prefix/version suffix 404'd
  against `api_get_core`. Current code takes `--core-version` (default `MPDS2026a`) as the exact key.
- **`api()` key-name collision:** the API helper's `**params` forwarding previously collided with the
  function's own `api_key` parameter name. Current `api()` (`src/fetch_manifesto.py:24`) takes `api_key`
  as an explicit positional argument, separate from `**params`, avoiding the collision.
- **Connection resets on long corpus requests:** the corpus endpoint drops long-running connections
  "fairly often" (code comment, `src/fetch_manifesto.py:25`). Current code retries up to `tries=4` times
  with backoff (`time.sleep(3 * (attempt + 1))`) on `requests.exceptions.RequestException`, and on a request
  that still fails after retries inside the per-manifesto loop, breaks out and writes whatever manifestos
  were already collected (`docs.write_text(...)` runs regardless of where the loop stopped) rather than
  discarding partial progress.

---

## 2. Fetches (FACT)

Two raw pulls, both gitignored (verified: `git check-ignore -v data/manifesto_raw.json
data/manifesto_raw_de_es.json` matches both against `.gitignore:58: src/crosslingual-political-repr/data/manifesto_*`).

| file | command (per docstring/argparse) | rows | manifestos | parties | party-id prefixes | election dates |
|---|---|---|---|---|---|---|
| `data/manifesto_raw.json` | `--countries <English-speaking countries> --limit 40` | 79,861 | 40 | 37 | 51 (UK), 53, 61 (US), 62 (Canada), 63 (Australia), 64 (New Zealand) | 2020-02 to 2024-11 |
| `data/manifesto_raw_de_es.json` | `--countries Germany Spain --limit 30` | 64,067 | 30 | 23 | 41 (Germany, 15 manifestos / 8 parties), 33 (Spain, 15 manifestos / 15 parties) | 2019-11 to 2025-02 |

Counts computed directly from the two files (`manifesto_id`, `partyname`, `party`, `date` fields), not from
any script's printed summary. File modification times: `manifesto_raw.json` 2026-09-15 20:41;
`manifesto_raw_de_es.json` 2026-09-16 00:57.

---

## 3. Label scheme: RILE (FACT / SUPERSEDED prior design)

**Current source of truth:** Manifesto Project Dataset codebook MPDS2024a, section 3.6 "Programmatic
dimensions", p.30
(https://manifesto-project.wzb.eu/down/data/2024a/codebooks/codebook_MPDataset_MPDS2024a.pdf), quoted
verbatim in `src/build_rile_set.py:3-8`:

```
right = per104 + per201 + per203 + per305 + per401 + per402 + per407 + per414 + per505
      + per601 + per603 + per605 + per606
left  = per103 + per105 + per106 + per107 + per403 + per404 + per406 + per412 + per413
      + per504 + per506 + per701 + per202
```

Categories outside both lists are dropped (RILE ignores them).

**SUPERSEDED earlier design** (`docs/bugs-squashed.md`, 2026-09-16 entry "Left/right keying was invented
before being checked against the codebook"): the first design keyed left vs. right from six *opposing
category pairs* (601/602, 603/604, 406/407, 504/505, 701/702, 203/204 — this exact set is also the `ISSUES`
list in `src/manifesto_tone_multimodel.py:23-25`; `src/manifesto_tone_check.py:23-27` used a larger 10-pair
list plus an excluded military pair) chosen from recalled knowledge of RILE, not verified against the
codebook. Categories 602, 604, 204 and 702 are in **neither** the RILE right nor left list, so four of the
six planned pairs would have been keyed on an invented scheme. This was caught before any probe was trained
(bugs-squashed, same entry) and replaced with the verified RILE formula above.

---

## 4. Tone-confound checks on the paired design (SUPERSEDED design, but the checks are informative)

These checks ran on the earlier opposing-category-pair design, before the RILE codebook lookup (§3). They
are reported because they surfaced a real tone confound in the underlying data, independent of which keying
scheme is used.

**`src/manifesto_tone_check.py` → `artifacts/results/manifesto_tone_check.json`.** Single-model (cardiff)
tone AUC per issue pair, `>=5`-word sentences, bootstrap 95% CI (300 draws, capped at 1,500/side, seed
20260915). Selected results (AUC 0.50 = tone uninformative of the pair label):

| issue | n+ | n− | tone AUC | 95% CI |
|---|---|---|---|---|
| labour (701/702) | 2,588 | 111 | 0.856 | [0.795, 0.866] |
| EU integration (108/110) | 230 | 128 | 0.735 | [0.681, 0.788] |
| protectionism (406/407) | 375 | 272 | 0.376 | [0.336, 0.417] |
| national way of life (601/602) | 1,578 | 722 | 0.532 | [0.508, 0.555] |
| traditional morality (603/604) | 429 | 154 | 0.482 | [0.431, 0.536] |
| constitutionalism (203/204) | 87 | 151 | 0.449 | [0.376, 0.527] |
| welfare (504/505) | 9,525 | 251 | 0.592 | [0.522, 0.602] |
| military (104/105, excluded from the main table by design — its "negative" side is condemnation of atrocities, not an anti-military stance) | 2,652 | 242 | 0.742 | [0.711, 0.777] |

**`src/manifesto_tone_multimodel.py` → `artifacts/results/manifesto_tone_multimodel.json`.** Checks whether
matching on cardiff tone alone (10 bins) generalizes to four independent validator models (nlptown, sst2,
siebert, twitter_rob), and whether jointly matching on cardiff+siebert (5x5 bins) does better. 6-issue
`ISSUES` list (601/602, 603/604, 406/407, 504/505, 701/702, 203/204), seed 20260915.

| matching scheme | n pairs | pooled worst independent-validator gap from 0.50 | worst single issue |
|---|---|---|---|
| cardiff only | 1,143 | 0.019 | labour / siebert, gap 0.203 |
| cardiff + siebert (joint) | 1,148 | 0.014 | labour / siebert, gap 0.160 |

Pooled matching brings the aggregate gap under ~0.02 either way, but per-issue, **labour stays separable by
siebert** (an independent model with no say in the cardiff-only matching) under both schemes, and
protectionism's unmatched AUC (`manifesto_tone_check.json`, above) is 0.376 — far from 0.50 in the opposite
direction. This is why "labour and protectionism [are] still separable by independent models" even after
tone-matching one or two models jointly; it is reported here as a property of this superseded design, not
re-checked against the final RILE-keyed, per-language-matched sets in §5/§7.

---

## 5. Filtering & building (FACT)

**Code:** `src/build_rile_set.py`. Untracked as of 2026-09-17 (`git status --short` → `??`; no commit
history for this path).

**Pipeline** (per the script): keep sentences with ≥5 words (`MIN_WORDS`) and a numeric `cmp_code` in the
RILE right/left union → label `1` (right) / `0` (left) → optional seeded per-sentence language filter
(`langdetect`, `DetectorFactory.seed = 0`) → cap each category at `--cap` (default 800) sentences
(`SEED = 20260915`, `np.random.default_rng`) → tone-match right vs. left in 10 bins (`--bins`) on a chosen
model's valence (`--match-model`, default `cardiff`).

**Per-language settings used:**

| lang | input | `--party-prefix` | `--match-model` | other flags |
|---|---|---|---|---|
| en | `manifesto_raw.json` | (none) | cardiff (default) | — |
| de | `manifesto_raw_de_es.json` | `41` | cardiff (default) | — |
| es | `manifesto_raw_de_es.json` | `33` | `spanish` | — |

**Stage counts**, from `artifacts/results/manifesto_{en,de,es}_rile_build.json` (`stages.raw` /
`stages.capped` / `stages.matched`, each with `n_right`, `n_left`, `tone_auc`) and the output data files
`data/manifesto_{en,de,es}_rile.json`:

| lang | raw (post language-filter) | capped | matched (final) | matched tone AUC | manifestos | parties |
|---|---|---|---|---|---|---|
| en | 16,712 right / 20,864 left | 7,680 / 6,304 | 6,197 / 6,197 (12,394 rows) | 0.500 | 39 | 36 |
| de | 7,175 / 10,974 | 5,719 / 6,213 | 5,339 / 5,339 (10,678 rows) | 0.498 | 15 | 8 |
| es | 2,153 / 4,983 | 2,153 / 3,829 | 2,153 / 2,153 (4,306 rows) | 0.499 | 15 | 15 |

Row/manifesto/party counts independently recomputed from the data files themselves match the build JSONs
exactly.

**Output locations:** `data/manifesto_en_rile.json`, `data/manifesto_de_rile.json`,
`data/manifesto_es_rile.json`, all gitignored (`git check-ignore -v` confirms all three against
`.gitignore:58`).

**File-timestamp finding (not in the original task description, found during verification):** the en/de
build JSONs (`manifesto_en_rile_build.json`, `manifesto_de_rile_build.json`, mtimes 2026-09-16 01:21 / 01:25)
predate `src/build_rile_set.py`'s current mtime (2026-09-16 08:58) and lack the `match_model` field that
the current script always writes; the es build JSON (mtime 09:01) has it (`"match_model": "spanish"`). This
means `--match-model` was added to the script after the en/de builds ran but before the es build ran. The
en/de *data* files are unaffected (they used the then-current default, cardiff, which is still the current
default), but their build-metadata JSONs are from an older script revision and are missing one field present
in the newer schema.

**Language contamination (`docs/bugs-squashed.md`, 2026-09-16, "Country is not language"):** before the
`langdetect` filter, `langdetect` over the built Spanish set found 1,906 of 6,010 sentences (31.7%) were
Catalan (mostly from Catalan Republican Left [707, 0% Spanish], In Common We Can [714, 0%], Together for
Catalonia [492, 1%]), plus 55 Portuguese/Galician; the English set had 44 French sentences from Bloc Quebecois
(0.8%); German was 99.8% clean. Filter-kept counts (RILE-restricted, ≥5-word sentences, before capping):
en 37,576 of 37,883 (99.2%), de 18,149 of 18,178 (99.8%), es 7,136 of 10,463 (68.2%) — independently
recomputed here by re-running the same RILE-code + word-count filter over `manifesto_raw.json` /
`manifesto_raw_de_es.json` and applying `langdetect` with `DetectorFactory.seed = 0`; these numbers match
`stages.raw` in the build JSONs exactly. One English manifesto (id `62901_202109`) had zero sentences detected
as English and was dropped entirely by the filter, which is why the English matched set has 39 manifestos
against 40 in the raw pull.

---

## 6. Tone models & validators (FACT)

`src/tone.py` (modified, uncommitted as of 2026-09-17 per `git status --short`). Pinned model revisions
(`TONE_MODELS`, `src/tone.py:16-26`, "Revisions pinned 2026-09-15"):

| key | model | revision |
|---|---|---|
| cardiff | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8` |
| nlptown | `nlptown/bert-base-multilingual-uncased-sentiment` | `8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d` |
| sst2 | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `714eb0fa89d2f80546fda750413ed43d93601a13` |
| siebert | `siebert/sentiment-roberta-large-english` | `74cea614e245b0832c770ec9aa51bd58df965b9c` |
| twitter_rob | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7` |
| german | `oliverguhr/german-sentiment-bert` | `b1177ff59e305c966836ba2825d3dc2efc53f125` |
| spanish | `pysentimiento/robertuito-sentiment-analysis` | `a2cc0f67ebd705c55191e25a05ba23d885fcc09b` |
| xlmr_multi | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | `82107f4ccba672c9ab1eee538e727e462cad21de` |

`valence()` (`src/tone.py:80-93`): for models with positive/negative labels (aliasing neg→negative,
pos→positive, neu→neutral), valence = P(positive) − P(negative). For star-rated models (nlptown), valence =
(expected stars over the softmax − 3) / 2, rescaling stars 1–5 to [−1, 1].

**Per-model max-length fix** (`docs/bugs-squashed.md`, 2026-09-16, "Tone scoring truncated every model at
512 tokens"): a fixed `max_length=512` crashed `pysentimiento/robertuito-sentiment-analysis`
(`max_position_embeddings` = 130) with an out-of-bounds index error. `src/tone.py:48-49` now caps each
model's max length at `max(16, min(512, max_position_embeddings - 2))`.

`VALIDATORS_BY_LANG` (`src/tone.py:29-33`): en → nlptown, sst2, siebert, twitter_rob, xlmr_multi; de →
nlptown, german, xlmr_multi; es → nlptown, spanish, xlmr_multi. cardiff is excluded from every language's
validator list because it is the model used for tone-matching.

**Negation patterns:** `NEGATION_BY_LANG` (`src/negation_reslice.py:30-34`, modified/uncommitted as of
2026-09-17): en uses the pre-registered `NEGATION` word list; de =
`\b(nicht|kein\w*|nie|niemals|nichts|niemand|ohne|weder)\b`; es =
`\b(no|nunca|ni|sin|nada|nadie|jamás|jamas|tampoco)\b` (case-insensitive). Code comment at
`src/build_rile_set.py:62` records the reason this is per-language: "an English pattern on German text
would report a meaningless 0.000 [negation gap]."

---

## 7. Validator gates (FACT, with two contradictory-looking results resolved by build provenance)

**Gate rule:** flag failure if any independent (non-matching) validator's pooled AUC on the finished,
tone-matched set deviates from 0.50 by more than 0.05.

Two Spanish gate artifacts exist and correspond to two different builds of the Spanish data:

| artifact | matched on | n (per `tone_scores_spanish.json` cache key / JSON `n`) | worst independent validator gap | verdict |
|---|---|---|---|---|
| `artifacts/results/rile_validator_gates.json` (`es` entry) | cardiff | 4,194 (`aucs`: cardiff 0.498, nlptown **0.551**, spanish 0.537, xlmr_multi 0.507) | 0.051 | **FAIL** |
| `artifacts/results/rile_gate_es_v3.json` | spanish (robertuito) | matches `es_rile_matched_v3` cache key, 4,306 rows | 0.047 (`aucs`: spanish 0.499, cardiff 0.453, nlptown 0.517, xlmr_multi 0.464) | PASS |

**Which one is current:** `data/manifesto_es_rile.json` has 4,306 rows, matching `rile_gate_es_v3.json`'s
build (confirmed by cross-referencing `artifacts/results/tone_scores_spanish.json`'s cached tone-score
entries: cache key `es_rile_matched_v2` has 4,194 scored rows and `es_rile_matched_v3` has 4,306), and
`manifesto_es_rile_build.json` records `"match_model": "spanish"`. The cardiff-matched, 4,194-row build
(`rile_validator_gates.json`) is **not** the build behind the current data file.

- The **cardiff-matched Spanish gate (`rile_validator_gates.json`, FAIL) is labeled NEGATIVE RESULT**: it is
  a real, completed check that failed the pre-declared threshold, not a bug — it directly motivated rebuilding
  matched on a different model (`docs/bugs-squashed.md`, 2026-09-16).
- The **robertuito-matched Spanish gate (`rile_gate_es_v3.json`, PASS) carries a LIMITATION**: the passing
  margin is thin (0.047 against a 0.05 threshold), it is the second attempt after the first failed, and it
  uses a different matching model from English/German (cardiff for en/de, robertuito/"spanish" for es), so
  the pipeline is not uniform across the three languages (`docs/bugs-squashed.md`, same entry, "Impact").

**`artifacts/results/manifesto_english_rile_validators.json` is SUPERSEDED.** It has no row-count field, but
its file mtime (2026-09-16 00:53) predates both `data/manifesto_en_rile.json` and
`manifesto_en_rile_build.json` (both 01:21). Cross-referencing `artifacts/results/tone_scores_cardiff.json`'s
cache: the `english_rile_matched` cache key (an earlier, differently-named English matched-set cache entry)
has exactly 12,498 scored rows, while `en_rile_matched_v2` — closer in time to the current build — has
12,394 rows, matching the current `data/manifesto_en_rile.json` row count (12,394) exactly. This row-count
evidence (not an AUC comparison — recomputing the AUC for the `english_rile_matched` cache entry was not
attempted, since it would require reproducing the exact row order of a build whose script version is not
recoverable) is consistent with `manifesto_english_rile_validators.json` having been computed on an earlier,
differently-sized English build (12,498 rows) rather than the current 12,394-row one. It is superseded by
whatever gate is run against the current English data; no English-specific gate artifact with a row count
matching 12,394 was found in `artifacts/results/` (flagged as INCOMPLETE below).

**2026-09-17 update — English/German gate artifact.** `artifacts/results/rile_gate_en_de.json` (untracked,
mtime 2026-09-17 10:29) now provides gates matching the current English and German data files, matched on
cardiff: en n=12,394 (`data/manifesto_en_rile.json`), worst independent-validator gap 0.019, **PASS**; de
n=10,678 (`data/manifesto_de_rile.json`), worst gap 0.006, **PASS**. Both read from cached tone scores (cache
key `{lang}_rile_matched_v2`) with no re-scoring. This closes the previously-flagged gap for English and
German (the row counts match the current data files exactly, unlike `manifesto_english_rile_validators.json`
in §7 above, which stays SUPERSEDED).

**Why it was missing:** the original combined en/de/es gate run crashed on Spanish (robertuito's 130-token
limit — the same class of bug as §6's "Tone scoring truncated every model at 512 tokens") before reaching its
`json.dump`, so no en/de gate output was ever written from that run. A later Spanish-only rerun (producing
`rile_validator_gates.json`, then `rile_gate_es_v3.json` after the match-model change) wrote only the `es`
key, leaving en/de ungated until the 2026-09-17 rerun above.

---

## 8. Known data-quality issues (LIMITATION / INCOMPLETE, observed 2026-09-16/17, not yet fixed)

All counts below were computed directly from the current data files (`data/manifesto_{en,de,es}_rile.json`)
on 2026-09-17.

- **RILE label follows the sentence's category, not the party.** Verified directly: in
  `data/manifesto_en_rile.json`, 46 sentences from Green parties (Australian Greens; Green Party [UK];
  Green Party of Aotearoa New Zealand) are coded under RILE-right categories (e.g. category 605
  "law and order"), and are labeled `1` (right) by the per-sentence scheme — e.g. one Australian Greens
  sentence coded 605 is labeled right. Separately, 132 Republican Party sentences are coded under RILE-left
  categories 107/202 and labeled `0` (left). This is expected behavior of a per-sentence, category-keyed
  label (not a bug in the code), but is a LIMITATION for any downstream framing that reads the label as
  "the sentence's party's overall ideology."
- **Category coding noise:** not independently re-verified beyond the category/party mismatches above; flagged
  as an open, unquantified LIMITATION (Manifesto Project's own hand-coding is not re-audited by this pipeline).
- **Residual Galician in the Spanish set:** could not be independently confirmed. Re-running `langdetect`
  (the same library, `DetectorFactory.seed = 0`, used as the build-time filter) over all 4,306 rows of the
  current `data/manifesto_es_rile.json` finds 0 rows detected as anything other than `es`. This check is
  circular (it reuses the exact tool that already filtered the file) and cannot rule out residual
  Galician/Catalan that `langdetect` itself misclassifies as Spanish. **INCOMPLETE** — no independent
  language-detection tool was run, per the read-only/no-re-fetch constraint on this task.
- **Mis-decoded characters, e.g. `<96>` (likely an en/em-dash mis-decode) in English text:** counted directly.
  54 of 12,394 English rows (0.44%) contain the literal substring `<96>`; 0 of 10,678 German rows and 0 of
  4,306 Spanish rows do. One short example (`data/manifesto_en_rile.json`): `"...r higher taxes <96> which
  discoura..."`.
- **Quasi-sentence fragments starting lowercase** (a proxy for fragments, not a direct defect count): counted
  by checking whether the first character of `text` is lowercase. en: 1,773 of 12,394 (14.3%); de: 802 of
  10,678 (7.5%); es: 879 of 4,306 (20.4%). Some fraction of these are legitimate quasi-sentence continuations
  from the Manifesto Project's own coding units, not necessarily errors; this count is reported as a
  LIMITATION (upper-bound proxy), not a verified defect rate.

---

## 9. Stage 1: text cleaning (FACT, run 2026-09-17)

**Code:** `src/clean_pools.py`. Uncommitted as of 2026-09-17 (`git status --short` → `??`; `git log --oneline
-- src/clean_pools.py` returns nothing, so no commit hash to cite yet).

**Run:** 2026-09-17, one background command, all three languages sequentially, local M1 (arm64) CPU. Slurm
ID: N/A — not run on Slurm.

```
uv run --with langdetect --with lingua-language-detector python -m src.clean_pools \
    --input data/manifesto_raw.json --lang en && \
uv run --with langdetect --with lingua-language-detector python -m src.clean_pools \
    --input data/manifesto_raw_de_es.json --lang de --party-prefix 41 && \
uv run --with langdetect --with lingua-language-detector python -m src.clean_pools \
    --input data/manifesto_raw_de_es.json --lang es --party-prefix 33
```

**Rules applied, in order** (per the module docstring; each rule is counted separately in the per-language
report):

1. **base** — keep sentences with ≥5 words and a numeric `cmp_code` inside the RILE right/left union (§3).
2. **mojibake repair** — literal `<92>`/`<96>`-style tags are single Windows-1252 bytes printed as hex;
   decoded back to real characters via `cp1252`.
3. **label-conflict removal** — every copy of a text whose duplicates disagree on left/right is dropped,
   since no correct label exists for it.
4. **dedup** — one copy kept per repeated text, deterministic sort by `(manifesto_id, date, text)`.
5. **fragment removal** — drop if the text starts lowercase AND lacks a terminal `.`/`!`/`?`.
6. **language** — keep only sentences where `langdetect` (`DetectorFactory.seed = 0`) and `lingua` **both**
   report the target language.

**Outputs, per language:** `data/clean_<lang>.jsonl`, `data/dropped_<lang>.jsonl` (every removal tagged with
a reason code), `artifacts/results/clean_<lang>_report.json`. The two `data/` files are gitignored
(`git check-ignore -v data/clean_en.jsonl data/dropped_en.jsonl` matches `.gitignore:62-63`); the counts-only
report JSONs under `artifacts/results/` remain tracked.

**Stage counts**, read directly from `artifacts/results/clean_{en,de,es}_report.json` — checked against this
task's brief and no mismatch found:

| lang | base | label-conflict (texts / sentences) | dedup (extra copies removed) | fragment | language (dropped / detectors disagreed) | final n | right / left | manifestos | parties |
|---|---|---|---|---|---|---|---|---|---|
| en | 37,883 | 111 / 484 | 8,368 | 1,192 | 233 / 96 | 27,606 | 11,326 / 16,280 | 38 | 35 |
| de | 18,178 | 12 / 24 | 267 | 240 | 28 / 27 | 17,619 | 6,939 / 10,680 | 15 | 8 |
| es | 10,463 | 1 / 3 | 42 | 636 | 3,254 / 135 | 6,528 | 2,004 / 4,524 | 14 | 14 |

Mojibake repair: 879 English sentences fixed (artefact tag tallies from `clean_en_report.json`'s `artefacts`
field: `<92>` 803, `<96>` 248, `<93>` 53, `<94>` 53, `<ae>` 5, `<85>` 2, `<91>` 7); 0 sentences fixed in de/es.
One short example (`data/manifesto_raw.json`, before → after repair): `"...Government<92>s economic
plan..."` → `"...Government's economic plan..."` (`<92>` is cp1252 0x92, a right single quotation mark).

- **FACT:** Spanish lost 3,254 sentences (31%) at the language step, consistent with the earlier
  language-contamination finding for Spanish reported in §5 — see that section for the Catalan/Galician
  breakdown; not re-derived here.
- **FACT:** label conflicts rose from 92 texts (measured pre-repair, independently recomputed 2026-09-17 by
  applying only the base filter and conflict grouping to `manifesto_raw.json` while skipping the mojibake
  step) to 111 texts after mojibake repair (`clean_en_report.json`). Mechanism, consistent with the code
  order in `src/clean_pools.py` (repair in step 2 runs before conflict grouping in step 3): pairs of texts
  that previously differed only by a mis-decoded character became identical strings after repair and were
  then grouped into the same conflict.
- **LIMITATION:** the fragment rule (starts lowercase AND has no terminal `.`/`!`/`?`) is a heuristic. The
  conservative "both conditions" form was chosen after comparing four candidate rules on deduplicated
  English: starts-lowercase alone 13.4%, no-terminal-punctuation alone 20.9%, both 4.0%, either 30.3%. Only
  aggregate counts of the alternatives were recorded, not artifacts.
- **LIMITATION:** `lingua` has no Galician language model; Galician text is detected by `lingua` as
  Portuguese, which is the mechanism by which the both-detectors-agree rule removes it from the Spanish pool.
  This is stated as the detection mechanism, not as a verified per-sentence Galician count.
- **FACT:** duplicates were overwhelmingly cross-manifesto campaign slogans: of 4,181 repeated texts in the
  base-filtered English pool (pre-cleaning), 3,941 spanned more than one `manifesto_id` (independently
  recomputed 2026-09-17 over `manifesto_raw.json`), so deduplication removes a memorisation shortcut rather
  than legitimate within-manifesto repetition.

**Paired-opposites availability on the cleaned pools**, independently recomputed 2026-09-17 by counting
`cmp_code` in `data/clean_{en,de,es}.jsonl` (left/right per the RILE keying in §3):

| pair (cmp codes) | en left / right | de left / right | es left / right |
|---|---|---|---|
| military (104/105) | 221 / 1,580 | 408 / 519 | 53 / 94 |
| welfare (504/505) | 7,160 / 219 | 2,976 / 230 | 1,727 / 4 |
| protectionism (406/407) | 324 / 191 | 73 / 87 | 57 / 15 |

English+German balanced availability (min(left, right), summed across en+de): military 629, welfare 449,
protectionism 264.

**`.gitignore` change (2026-09-17):** added `data/clean_*`, `data/dropped_*`, `data/rile_v*/`
(`.gitignore:62-64`), alongside the pre-existing `manifesto_*` ignores from §1/§2, so Stage 1 outputs are
never committed; the counts-only `clean_*_report.json` files under `artifacts/results/` remain tracked.

---

## 10. Open decisions (2026-09-17, INCOMPLETE — plan, not done)

Proposal, not yet implemented or approved as of this writing:

1. Clean the English set (address §8's issues where feasible). See §9 above — Stage 1 cleaning
   (`src/clean_pools.py`, run 2026-09-17) implements this for all three languages, not English alone.
2. Translate a subset of roughly 1,500 sentences per side (right/left) into es, de, zh, hi, mr, to build a
   parallel design that isolates language as the sole varying factor, while keeping the existing native
   German set as a translationese check (native vs. translated German).
3. Tone must be re-checked per language after translation (translation can shift valence).
4. Any produced translations remain Manifesto Project derived data and fall under the same redistribution
   restriction as the source corpus (§1) — they cannot be published or uploaded alongside this repo's public
   history.

Stage 1 cleaning and v1 translation, vetting, and repairs were completed before this update. Final assembly and the per-language tone check were completed on 2026-09-25; see §§11 and 13. The broader translator-family comparison in §12 remains planned and has not been run.

---

## 11. Stage 3: translation provenance (FACT, recorded 2026-09-17)

v1 corpus translations (`data/rile_v1/translations.jsonl`, 6,169 rows) were produced with `gemini-3.5-flash-lite`
(`MODEL`, `src/translate_statements.py:18`). Verified directly against the data: every one of the 6,169 rows'
own `model` field reads `gemini-3.5-flash-lite` (no other value present).

The vetting judge is `gemini-3.8-flash` (`JUDGE_MODEL`, `src/vet_translations.py:26`), deliberately a
different model from the translator, per the code comment on that line, so the judge is not grading its own
output. The 663 recorded repairs use `gemini-3.7-flash` (`src/fix_translations.py:28` and the `model` fields
in `data/rile_v1/fixes.jsonl`). They were re-judged by `gemini-3.8-flash` in the separate vetting pass, so
the repair writer and judge were different models.

**2026-09-17 auth change (FACT):** the three Gemini-calling scripts (`src/translate_statements.py`,
`src/vet_translations.py`, `src/fix_translations.py`) were changed to authenticate via Vertex AI
application-default credentials (ADC, billed to GCP credits) by default (`get_client()` in each file calls
`google.auth.default()` and constructs a Vertex-backed client); the bare Gemini API key path now requires an
explicit `USE_GEMINI_API_KEY=1` environment variable. This was made after an initial portion of the
translation run was billed to the API key instead of Vertex/ADC.

---

## 12. Translator-family bias (teacher-student effect) (HYPOTHESIS, recorded 2026-09-17)

**Status: nothing below has been measured yet.** This section records a hypothesis and a planned test, not a
finding.

**The concern (HYPOTHESIS).** §11 above establishes that the corpus translations were produced by Gemini
models. One of the four probed models, `google/gemma-2-9b-it`, comes from the same model family as the
translator. Gemini-generated text may therefore be intrinsically easier for Gemma to represent than for OLMo,
Ministral or Qwen. Three candidate mechanisms, all unverified:

1. Shared tokenizer lineage — Gemma's tokenizer derives from the same family as Gemini's, so Gemini-produced
   text may segment more favourably for Gemma.
2. In-distribution familiarity — Gemma was trained with Gemini-family supervision, so Gemini output is closer
   to its training distribution (typically lower perplexity).
3. Style fingerprints — consistent lexical/phrasing choices that a same-family model predicts more easily.

**Why it matters retroactively (LIMITATION).** The earlier survey-derived dataset was also Gemini-translated
(`src/translate_statements.py`, `MODEL = "gemini-3.5-flash-lite"`), and Gemma had the highest cross-lingual
transfer of the four models in those results: off-diagonal 84.67% (diagonal 86.09%) at layer 23, baseline
`current_raw` extraction (`docs/rq2_findings.md`, "8-Condition Representation Extraction & Diagnostic
Findings" table, row "Gemma 2 (9B)"). "Gemma transfers best" and "Gemma finds Gemini-generated text easy" are
not currently distinguishable in those results. This is stated as a limitation of the existing survey-derived
results, not as a retraction of them.

**The built-in test (planned, not run).** The v1 corpus tags every item with `source_lang`
(`data/rile_v1/items.jsonl`), so within a single target language the corpus contains both native manifesto
sentences (items sourced in that language) and Gemini translations (items sourced in the other two). Counts
independently recomputed 2026-09-17 by tallying `source_lang` over all 6,169 rows of
`data/rile_v1/items.jsonl` (en/de/es only — no other `source_lang` value is present):

| target language | native (source_lang = target) | translated (source_lang ≠ target) |
|---|---:|---:|
| en | 2,113 | 4,056 (de 2,056 + es 2,000) |
| de | 2,056 | 4,113 (en 2,113 + es 2,000) |
| es | 2,000 | 4,169 (en 2,113 + de 2,056) |

This native/translated split per language is the sample available for the planned test: compare Gemma's
margin over the other three models on native versus translated items within the same language. Equal margins
argue against the confound; a margin that appears only on translated text supports it.

**Cheap supporting diagnostics (planned, not run).** Per-model perplexity on native versus translated
sentences; tokens-per-sentence per model on identical text (tests the tokenizer mechanism specifically).

**What the design cannot settle (LIMITATION).** Chinese, Hindi and Marathi have no native manifesto source in
this corpus (confirmed: `source_lang` never takes the value `zh`, `hi`, or `mr` in `data/rile_v1/items.jsonl`)
— all their text is translated, so the native-vs-translated confound cannot be separated in those three
languages. Any Gemma advantage there stays ambiguous.

**Stronger test not yet approved (INCOMPLETE).** Retranslating a subset with a non-Google system (an open
model or a dedicated MT system) and checking whether the model ranking changes. Requires owner approval
because it costs money.

---

## 13. Stage 4: finalized translations and tone check (FACT / LIMITATION, 2026-09-25)

`src/finalize_rile_translations.py` assembled the six language fields from the original translations plus
the latest repair for each item/language pair, applied the two owner `drop_item` decisions and the Marathi
revert, and wrote the local ignored file `data/rile_v1/final_translations.jsonl`. It contains 6,167 items:
3,075 right-coded and 3,092 left-coded. All six language fields are present for every item. The two dropped
items are excluded globally. Source files were left unchanged.

`src/check_rile_tone.py` scored all 6,167 texts in each language once with the pinned Cardiff multilingual
sentiment model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`, revision
`f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8`). Tone is valence, P(positive) − P(negative); tone AUC is the
probability that a random right-coded text receives a higher valence than a random left-coded text, with
ties counted as one half. The output `artifacts/results/rile_v1_tone.json` contains language totals and
native-versus-translated results for English, German, and Spanish.

| language | n | left / right | mean valence left / right | tone AUC |
|---|---:|---:|---:|---:|
| English | 6,167 | 3,092 / 3,075 | 0.042 / 0.001 | 0.483 |
| Spanish | 6,167 | 3,092 / 3,075 | 0.062 / 0.024 | 0.486 |
| German | 6,167 | 3,092 / 3,075 | -0.039 / -0.063 | 0.479 |
| Chinese | 6,167 | 3,092 / 3,075 | -0.023 / -0.049 | 0.485 |
| Hindi | 6,167 | 3,092 / 3,075 | 0.064 / 0.038 | 0.483 |
| Marathi | 6,167 | 3,092 / 3,075 | -0.039 / -0.057 | 0.482 |

For English, the native and translated AUCs are 0.468 (n=2,113) and 0.492 (n=4,054); for Spanish, 0.463
(n=1,998) and 0.497 (n=4,169); for German, 0.525 (n=2,056) and 0.454 (n=4,111). Overall, these scores show
weak left/right separation under this one tone score. The translated-German AUC of 0.454 is a modest reverse
ranking. These descriptive scores have no confidence intervals or human tone ratings, so they do not
establish whether subgroup differences are meaningful, that tone is absent, or that political content is
tone-balanced. Chinese and Marathi results are exploratory because Cardiff's training-language coverage is
limited; all model-based tone results remain subject to sentiment model error.

### 13.1 Superseding dataset and tone caveats (2026-09-26 audit; dataset repaired 2026-09-27)

The counts and tone scores in §13 describe the assembled 6,167-row file **before** the RILE mapping audit.
They are provisional and must be recomputed on a corrected dataset. The builder strips category suffixes
before membership validation (`src/build_rile_set.py:49-55`), and the sampler can retain subcategories
(`src/sample_corpus.py:37-50`). The official MPDS2024a codebook excludes 202_2 and 605_2 from the RILE
component categories (pp. 10, 30: [codebook](https://manifesto-project.wzb.eu/down/data/2024a/codebooks/codebook_MPDataset_MPDS2024a.pdf)).
An audit of the current final file found 30 invalid inclusions: one CMP 202.2 left item and 29 CMP 605.2 right
items, all in development. They should be excluded rather than relabeled. At the time of this audit, correction
was open; §13.2 records the completed candidate. Removing those rows alone would yield 6,137 items (5,048
development and 1,089 category-heldout); those are expected counts only, not a completed artifact.

A normalized exact-text audit also found duplicate groups: two English groups/four rows and one group/two
rows in each of German, Chinese, Hindi, and Marathi. One English duplicate crosses development and
category-heldout and has conflicting CMP 407/107 labels within one manifesto. English and Hindi duplicate
groups also cross current development CV folds. Resolve or remove complete groups before confirmatory
scoring. Keep the already approved translations, stable item references, earlier two drops, and category
assignments where valid; do not claim 6,137 is the final count until duplicate handling is decided and
verified.

The category-heldout sample spans four categories but shares all 58 manifestos with development. It tests
held-out categories inside known manifestos, not generalization to unseen manifestos. Nine parties have
separate manifestos in two current development folds (18 manifestos), so current folds also do not establish
unseen-party generalization. All current layer and transfer scores are exploratory diagnostics: heldout scores
have already been viewed, and fit warnings require review. The current native/translated comparison is
confounded by country, party, topic, and translation choices. It cannot isolate a translation effect. Public
manifesto documents may have appeared in a base model's pretraining; probing heldout rows does not establish
that the model itself has never seen their source text.

The Cardiff tone table in §13 was computed on the 6,167-row version with invalid RILE rows. Recompute it
after dataset correction and duplicate handling. Even then, model-scored tone is only a limited diagnostic;
it does not establish that political content is tone-balanced or that tone confounding is absent. Previous
notes flag potentially problematic military examples; sample and review current examples before claiming
within-issue stance separation. Development issue-pair counts are military 529 (247 left, 282 right), welfare
533 (329 left, 204 right), and protectionism 366 (202 left, 164 right). Spanish welfare has only four
right-coded items against 129 left-coded items, so report subgroup support and do not treat its result as a
balanced test.

The Manifesto Project terms prohibit redistribution of the corpus without written permission
([terms](https://manifestoproject.wzb.eu/information/documents/terms_of_use)). The permission scope for
derived labels, translations, and metadata still needs confirmation before release; this document does not
assert that text-free metadata may be redistributed. These caveats supersede earlier descriptions in §§12–13
that treated the native/translated split or tone check as resolving those concerns.

### 13.2 Corrected dataset candidate (2026-09-27)

**FACT:** `data/rile_v2/` was built from the finalized RILE-v1 translations, without resampling or
retranslation. The builder and sampler now reject exact CMP 202.2, 605.2, and 703.2 before parent-code
mapping, following [MPDS2024a codebook p. 10](https://manifesto-project.wzb.eu/down/data/2024a/codebooks/codebook_MPDataset_MPDS2024a.pdf).
The candidate excludes 30 such items (one 202.2 and 29 605.2; no 703.2 was present). For duplicate
components linked by identical NFKC/strip/casefold text in any of the six languages, it removes both
items in the one conflicting-label/category/split pair and retains the smallest item ID in four
consistent pairs. No sentence text or metadata was edited on retained items. The prior two approved
drops remain absent. Source: `src/finalize_rile_translations.py --v2`, `data/rile_v2/manifest.json`.

**FACT:** The candidate has 6,131 aligned items: 5,044 development and 1,087 category holdout; 3,089
left and 3,042 right; 2,098 English, 2,048 German, and 1,985 Spanish originals; 66 manifestos and
57 party IDs. Each item still has all six language texts. Four bounded RILE-v2 optimization pilots are
complete, but no scored outer-test evaluation has run (see `requirements.md` §6). RILE-v1
tone and probe scores remain diagnostics for the original 6,167-item file and do not describe this candidate.

**LIMITATION:** The category holdout still shares manifesto documents with development. The four pilots
checked only two English layers per model on one outer-training partition; convergence at other layers
and in other languages is unresolved (`requirements.md` §6). Layer-selection, tone, lexical, and
translation checks remain open for the evaluation plan.

## 14. RILE-v2 saved-prediction uncertainty report (2026-09-30)

**FACT (2026-09-30):** This completed CPU analysis resamples saved predictions. Its records are the [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json), [table view](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/tables.html), and [bootstrap implementation](../src/bootstrap_rile_transfer.py). The numerical artifacts are local and Git-ignored; this section records no public release.

### 14.1 Cohort and measurement (2026-09-30)

**FACT:** Each cell contains 5,044 development items from 66 manifestos, with class counts 2,496 for class 0 and 2,548 for class 1, aligned across English, Spanish, German, Chinese, Hindi, and Marathi. The 1,087 category-holdout items are excluded from this analysis. Sources: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`method`); corrected-cohort record in §13.2.

**FACT:** Saved predictions use five outer folds separated by manifesto. The original selection used four inner folds and a budgeted layer search. Each source language and outer fold supplies its own selected layer and fitted probe, which are then applied unchanged to all six target languages. There are 36 source–target cells, 30 source/fold refits, and 181,584 saved predictions per model. A selected layer is the recorded choice within that search budget. Sources: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`provenance`, including the input and selection summaries); [bootstrap implementation](../src/bootstrap_rile_transfer.py) (selected-layer consistency and cell alignment checks).

**FACT:** Balanced accuracy is 100 × (recall for class 0 + recall for class 1) / 2, pooled over out-of-fold predictions. A cell pools all 5,044 items rather than averaging fold scores. The cross-language mean is the unweighted mean of the 30 off-diagonal cells; the own-language mean is the unweighted mean of six diagonal cells. Source-language and target-language means each average five off-diagonal cells. Sources: [bootstrap implementation](../src/bootstrap_rile_transfer.py) (`balanced`, `off`, and aggregate calculations); [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`models`).

### 14.2 Resampling settings and scope (2026-09-30)

**FACT:** The analysis draws 66 manifesto clusters with replacement 10,000 times using seed 20260930. Shared multinomial cluster counts keep all translations, cells, and models paired within each draw. Each draw recomputes pooled class recalls and the reported aggregates. Interval bounds are the 2.5th and 97.5th percentiles of those draws. Zero draws lacked either class; no zero-class draws were discarded. Sources: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`method`); [bootstrap implementation](../src/bootstrap_rile_transfer.py) (`weights`, `denominators`, `interval`).

**LIMITATION:** These pointwise 95% intervals condition on the saved fitted probes, selected layers, and evaluation splits. They exclude variation from retraining or repeating layer selection, are not simultaneous or adjusted for multiple comparisons, and do not measure party-independent generalization. Source: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`method`).

### 14.3 Full source–target tables (2026-09-30)

**FACT:** Values below are balanced accuracy percent, formatted as point estimate [lower, upper]. Rows identify training source language; columns identify evaluation target language. Diagonal cells use the same language. All displayed values are rounded to two decimal places; unrounded values are in [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`models.*.matrix`).

**OLMo — allenai/Olmo-3-7B-Instruct (2026-09-30)**

| Source ↓ / Target → | English | Spanish | German | Chinese | Hindi | Marathi |
|---|---|---|---|---|---|---|
| English | 64.72 [63.21, 66.31] | 52.78 [50.96, 54.97] | 50.74 [49.77, 51.87] | 53.74 [51.16, 56.02] | 51.97 [48.11, 55.42] | 53.39 [50.02, 56.52] |
| Spanish | 56.98 [54.19, 59.67] | 57.88 [56.35, 59.63] | 52.30 [50.39, 54.47] | 53.89 [51.23, 56.66] | 51.70 [46.88, 56.45] | 50.04 [46.56, 53.71] |
| German | 60.07 [58.09, 62.04] | 54.35 [52.02, 57.04] | 56.64 [54.60, 58.96] | 53.80 [50.71, 56.80] | 52.24 [50.33, 54.40] | 51.50 [49.58, 53.92] |
| Chinese | 60.37 [58.98, 61.55] | 54.59 [51.17, 57.99] | 50.31 [47.87, 52.92] | 60.80 [59.21, 62.33] | 51.17 [46.82, 55.63] | 51.39 [46.79, 56.41] |
| Hindi | 52.52 [50.13, 55.11] | 50.10 [47.15, 53.10] | 50.99 [47.41, 54.52] | 51.19 [49.31, 53.24] | 58.66 [57.18, 60.24] | 51.54 [48.40, 55.01] |
| Marathi | 56.58 [53.82, 59.23] | 51.51 [49.14, 53.77] | 50.43 [48.69, 52.42] | 53.05 [50.63, 55.36] | 55.15 [52.37, 57.47] | 54.93 [53.51, 56.66] |

**Qwen3.5 — Qwen/Qwen3.5-9B (2026-09-30)**

| Source ↓ / Target → | English | Spanish | German | Chinese | Hindi | Marathi |
|---|---|---|---|---|---|---|
| English | 69.95 [68.55, 71.39] | 59.59 [57.21, 61.81] | 59.02 [57.01, 60.87] | 62.82 [60.66, 64.68] | 57.07 [53.17, 60.77] | 53.15 [49.53, 56.21] |
| Spanish | 66.23 [64.17, 68.44] | 66.17 [64.49, 67.67] | 63.70 [61.14, 66.28] | 59.48 [55.78, 62.94] | 57.63 [53.75, 61.30] | 53.10 [51.26, 55.01] |
| German | 61.66 [58.99, 64.69] | 54.79 [50.57, 59.08] | 67.33 [65.54, 69.11] | 61.23 [58.76, 63.43] | 55.22 [50.22, 59.68] | 51.25 [50.01, 52.59] |
| Chinese | 61.44 [57.12, 66.18] | 51.44 [46.66, 56.91] | 53.11 [50.81, 55.88] | 69.68 [67.80, 71.44] | 56.38 [52.51, 59.48] | 55.63 [52.74, 58.50] |
| Hindi | 56.96 [52.65, 61.12] | 52.99 [47.95, 57.98] | 54.35 [51.17, 58.01] | 54.73 [50.08, 59.55] | 64.51 [62.86, 66.20] | 56.74 [52.00, 61.59] |
| Marathi | 56.47 [52.72, 60.01] | 56.04 [51.83, 59.63] | 54.69 [51.41, 57.83] | 60.07 [56.16, 63.39] | 59.13 [56.62, 61.21] | 58.67 [57.19, 60.34] |

**Ministral — mistralai/Ministral-8B-Instruct-2410 (2026-09-30)**

| Source ↓ / Target → | English | Spanish | German | Chinese | Hindi | Marathi |
|---|---|---|---|---|---|---|
| English | 72.82 [71.05, 74.42] | 63.65 [59.55, 67.06] | 59.29 [56.21, 62.00] | 57.76 [53.76, 61.97] | 55.40 [51.63, 59.38] | 55.50 [51.27, 59.14] |
| Spanish | 67.22 [65.34, 68.98] | 70.80 [69.12, 72.60] | 58.26 [56.43, 60.31] | 53.03 [51.95, 54.08] | 54.60 [51.72, 57.32] | 52.77 [49.73, 55.89] |
| German | 67.39 [65.42, 69.24] | 63.46 [61.06, 65.78] | 71.67 [69.67, 73.62] | 58.98 [56.49, 61.75] | 60.61 [57.61, 63.69] | 56.85 [54.33, 58.85] |
| Chinese | 54.31 [52.88, 55.70] | 52.87 [51.47, 54.22] | 52.84 [51.59, 54.09] | 71.62 [70.24, 73.04] | 62.63 [59.46, 65.45] | 51.38 [50.04, 52.77] |
| Hindi | 51.95 [49.58, 54.29] | 51.79 [49.27, 54.39] | 49.68 [47.11, 52.50] | 58.71 [56.86, 60.62] | 69.35 [67.99, 70.90] | 50.63 [47.96, 53.47] |
| Marathi | 60.50 [56.66, 64.18] | 59.65 [55.15, 63.88] | 61.35 [57.88, 64.82] | 59.90 [57.97, 61.67] | 60.18 [57.58, 62.82] | 63.19 [61.19, 65.24] |

**Gemma — google/gemma-2-9b-it (2026-09-30)**

| Source ↓ / Target → | English | Spanish | German | Chinese | Hindi | Marathi |
|---|---|---|---|---|---|---|
| English | 71.49 [70.01, 72.87] | 62.42 [60.81, 64.18] | 63.33 [60.98, 65.76] | 65.74 [63.71, 67.87] | 62.12 [59.42, 64.64] | 59.93 [57.54, 62.18] |
| Spanish | 67.89 [65.89, 69.90] | 69.66 [68.34, 70.89] | 63.43 [60.09, 66.93] | 66.20 [64.13, 68.27] | 65.50 [62.93, 67.82] | 63.47 [60.74, 66.31] |
| German | 68.21 [66.47, 69.87] | 64.30 [61.99, 66.82] | 70.47 [68.96, 72.05] | 65.35 [63.66, 67.14] | 63.98 [61.75, 66.47] | 61.94 [59.32, 64.49] |
| Chinese | 67.26 [65.63, 68.64] | 61.50 [58.83, 63.79] | 60.64 [58.72, 62.59] | 69.87 [68.48, 71.26] | 65.19 [63.01, 67.32] | 61.74 [59.97, 63.61] |
| Hindi | 67.21 [64.85, 69.40] | 61.64 [59.11, 64.24] | 63.49 [60.23, 66.24] | 64.79 [61.41, 67.91] | 68.18 [66.55, 69.80] | 61.14 [57.96, 64.39] |
| Marathi | 68.41 [66.11, 70.68] | 63.82 [61.12, 66.54] | 66.11 [63.62, 68.40] | 63.95 [61.50, 66.53] | 65.80 [63.84, 67.79] | 66.13 [64.70, 67.55] |

### 14.4 Aggregate tables (2026-09-30)

**FACT:** These are unweighted cell means with intervals recomputed from paired draws, in percent. Source: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`off_diagonal_mean`, `diagonal_mean`, `source_means`, `target_means`); [bootstrap implementation](../src/bootstrap_rile_transfer.py) (aggregate calculations).

| Model | Cross-language mean: 30 cells | Own-language mean: 6 cells |
|---|---|---|
| OLMo | 53.01 [52.44, 53.65] | 58.94 [57.98, 60.02] |
| Qwen3.5 | 57.20 [56.24, 58.15] | 66.05 [65.03, 67.15] |
| Ministral | 57.44 [56.77, 58.13] | 69.91 [68.54, 71.29] |
| Gemma | 64.22 [63.21, 65.25] | 69.30 [68.27, 70.33] |

**OLMo: language means (2026-09-30)**

| Language | Mean as source: 5 transfers | Mean as target: 5 transfers |
|---|---|---|
| English | 52.52 [51.31, 53.62] | 57.31 [56.43, 58.16] |
| Spanish | 52.98 [50.86, 55.14] | 52.66 [51.39, 54.13] |
| German | 54.39 [52.90, 56.18] | 50.95 [49.72, 52.30] |
| Chinese | 53.56 [50.95, 56.39] | 53.13 [52.28, 53.99] |
| Hindi | 51.27 [48.85, 53.85] | 52.44 [51.46, 53.32] |
| Marathi | 53.34 [52.34, 54.34] | 51.57 [50.53, 52.76] |

**Qwen3.5: language means (2026-09-30)**

| Language | Mean as source: 5 transfers | Mean as target: 5 transfers |
|---|---|---|
| English | 58.33 [56.33, 60.09] | 60.55 [59.38, 61.90] |
| Spanish | 60.03 [58.51, 61.45] | 54.97 [53.67, 56.40] |
| German | 56.83 [55.27, 58.36] | 56.97 [55.99, 58.11] |
| Chinese | 55.60 [53.57, 57.78] | 59.67 [58.19, 61.17] |
| Hindi | 55.16 [51.95, 58.40] | 57.09 [54.84, 59.04] |
| Marathi | 57.28 [54.37, 59.84] | 53.97 [51.86, 56.10] |

**Ministral: language means (2026-09-30)**

| Language | Mean as source: 5 transfers | Mean as target: 5 transfers |
|---|---|---|
| English | 58.32 [56.34, 60.24] | 60.27 [59.06, 61.49] |
| Spanish | 57.17 [56.12, 58.22] | 58.28 [57.11, 59.52] |
| German | 61.46 [60.16, 62.69] | 56.29 [55.08, 57.45] |
| Chinese | 54.81 [53.55, 55.94] | 57.68 [56.50, 58.75] |
| Hindi | 52.55 [51.04, 54.25] | 58.68 [57.41, 59.90] |
| Marathi | 60.32 [58.34, 62.34] | 53.42 [52.58, 54.12] |

**Gemma: language means (2026-09-30)**

| Language | Mean as source: 5 transfers | Mean as target: 5 transfers |
|---|---|---|
| English | 62.71 [61.57, 63.83] | 67.80 [66.63, 68.96] |
| Spanish | 65.30 [63.55, 67.04] | 62.74 [61.59, 63.95] |
| German | 64.76 [63.31, 66.39] | 63.40 [62.07, 64.64] |
| Chinese | 63.27 [61.95, 64.47] | 65.21 [63.81, 66.72] |
| Hindi | 63.65 [62.16, 65.22] | 64.52 [63.25, 65.72] |
| Marathi | 65.62 [63.79, 67.39] | 61.64 [60.47, 62.82] |

### 14.5 Paired numerical differences (2026-09-30)

**FACT:** Each difference subtracts the named models’ 30-cell cross-language means within the same manifesto draw. Units are percentage points. Source: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`paired_model_differences`); [bootstrap implementation](../src/bootstrap_rile_transfer.py) (paired difference calculation).

| Difference | Point [95% interval], percentage points |
|---|---|
| Qwen3.5 minus OLMo | 4.19 [2.93, 5.41] |
| Ministral minus OLMo | 4.43 [3.71, 5.10] |
| Gemma minus OLMo | 11.21 [10.07, 12.30] |
| Ministral minus Qwen3.5 | 0.23 [-0.91, 1.33] |
| Gemma minus Qwen3.5 | 7.01 [6.28, 7.70] |
| Gemma minus Ministral | 6.78 [5.76, 7.81] |

**FACT:** All source-own-language minus transfer differences and their intervals are retained in [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`models.*.source_own_minus_transfer_pp`). For source s and target t, the recorded quantity is BA(s → s) − BA(s → t), in percentage points. The diagonal quantity is zero by construction. Source: [bootstrap implementation](../src/bootstrap_rile_transfer.py) (difference calculation).

### 14.6 Provenance and recorded checks (2026-09-30)

**FACT:** Model snapshots and input paths/hashes are recorded in [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`provenance`). Snapshot identifiers are reproduced below.

| Model | Model identifier | Snapshot revision |
|---|---|---|
| OLMo | `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| Qwen3.5 | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Ministral | `mistralai/Ministral-8B-Instruct-2410` | `2f494a194c5b980dfb9772cb92d26cbb671fce5a` |
| Gemma | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |

**FACT:** The bootstrap script SHA-256 is `6ed5068458cf0689ed0ee69b366d3762d8ca5169a9957c2f11fd41470d5e759b`. The results JSON SHA-256 is `7a67f94b24120001dc5445c50589001f4b742e22b0bedeae45ecca48f032887e`. Per-model prediction, summary, selection-summary, and dataset-input hashes are in [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`provenance`).

**FACT:** The completed analysis is dated 2026-09-30 with 10,000 draws and seed 20260930. It used CPU analysis of saved local files, with no Slurm job or cloud resource launched for this bootstrap. Sources: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`method`); [bootstrap implementation](../src/bootstrap_rile_transfer.py) (CPU calculations and elapsed-time output).

**FACT:** Reproduction command from the repository root, using the saved local inputs:

```sh
UV_CACHE_DIR=/private/tmp/uv-rile-bootstrap-cache uv run --no-project --offline python src/crosslingual-political-repr/src/bootstrap_rile_transfer.py --draws 10000 --seed 20260930 --output gcp-workspace/rile_v2_source_matrix_bootstrap
```

**FACT:** The saved analysis checks all 144 pooled cell estimates against input summaries to tolerance 1e-12; unique items, item/manifesto/fold/label alignment across models and cells; selected-layer consistency; and one fold per manifesto. Its small hand-computed fixture checks class recall, repeated-cluster weights, and an identical paired difference. Sources: [numerical results and provenance](../../../gcp-workspace/rile_v2_source_matrix_bootstrap/bootstrap_results.json) (`method.point_checks`, `fixture_checks`); [bootstrap implementation](../src/bootstrap_rile_transfer.py) (assertions and `fixture`).

## 15. Matched text controls (2026-09-30)

**FACT:** The local CPU run completed on 2026-09-30: 90 fits, no convergence warnings, and 75.06 seconds recorded runtime. It used 5,044 development items from 66 manifestos (class 0: 2,496; class 1: 2,548), excluding the 1,087 category-held-out items. The labels remain category-derived RILE formula sides. Sources: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`status`, `elapsed_seconds`, `fits`, `matrices`); [control runner](../src/run_rile_text_controls.py) (`load`).

**FACT:** All three controls use the exact item-to-fold assignment saved in the OLMo English-to-English prediction file, with fold 0–4 item counts 1,011 / 1,007 / 1,009 / 1,007 / 1,010 and manifesto counts 12 / 14 / 13 / 14 / 13. Each manifesto occupies one fold, and all translations of an item share that fold. Sources: [control runner](../src/run_rile_text_controls.py) (`load`); [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`provenance.fold_reference`).

**FACT:** Each control trains six source-language classifiers over five folds, then applies each fitted classifier to six target languages. Each family saves 181,584 prediction rows across 36 cells; each cell contains 5,044 unique items. The three families save 544,752 rows across 108 cells; these rows reuse the same 5,044 underlying items. Saved fields are source language, target language, fold, item identifier, manifesto identifier, true label, predicted label, and decision score. Sources: [control runner](../src/run_rile_text_controls.py) (`main`); [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`prediction_files`, `matrices`).

### 15.1 Fixed settings and measurement (2026-09-30)

**FACT:** All fits use logistic regression with C = 1, `lbfgs`, `max_iter=3000`, and `random_state=42`, without hyperparameter tuning. Vocabulary, inverse-document-frequency weights, and numeric scaling are fitted using only the source-language training items in the relevant fold. The same fitted classifier evaluates all targets. Sources: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`settings`); [control runner](../src/run_rile_text_controls.py) (`main`).

| Control | Features | Recorded training-feature count, minimum–maximum |
|---|---|---|
| Character 3–5-grams | char 3-5, lowercase, raw tf, smooth idf, l2 norm | 78,150–338,627 |
| Word 1–2-grams | word 1-2, Unicode LMN runs; Chinese jieba HMM=False; lowercase; raw tf, smooth idf, l2 norm | 50,301–66,849 |
| Length and final punctuation | character length, whitespace segments, segments/characters, seven final punctuation flags and Unicode final punctuation flag; training-only StandardScaler | 11–11 |

**FACT:** Character and word features use raw term frequency, smoothed inverse-document frequency, and L2 normalization. Word preprocessing lowercases each language independently: Unicode letter/mark/number runs for English, Spanish, German, Hindi, and Marathi; fixed-dictionary jieba segmentation with `HMM=False` for Chinese. Hindi and Marathi combining marks remain within tokens. The 11 numeric features are character length, whitespace-segment count, segment/character ratio, seven final-character flags for `.。`, `?？`, `!！`, `:：`, `;；`, `,，`, and `।॥`, plus a Unicode final-punctuation flag; numeric features use source-training-only standardization. Sources: [control runner](../src/run_rile_text_controls.py) (`word_tokens`, `surface`); [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`settings`, `fits`).

**FACT:** The legacy character recipe is reused in the new RILE-specific runner; the older study script remains separate. The old terminal-token-ID scalar mode is omitted here. Sources: [control runner](../src/run_rile_text_controls.py); [older surface-control runner](../src/run_text_surface_controls.py). **LIMITATION / OPEN:** A future token-identity control would require a categorical representation and separately defined tokenization; no result from that control is reported in this section.

**FACT:** Balanced accuracy is the mean of class-0 recall and class-1 recall after pooling the five out-of-fold predictions for each source–target cell. Saved summaries also contain pooled accuracy, class-1 F1, class counts, feature counts, per-fit iterations/warnings, fitted-coefficient hashes, vocabulary/IDF or scaler hashes, and matched-feature diagnostics. Sources: [control runner](../src/run_rile_text_controls.py) (`main`); [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`matrices`, `fits`).

### 15.2 Balanced accuracy matrices with pointwise intervals (2026-09-30)

**FACT:** The tables report balanced accuracy in percent, followed by a pointwise 95% percentile interval. The analysis samples the same 66 manifestos with replacement in 10,000 shared draws, seed 20260930; translations and all seven saved control/probe results remain paired. Each cell pools 5,044 held-out predictions. Sources: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`method`, `controls`); [bootstrap runner](../src/bootstrap_rile_text_controls.py).

**FACT — Character 3–5-grams:** Rows are source languages; columns are target languages. Source: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`controls.char.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 68.72 [66.68, 70.84] | 57.73 [55.54, 59.99] | 53.89 [52.39, 55.37] | 50.70 [50.13, 51.33] | 50.88 [50.13, 51.65] | 50.55 [49.88, 51.21] |
| es | 59.77 [58.12, 61.44] | 67.97 [66.33, 69.72] | 53.28 [50.69, 55.86] | 50.24 [49.81, 50.62] | 50.22 [49.84, 50.61] | 50.25 [49.78, 50.70] |
| de | 54.66 [52.77, 56.66] | 54.43 [52.13, 57.08] | 68.10 [66.36, 69.86] | 50.74 [50.15, 51.33] | 50.85 [50.15, 51.58] | 50.52 [49.74, 51.34] |
| zh | 49.09 [45.72, 52.77] | 49.20 [45.75, 53.14] | 48.93 [45.75, 52.28] | 63.05 [61.02, 65.24] | 47.83 [43.38, 52.31] | 47.31 [42.80, 51.84] |
| hi | 50.92 [49.40, 52.57] | 50.00 [49.76, 50.26] | 49.95 [49.25, 50.82] | 50.02 [49.87, 50.20] | 68.36 [66.61, 70.17] | 61.74 [60.53, 63.10] |
| mr | 50.08 [49.46, 50.80] | 50.34 [49.87, 50.84] | 50.63 [49.57, 51.69] | 50.27 [49.78, 50.74] | 62.00 [60.35, 63.64] | 68.64 [67.01, 70.20] |

**FACT — Word 1–2-grams:** Rows are source languages; columns are target languages. Source: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`controls.word.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 67.37 [65.40, 69.31] | 51.96 [48.08, 56.46] | 50.42 [47.54, 53.48] | 50.83 [49.88, 51.74] | 51.13 [50.24, 52.00] | 50.29 [49.63, 50.86] |
| es | 52.45 [50.95, 53.99] | 67.52 [65.62, 69.49] | 50.14 [46.72, 53.45] | 50.60 [49.91, 51.26] | 50.92 [50.34, 51.50] | 50.19 [49.82, 50.56] |
| de | 49.61 [46.76, 52.50] | 49.56 [46.10, 53.43] | 61.23 [59.46, 63.30] | 47.07 [41.53, 52.95] | 47.53 [42.04, 53.32] | 46.67 [40.89, 52.87] |
| zh | 49.28 [45.26, 53.62] | 50.15 [48.69, 51.99] | 47.89 [43.02, 52.90] | 68.09 [66.24, 70.07] | 47.58 [42.21, 53.08] | 47.34 [41.45, 53.17] |
| hi | 50.48 [48.31, 52.87] | 50.46 [50.02, 50.83] | 49.16 [45.40, 53.02] | 50.55 [49.74, 51.34] | 66.26 [64.52, 67.99] | 59.21 [57.65, 60.92] |
| mr | 48.55 [45.11, 52.46] | 47.88 [43.75, 53.02] | 48.82 [45.50, 52.27] | 46.88 [41.25, 53.67] | 58.95 [57.07, 60.90] | 64.36 [62.86, 66.08] |

**FACT — Length and final punctuation:** Rows are source languages; columns are target languages. Source: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`controls.surface.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 47.94 [45.35, 50.92] | 48.01 [45.38, 50.80] | 48.33 [46.01, 50.53] | 49.61 [44.53, 54.84] | 49.75 [48.79, 50.83] | 48.05 [45.77, 50.35] |
| es | 48.01 [45.33, 50.96] | 48.15 [45.47, 51.01] | 48.56 [46.69, 50.40] | 50.07 [49.24, 50.90] | 49.03 [47.61, 50.47] | 48.34 [46.67, 49.98] |
| de | 48.26 [46.10, 50.48] | 48.42 [45.99, 51.02] | 46.44 [42.98, 50.38] | 47.02 [41.37, 53.60] | 50.40 [49.49, 51.41] | 47.26 [44.32, 50.53] |
| zh | 50.03 [47.66, 52.42] | 50.03 [48.09, 52.08] | 49.78 [46.91, 52.67] | 47.35 [44.40, 50.72] | 50.02 [49.06, 51.06] | 49.67 [47.31, 52.16] |
| hi | 50.41 [49.33, 51.72] | 50.17 [48.99, 51.47] | 50.49 [49.10, 52.17] | 46.79 [41.41, 53.28] | 47.71 [44.41, 51.61] | 50.25 [48.98, 51.70] |
| mr | 47.79 [45.01, 50.65] | 47.65 [44.83, 50.72] | 47.00 [43.22, 51.58] | 49.73 [49.43, 50.04] | 50.14 [49.21, 51.13] | 47.37 [43.86, 51.49] |

**FACT:** Same-language means average the six diagonal cells; cross-language means average the 30 off-diagonal cells with equal cell weights. Source: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`controls.*.diagonal_mean`, `off_diagonal_mean`).

| Control | Same-language mean, percent [95% interval] | Cross-language mean, percent [95% interval] |
|---|---|---|
| Character 3–5-grams | 67.47 [65.98, 69.03] | 51.90 [50.93, 52.91] |
| Word 1–2-grams | 65.81 [64.29, 67.48] | 50.08 [47.91, 52.42] |
| Length and final punctuation | 47.49 [44.62, 50.79] | 48.97 [47.42, 50.76] |

### 15.3 Paired probe-minus-control means (2026-09-30)

**FACT:** Each entry subtracts the control’s 30-cell cross-language mean from the named saved probe’s 30-cell mean within the same manifesto draw. Units are percentage points; intervals are pointwise 95% percentile intervals. All predictions use identical item, manifesto, fold, and label keys. Sources: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`probe_minus_control_pp`, `provenance`); [bootstrap runner](../src/bootstrap_rile_text_controls.py).

| Saved probe minus control | Character 3–5-grams | Word 1–2-grams | Length/final punctuation |
|---|---|---|---|
| OLMo | 1.11 [0.31, 1.89] | 2.93 [0.98, 4.72] | 4.04 [2.43, 5.42] |
| Qwen3.5 | 5.30 [3.72, 6.85] | 7.12 [4.07, 9.99] | 8.24 [5.90, 10.35] |
| Ministral | 5.54 [4.33, 6.68] | 7.35 [5.02, 9.43] | 8.47 [6.50, 10.18] |
| Gemma | 12.32 [10.73, 13.87] | 14.13 [11.25, 16.87] | 15.25 [12.91, 17.36] |

**FACT:** Source-language means, target-language means, and source-own-language minus each transfer-cell differences are retained in [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`controls.*.source_means`, `target_means`, `source_own_minus_transfer_pp`).

### 15.4 Empty target feature rows (2026-09-30)

**FACT:** Each entry is the percentage of target items with no matched feature in the source-trained vocabulary, calculated as the sum of empty rows over the five source folds divided by 5,044. This measures vocabulary overlap; it does not mean that the text was missing or that tokenization failed. Sources: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`fits.*.targets.*.empty_rows`); [control runner](../src/run_rile_text_controls.py) (target sparse feature counts).

**FACT — Character 3–5-grams empty rows, percent:** Source: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`fits`, family `char`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 0.00 | 0.00 | 0.00 | 84.14 | 77.93 | 87.13 |
| es | 0.00 | 0.00 | 0.00 | 84.58 | 78.21 | 87.17 |
| de | 0.00 | 0.00 | 0.00 | 84.60 | 77.93 | 87.19 |
| zh | 0.00 | 0.00 | 0.00 | 0.14 | 82.30 | 91.40 |
| hi | 0.00 | 0.00 | 0.00 | 85.41 | 0.00 | 0.00 |
| mr | 0.00 | 0.00 | 0.00 | 87.79 | 0.00 | 0.00 |

**FACT — Word 1–2-grams empty rows, percent:** Source: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`fits`, family `word`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 0.00 | 4.44 | 11.00 | 84.16 | 82.91 | 93.00 |
| es | 2.84 | 0.00 | 17.45 | 84.54 | 83.86 | 93.81 |
| de | 0.20 | 6.62 | 0.00 | 84.46 | 83.74 | 93.66 |
| zh | 7.73 | 18.62 | 42.61 | 0.00 | 84.56 | 94.61 |
| hi | 4.84 | 10.35 | 34.18 | 85.19 | 0.00 | 4.70 |
| mr | 7.30 | 13.07 | 35.33 | 88.12 | 0.77 | 0.00 |

### 15.5 Provenance, reproduction, and recorded checks (2026-09-30)

**FACT:** Runtime/dependency metadata: Python 3.12.7; NumPy 2.5.3; scikit-learn 1.9.1; jieba 0.42.1; Unicode 15.0.0. Source: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`provenance`).

| Provenance field | SHA-256 |
|---|---|
| Run `input_sha256` | `fe6eb10383707b44a0fdfbbf65b95f6258c0ef459c30fec8240e8449f01700ca` |
| Run `fold_reference_sha256` | `2b7a914cf6b3710eb3f1d227bae772a448db3f26895ebd504062496206d694b7` |
| Run `code_sha256` | `2b9cacd5e14b4a8947fe53caba23df529f24b9b6e31bfa9bf401bd32fd8ec67f` |
| Run `jieba_dictionary_sha256` | `7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8` |
| Bootstrap `code_sha256` | `8738624ccf2423cd6d41bbe65b06b53abd28b321f15750762c7cfa195716a3ed` |
| Bootstrap `weights_sha256` | `0ca701338c292f477d711147b48e1d794a9f962c5f3aceb4049528866b31b6a7` |
| Bootstrap helper `bootstrap_rile_transfer.py` | `6ed5068458cf0689ed0ee69b366d3762d8ca5169a9957c2f11fd41470d5e759b` |
| Bootstrap helper `run_rile_text_controls.py` | `2b9cacd5e14b4a8947fe53caba23df529f24b9b6e31bfa9bf401bd32fd8ec67f` |
| `char_predictions.jsonl.gz` | `63fe92310c5e06cec1668c9ab80ba12445bee1c9fb78d58f343fc9a360df188d` |
| `word_predictions.jsonl.gz` | `b34daf5403ee064af04e6b879ad544ff86fafdf0307f2fd08679781c4c4d7c4a` |
| `surface_predictions.jsonl.gz` | `3d1477574d15cdcd6c9a64d8957e1a857d91b3b1663d0ea2e76e6ddd3ca0186f` |

**FACT:** Dataset and fold-reference paths, per-fit hashes, three prediction-file paths/hashes, and all four saved probe prediction paths/hashes are retained in [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`provenance`, `fits`, `prediction_files`) and [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`provenance`). The local output folders are ignored infrastructure/work files. [HTML tables](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/tables.html) provide an alternate view of the three matrices.

**FACT:** Reproduction commands from the repository root use pinned CPU dependencies and distinct output paths; existing output paths are rejected by the runners. Sources: [control runner](../src/run_rile_text_controls.py); [bootstrap runner](../src/bootstrap_rile_text_controls.py).

```sh
UV_CACHE_DIR=/tmp/rile-text-control-uvcache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --offline --with numpy==2.5.3 --with scikit-learn==1.9.1 --with jieba==0.42.1 python src/crosslingual-political-repr/src/run_rile_text_controls.py --output gcp-workspace/rile_v2_text_controls/full-20260930-luna
UV_CACHE_DIR=/tmp/rile-text-control-uvcache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --offline --with numpy==2.5.3 --with scikit-learn==1.9.1 --with jieba==0.42.1 python src/crosslingual-political-repr/src/bootstrap_rile_text_controls.py --controls gcp-workspace/rile_v2_text_controls/full-20260930-luna --output gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap
```

**FACT:** Execution used local CPU files, with no Slurm job or cloud resource launched for this stage. The recorded smoke fit used English source, fold 0, for all three controls: 3.63 seconds, no convergence warnings, and 18 aligned source–target cells of 1,011 items each. Fixtures checked training-only vocabulary, Chinese segmentation, Hindi combining marks, final punctuation, and rejection of duplicate/mismatched fold records. Sources: [smoke summary](../../../gcp-workspace/rile_v2_text_controls/smoke-20260930-sol/summary.json); [control runner](../src/run_rile_text_controls.py) (`fixture`, `load`).

**FACT:** Post-run verification reported 90 distinct fits, no warnings, no fits with zero features, 108 cells each with 5,044 unique aligned items, one fold per manifesto, and class counts 2,496 / 2,548. Saved prediction hashes matched the run summary, and the four input probe artifacts remained unchanged. The bootstrap recomputed all control and probe cell point estimates against input summaries to tolerance 1e-12. Durable sources: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`fits`, `prediction_files`, `matrices`); [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`provenance`); [bootstrap runner](../src/bootstrap_rile_text_controls.py) (alignment and point-check assertions).

### 15.6 Limits and incomplete controls (2026-09-30)

**LIMITATION:** These results measure prediction of the dataset’s category-derived labels from lexical and surface features under the saved manifesto folds. They do not establish sentence-level ideology, eliminate topic/category or translation cues, or establish party-independent generalization. The intervals condition on saved fitted classifiers, probes, selected probe layers, and splits; they exclude retraining uncertainty and are not simultaneous or adjusted for multiple comparisons. Sources: [bootstrap results](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna-bootstrap/bootstrap_results.json) (`method.limitation`, `interval`); §14 and the dataset-label definition in this document.

**LIMITATION:** A `tee` stdout-log attempt used a nonexistent parent directory, so no persistent stdout log was saved for the full run. Live completion output was observed; the completed summary preserves all 90 fit records, warnings, iterations, elapsed time, and output hashes. No rerun was performed solely for stdout logging. Source: [run summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json); [issue record](../../../docs/bugs-squashed.md) (2026-09-30 matched-control entry).

**INCOMPLETE / OPEN:** Categorical terminal-token-identity testing remains outside this run. Character/word empty-row diagnostics are saved without confidence intervals; their denominator and fold aggregation are given in §15.4.

## 16. Frozen category holdout: approved pre-run protocol (2026-09-30)

**FACT / FROZEN METHOD (2026-09-30):** The approved evaluation uses the canonical 1,087 held-out items, with category-derived labels 0 / 1 counted as 593 / 494. The four excluded categories are 107 (287 items), 506 (306), 601 (271), and 603 (223). Their 58 manifestos all map to the existing development fold assignment; held-out fold counts are 165, 167, 178, 251, and 326. No held-out labels select layers, tune classifiers, or change this protocol. Sources: [canonical rows](../data/rile_v2/final_translations.jsonl); [evaluation runner](../src/evaluate_rile_category_holdout.py) (`load_cohorts`, `model_run`, `control_run`).

**FACT / FROZEN METHOD (2026-09-30):** Each held-out item is scored only by the single saved fold probe whose training excluded that item's manifesto. The other four fold probes may have trained on development statements from that document and are never ensembled for this evaluation. This protocol supersedes the generic shared-document leakage caveat for this specific evaluation: reuse of documents across development and held-out categories does not put a prediction's document into its classifier's training rows. It still does not establish party independence or expert-verified sentence ideology. Sources: [evaluation runner](../src/evaluate_rile_category_holdout.py) (`load_cohorts`, per-fold scoring and control training-document exclusion assertion); [original replay helper](../src/multilingual_layerwise_probe.py) (`run_rile_source_matrix`, `nested_group_folds`).

**FACT / FROZEN METHOD (2026-09-30):** Four model families use 36 source–target pairs each: OLMo, Qwen3.5, Ministral, and Gemma. All 30 saved fold/source layers and all 90 NPZ parameter arrays are reused unchanged per model; there is no refitting or layer selection. Exact cached revisions are pinned in the evaluation runner. **SUPERSEDED batch-size statement (2026-09-30):** The pre-run record stated, “Extraction uses the original batch size 4.” The original selection summaries instead record batch size 8; the corrected runner derives that setting from the hash-verified summaries (see §16.6). Tokenizer options, final-token hidden states, float16 storage followed by float32 scoring, and zero-threshold binary prediction with the saved class order remain frozen. Full evaluation saves 39,132 prediction rows per model, with item/document IDs, fold, source/target languages, category code, label, prediction, decision margin, and selected layer; raw statement text is omitted. Source: [evaluation runner](../src/evaluate_rile_category_holdout.py) (`PINS`, `model_run`, `prediction`); the four replay artifact directories registered in [existing bootstrap helper](../src/bootstrap_rile_transfer.py) (`INPUTS`).

**FACT / FROZEN METHOD (2026-09-30):** The three CPU controls reproduce the existing character, word, and surface classifiers on development rows only, in the same row order and fold assignments. Full evaluation repeats 90 fits with unchanged segmentation, vectorizers, scaler, and classifier settings. Every coefficient, intercept, vocabulary/IDF, and scaler hash must match its frozen fit record exactly. Python 3.12.7, NumPy 2.5.3, scikit-learn 1.9.1, jieba 0.42.1, the dictionary hash, and Unicode version are fixed. A mismatch or fit warning stops evaluation with `needs_review`; no silent substitute classifier is accepted. Empty-row and matched-feature diagnostics are saved for the character and word controls. Sources: [frozen control summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json); [control helper](../src/run_rile_text_controls.py); [evaluation runner](../src/evaluate_rile_category_holdout.py) (`control_run`).

**FACT / FROZEN METHOD (2026-09-30):** Before accepting any model output, extraction also replays the first eight development rows in original JSONL order. **SUPERSEDED batch-context statement (2026-09-30):** The pre-run record described “the original two batches of four”; the corrected setting preserves one original batch of eight (see §16.6). All six target languages and corresponding saved source probes must reproduce predictions exactly. Decision margins must satisfy the predeclared `atol=0.001`, `rtol=0.0001`: the same snapshot, storage precision, and batch contexts should agree, with only small GPU numerical drift allowed. Smoke held-out sampling chooses the first two item IDs in every present fold/label stratum; the CPU smoke limits fits and held-out rows to English source, fold 0. Smoke and full outputs use distinct folders and cannot overwrite existing outputs. Sources: [evaluation runner](../src/evaluate_rile_category_holdout.py) (`sample`, `model_run`, `main`).

**FACT / FROZEN METHOD (2026-09-30):** Balanced accuracy is the mean of the two pooled class recalls. The analysis checks all 252 held-out cells against their summaries and all 252 original development cells against the saved summaries before computing intervals. It samples 58 manifestos with replacement 10,000 times, seed 20260930; every family, translation, and both cohorts receive the same weights. Outputs include seven matrices, equal-weight means over 36 cells, 30 cross-language cells and six same-language cells, source/target cross-language means, 12 probe-minus-control comparisons, and seven held-out-minus-development comparisons. Sources: [analysis runner](../src/bootstrap_rile_category_holdout.py) (`read_cells`, `counts`, `aggregates`, `main`).

**LIMITATION / FROZEN METHOD (2026-09-30):** The development comparison is restricted to the same 58 documents, using different statements/categories. Its difference is a descriptive category shift, not a paired same-item or causal effect. The original 66-document development matrices are retained only as context. Pointwise 95% percentile intervals condition on saved classifiers, selected layers, and folds; they exclude retraining uncertainty and are not simultaneous or adjusted for multiple comparisons. Four fixed categories do not support inference about a wider category population. Sources: [analysis runner](../src/bootstrap_rile_category_holdout.py) (`method`, `full_66_manifesto_development_context_only`); the category-label definition in this document.

**INCOMPLETE / PRE-RUN CARD (2026-09-30):** No full category-holdout result is recorded here yet. Expected CPU runtime is approximately 90 seconds based on the prior matched-control run; the heuristic for four GPU model runs is 20–45 minutes and must be checked against actual smoke timings. GPU extraction processes 6,522 held-out statements per model plus 48 development-parity inputs, in six sequential language memmaps. Temporary activation files stay on VM scratch and are deleted after each language; no local activation cache or model download is created. Known risks are insufficient pinned model cache, GPU numerical parity failure, and exact control fit-hash differences. Sources: [prior CPU summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`elapsed_seconds`); [evaluation runner](../src/evaluate_rile_category_holdout.py) (`model_run`, `control_run`).

Reproduction commands from the repository root (use a new run directory; full GPU execution requires the existing pinned VM environment):

```sh
UV_CACHE_DIR=/tmp/rile-text-control-uvcache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --offline --with numpy==2.5.3 --with scikit-learn==1.9.1 --with jieba==0.42.1 python src/crosslingual-political-repr/src/evaluate_rile_category_holdout.py --model controls --smoke --output gcp-workspace/rile_v2_category_holdout/smoke-controls
# Repeat with --model OLMo, Qwen3.5, Ministral, Gemma in the existing GPU environment.
# Then run each full mode without --smoke, into its own model-named folder.
UV_CACHE_DIR=/tmp/rile-text-control-uvcache uv run --offline --with numpy==2.5.3 python src/crosslingual-political-repr/src/bootstrap_rile_category_holdout.py --holdout gcp-workspace/rile_v2_category_holdout/full --output gcp-workspace/rile_v2_category_holdout/bootstrap
```

### 16.1 Partial CPU results and campaign status (2026-09-30)

**FACT / PARTIAL (2026-09-30):** The character, word, and surface controls completed on the frozen 1,087-item category holdout. Each family has 39,132 saved predictions across 36 source–target cells; every cell contains 593 label-0 and 494 label-1 items from 58 manifestos. The CPU run took 36.351 seconds, with 90 reproduced fits and zero fit warnings. Sources: [CPU run summary](../../../gcp-workspace/rile_v2_category_holdout/full-20260930-luna/controls/summary.json) (`status`, `fits`, `prediction_files`, `matrices`, `elapsed_seconds`); [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`status`, `method`, `provenance`).

**INCOMPLETE / PARTIAL (2026-09-30):** This is a controls-only result, not a completed category-holdout campaign. The four GPU model evaluations have not launched, so the seven-family comparison and probe-minus-control intervals remain unavailable. The pre-run protocol above is retained as frozen; the tables below report only the three completed controls. Source: current work-session execution record (2026-09-30); [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`status=PARTIAL_CONTROLS_ONLY`, three `families`).

### 16.2 Three held-out matrices (2026-09-30)

**FACT (2026-09-30):** Rows are source training languages; columns are target evaluation languages. Values are pooled balanced accuracy (%) [pointwise 95% percentile interval], rounded to two decimals. Intervals use 10,000 paired draws of 58 manifestos, seed 20260930, with identical weights across controls, translations, and both cohorts; no zero-class draws were discarded. Source: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`method`, `families.*.holdout.matrix`).

#### Character control (2026-09-30)

**FACT (2026-09-30):** Source: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`families.char.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 51.41 [48.45, 54.88] | 49.77 [45.84, 54.18] | 50.93 [47.71, 53.45] | 50.57 [49.52, 51.66] | 50.69 [49.18, 51.96] | 50.44 [48.34, 52.02] |
| es | 50.91 [47.73, 53.72] | 54.90 [52.21, 58.23] | 46.99 [42.87, 52.01] | 50.56 [49.93, 51.19] | 50.91 [49.91, 52.01] | 50.59 [49.52, 51.48] |
| de | 50.77 [46.97, 53.85] | 49.81 [45.38, 54.82] | 49.37 [46.24, 52.70] | 50.71 [49.85, 51.77] | 50.84 [49.66, 52.17] | 50.96 [49.83, 52.04] |
| zh | 42.41 [33.92, 54.32] | 45.90 [37.54, 57.48] | 41.87 [35.14, 50.28] | 51.39 [46.13, 56.70] | 38.64 [28.49, 53.54] | 36.45 [24.87, 53.09] |
| hi | 52.07 [47.51, 56.99] | 50.13 [49.83, 50.42] | 49.43 [48.01, 51.10] | 50.07 [49.79, 50.38] | 51.82 [48.06, 54.79] | 54.94 [52.64, 57.14] |
| mr | 49.07 [47.82, 50.47] | 49.46 [48.77, 50.29] | 48.92 [47.37, 50.38] | 49.48 [48.82, 50.41] | 46.23 [42.27, 51.62] | 50.62 [47.06, 53.99] |

#### Word control (2026-09-30)

**FACT (2026-09-30):** Source: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`families.word.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 49.83 [46.53, 54.07] | 50.19 [43.21, 56.67] | 49.75 [44.11, 56.97] | 50.07 [48.80, 51.82] | 50.64 [49.46, 51.97] | 50.69 [50.02, 51.35] |
| es | 50.55 [47.14, 53.90] | 51.01 [47.79, 54.81] | 50.41 [43.91, 58.30] | 50.24 [49.37, 51.42] | 50.67 [49.78, 51.66] | 50.59 [50.19, 51.09] |
| de | 45.47 [39.05, 53.96] | 52.49 [45.95, 59.18] | 50.55 [47.25, 54.71] | 34.61 [26.19, 47.39] | 35.05 [26.93, 47.29] | 33.60 [24.87, 46.82] |
| zh | 46.35 [39.32, 55.99] | 49.55 [45.60, 53.94] | 47.34 [40.03, 55.91] | 51.36 [48.16, 54.66] | 48.47 [40.84, 57.52] | 47.37 [39.41, 56.87] |
| hi | 48.41 [45.58, 52.30] | 50.37 [49.63, 51.06] | 49.30 [41.24, 58.67] | 49.76 [48.60, 51.29] | 53.38 [50.59, 56.76] | 51.75 [48.87, 54.14] |
| mr | 44.10 [36.44, 54.61] | 42.72 [35.12, 51.15] | 45.47 [40.84, 50.69] | 41.74 [31.52, 53.69] | 47.29 [43.56, 52.47] | 53.20 [49.87, 56.79] |

#### Surface control (2026-09-30)

**FACT (2026-09-30):** Source: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`families.surface.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 45.98 [37.89, 55.25] | 44.28 [35.40, 54.47] | 46.94 [43.93, 50.49] | 39.69 [33.42, 47.62] | 52.65 [50.67, 54.69] | 47.80 [44.43, 51.44] |
| es | 45.32 [37.84, 54.04] | 45.37 [37.19, 55.21] | 49.22 [45.56, 53.45] | 49.53 [48.03, 50.69] | 49.73 [46.70, 53.56] | 48.18 [45.07, 52.14] |
| de | 49.36 [42.31, 56.70] | 49.46 [42.62, 56.89] | 45.81 [36.70, 57.54] | 42.47 [33.59, 53.79] | 52.07 [50.50, 53.90] | 44.31 [35.48, 55.10] |
| zh | 44.85 [39.98, 49.83] | 45.92 [42.62, 49.97] | 45.00 [39.56, 51.79] | 44.97 [37.57, 54.71] | 50.42 [48.10, 52.81] | 43.20 [37.59, 51.12] |
| hi | 53.24 [51.06, 55.40] | 52.97 [50.68, 55.20] | 53.05 [49.97, 56.07] | 42.28 [33.29, 53.19] | 49.23 [39.81, 60.52] | 53.17 [50.84, 55.35] |
| mr | 46.47 [38.35, 57.12] | 47.92 [39.63, 58.09] | 44.90 [35.19, 57.26] | 49.56 [48.91, 50.44] | 52.12 [50.84, 53.76] | 44.85 [35.17, 56.95] |

### 16.3 Held-out cell means (2026-09-30)

**FACT (2026-09-30):** Each mean gives equal weight to its cells, after pooled class recall is computed within each cell. All-cell means average 36 cells; same-language means average six diagonal cells; cross-language means average 30 off-diagonal cells. Values are balanced accuracy (%) [pointwise 95% interval]. Source: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`families.*.holdout.all_cell_mean`, `diagonal_mean`, `off_diagonal_mean`).

| Control | All 36 cells | Same language: 6 cells | Cross-language: 30 cells |
|---|---|---|---|
| Character control | 49.17 [47.09, 51.92] | 51.59 [49.02, 54.29] | 48.68 [46.55, 51.57] |
| Word control | 47.90 [44.69, 52.28] | 51.55 [49.02, 54.59] | 47.17 [43.67, 52.03] |
| Surface control | 47.56 [43.48, 52.74] | 46.04 [37.57, 56.47] | 47.87 [44.63, 52.05] |

### 16.4 Holdout minus development on the same documents (2026-09-30)

**FACT / DESCRIPTIVE (2026-09-30):** Development predictions are restricted to the same 58 manifestos: 4,901 items, with 2,402 label-0 and 2,499 label-1 items. The table subtracts these development cell means from held-out cell means, in percentage points [pointwise 95% interval], using the same manifesto weights for both cohorts. Source: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`method.matched_development_items`, `matched_development_class_counts`, `families.*.holdout_minus_matched_development_pp`).

| Control | Change over all 36 cells | Change over 6 same-language cells | Change over 30 cross-language cells |
|---|---|---|---|
| Character control | -5.27 [-7.26, -2.61] | -15.79 [-18.51, -13.10] | -3.16 [-5.23, -0.37] |
| Word control | -4.67 [-7.55, -0.87] | -14.14 [-16.65, -11.26] | -2.77 [-5.96, 1.35] |
| Surface control | -1.09 [-4.77, 3.69] | -1.35 [-8.91, 8.41] | -1.04 [-3.99, 2.80] |

**LIMITATION (2026-09-30):** These cohorts contain different statements and categories from the same documents. Their differences describe this category shift; they do not estimate a paired same-statement or causal effect. The intervals condition on fixed classifiers and folds, exclude retraining uncertainty, and are not simultaneous or adjusted for multiple comparisons. The results do not establish sentence-level ideology, party independence, or generalization across a population of categories. Sources: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`method.difference`, `interval`, `cluster`); frozen limitations in §16 above.

### 16.5 Verification, provenance, and open GPU blocker (2026-09-30)

**FACT (2026-09-30):** All 90 fits matched the frozen coefficient, intercept, vocabulary/IDF, and scaler hash sets exactly. The three prediction-file hashes matched the CPU summary. The analysis checked all 108 held-out cells and all 108 full-66-manifesto development cells against their summaries to tolerance 1e-12; the matched-58-manifesto development cells were recomputed from primary predictions. Item labels, fold assignments, and cell alignment were checked before resampling. Sources: [CPU run summary](../../../gcp-workspace/rile_v2_category_holdout/full-20260930-luna/controls/summary.json) (`fits`, `prediction_files`); [frozen development control summary](../../../gcp-workspace/rile_v2_text_controls/full-20260930-luna/summary.json) (`fits`); [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`method.checks`, `provenance`); [controls-only reproduction script](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/run_controls_bootstrap.py).

**FACT (2026-09-30):** The partial analysis preserves input/output hashes, Python and dependency versions, helper hashes, and bootstrap weight hash. The reproducible controls-only script is retained beside the artifacts. Sources: [partial bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_bootstrap.json) (`runtime`, `provenance`, `method.weights_sha256`); [controls-only reproduction script](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/run_controls_bootstrap.py). [Partial HTML tables](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-controls-only-20260930-luna/PARTIAL_controls_only_tables.html) display the same three matrices.

**OPEN / INCOMPLETE (2026-09-30):** Automatic approval review initially rejected the proposed SCP export of nonpublic code, translated statements, predictions, and fitted probes because the existing task authorization did not explicitly authorize that export. The owner subsequently explicitly authorized the private transfer and evaluation after the push to main. The transfer and four model evaluations remain unlaunched at this snapshot; no workaround was performed. Source: automatic approval review response, subsequent owner authorization, and current work-session execution record (2026-09-30); [open issue](../../../docs/bugs-squashed.md#2026-09-30---category-holdout-gpu-transfer-approval-open).

### 16.6 OLMo smoke failure and recorded batch-size correction (2026-09-30)

**FACT / INCOMPLETE (2026-09-30):** The authorized transfer subsequently completed: all 23 transferred-file hashes matched the prepared local manifest. The transfer-approval blocker recorded in §16.5 is resolved. One OLMo GPU smoke then ran and stopped with `needs_review` after 101.080 seconds at the development-prediction parity gate. Its metadata-selected holdout smoke cohort contained 20 items; no accepted held-out model result was saved. Full OLMo and the later model runs were not launched. Sources: [local transfer manifest](../../../gcp-workspace/rile_v2_category_holdout/bundle-20260930-luna/local_relative_sha256.txt); [returned transfer manifest](../../../gcp-workspace/rile_v2_category_holdout/bundle-20260930-luna/remote_sha256.txt); [failed smoke summary](../../../gcp-workspace/rile_v2_category_holdout/diagnostic-20260930-luna/summary.json) (`status`, `cohort_items`, `error`, `elapsed_seconds`); current work-session execution record (2026-09-30).

**FACT (2026-09-30):** The first reported English-to-English development parity failure had absolute decision-margin error 0.0211628636, exceeding the frozen tolerance. Its saved reference margin was 6.4036789168, from fold 0 at layer 32. The failed runner used extraction batch size 4; every original selection summary records 8. The source-replay validator requires its extraction batch size to match that original recorded setting. Sources: [failed smoke summary](../../../gcp-workspace/rile_v2_category_holdout/diagnostic-20260930-luna/summary.json) (`error`, recorded runner hash); [saved OLMo predictions](../../../gcp-workspace/rile-v2-source-matrix-replay-20260929-retrieval/olmo/rile_v2_source_matrix_replay_allenai_Olmo-3-7B-Instruct_predictions.jsonl.gz); [source replay helper](../src/multilingual_layerwise_probe.py) (`replay_source_layers`, `run_rile_source_matrix`); original [OLMo selection summary](../../../gcp-workspace/rile-v2-budgeted-full-20260929-retrieval/rile_v2_budgeted_source_peak_allenai_Olmo-3-7B-Instruct_summary.json), [Qwen selection summary](../../../gcp-workspace/rile-v2-budgeted-full-20260929-retrieval/rile_v2_budgeted_source_peak_Qwen_Qwen3.5-9B_summary.json), [Ministral selection summary](../../../gcp-workspace/rile-v2-budgeted-full-20260929-retrieval/rile_v2_budgeted_source_peak_mistralai_Ministral-8B-Instruct-2410_summary.json), and [Gemma selection summary](../../../gcp-workspace/rile-v2-budgeted-full-20260929-retrieval/rile_v2_budgeted_source_peak_google_gemma-2-9b-it_summary.json) (`batch_size`).

**FACT / REPAIR RECORDED (2026-09-30):** The evaluator now resolves each original selection summary from the replay's recorded basename, verifies its SHA-256 against the frozen replay summary, validates model/dataset/protocol/development count, and requires the recorded positive integer batch size to be 8. It uses that setting for extraction and records the selection path/hash and batch size in output provenance, including failed-run summaries. The first eight original-order development rows now occupy one batch of eight. Source: [evaluation runner](../src/evaluate_rile_category_holdout.py) (`selection_config`, `model_run`, `main`).

**HYPOTHESIS / OPEN (2026-09-30):** The batch-size mismatch may explain the margin failure; that explanation has not been confirmed by a GPU parity rerun. The repair changes no selected layer, fitted probe, fold, label, threshold, feature precision, or tolerance (`atol=0.001`, `rtol=0.0001`, exact predictions). Local checks verified all four original selection-summary hashes and batch settings, rejected model/count/hash mismatches, parsed the repaired script, and confirmed CPU imports leave Torch unloaded. GPU verification remains pending; a repaired smoke must pass before any full model evaluation is accepted. Sources: [evaluation runner](../src/evaluate_rile_category_holdout.py); original selection summaries cited above; current work-session local-check record (2026-09-30); [open parity issue](../../../docs/bugs-squashed.md#2026-09-30---category-holdout-development-parity-and-batch-size-open).

### 16.7 Completed seven-family category holdout (2026-09-30)

**FACT / COMPLETED (2026-09-30):** The four corrected model smokes and full evaluations completed, and their predictions were combined with the three completed CPU controls. This complete result supersedes the campaign-status snapshots in §§16.1, 16.5, and 16.6; those partial results and failed-run records remain historical evidence. Each of the seven families has 36 source–target cells, 1,087 items per cell, and 39,132 predictions. The cohort contains 593 label-0 and 494 label-1 items from 58 manifestos and the four fixed excluded categories 107, 506, 601, and 603. Sources: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method`, `families`, `provenance`); [OLMo full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/OLMo/summary.json); [Qwen3.5 full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Qwen3.5/summary.json); [Ministral full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Ministral/summary.json); [Gemma full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Gemma/summary.json); [corrected driver status](../../../gcp-workspace/rile_v2_category_holdout/diagnostic-batch8-20261001-luna/driver_status_batch8_v4.tsv).

**FACT / FROZEN METHOD (2026-09-30):** Every item uses only its manifesto-excluding fold classifier, with the saved source-language-selected layer and probe parameters unchanged. The five probes are not ensembled; no holdout refitting, layer selection, or tuning was performed. Character/word/surface controls reuse the exact reproduced frozen development fits. Sources: [evaluation runner](../src/evaluate_rile_category_holdout.py) (`model_run`, `control_run`); [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method.document_exclusion`, `conditional_on`); [CPU control summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/controls/summary.json) (`fits`).

**FACT (2026-09-30):** Balanced accuracy is the mean of the two pooled class recalls within a cell. Tables report percent [pointwise 95% percentile interval], rounded to two decimals. The analysis used 10,000 draws of 58 manifestos with replacement, seed 20260930, with shared weights across all families, translations, and both cohorts; zero draws lacked a class. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method`, `families.*.holdout.matrix`).

#### OLMo: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.OLMo.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 56.08 [53.17, 59.12] | 50.36 [46.19, 53.95] | 51.17 [49.42, 52.96] | 46.53 [41.27, 51.27] | 48.46 [41.95, 54.59] | 48.88 [42.43, 54.60] |
| es | 49.94 [45.71, 55.25] | 55.14 [52.37, 57.99] | 49.49 [45.52, 53.67] | 43.41 [39.03, 49.00] | 46.40 [38.59, 57.73] | 51.91 [44.22, 61.33] |
| de | 58.31 [52.45, 62.76] | 50.29 [46.81, 54.97] | 54.18 [50.10, 57.86] | 53.31 [47.74, 57.61] | 46.55 [41.24, 54.12] | 48.49 [44.25, 52.85] |
| zh | 52.71 [49.48, 55.86] | 51.60 [44.20, 57.42] | 49.25 [43.75, 53.26] | 56.84 [54.33, 59.39] | 55.02 [42.60, 64.50] | 54.56 [44.21, 62.21] |
| hi | 48.46 [42.68, 55.47] | 47.92 [41.05, 56.71] | 43.62 [35.33, 55.39] | 45.88 [41.10, 52.50] | 52.84 [48.64, 56.06] | 47.21 [40.05, 57.09] |
| mr | 49.69 [45.51, 54.50] | 50.67 [43.41, 56.62] | 45.90 [40.81, 51.68] | 50.11 [46.04, 53.55] | 51.73 [46.32, 56.81] | 51.07 [48.57, 53.83] |

#### Qwen3.5: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.Qwen3.5.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 61.97 [58.77, 65.45] | 59.25 [53.37, 64.29] | 50.30 [46.58, 53.86] | 56.00 [53.85, 58.13] | 58.42 [50.79, 64.81] | 50.23 [46.46, 53.53] |
| es | 58.36 [54.55, 62.23] | 57.50 [53.35, 61.09] | 56.71 [52.32, 60.67] | 53.25 [46.91, 60.91] | 51.11 [46.11, 56.08] | 51.52 [49.86, 53.46] |
| de | 50.70 [45.31, 58.11] | 42.17 [34.84, 52.43] | 56.56 [53.61, 59.52] | 60.70 [54.72, 67.07] | 55.53 [48.75, 63.59] | 50.64 [48.51, 53.15] |
| zh | 58.17 [51.64, 64.29] | 47.29 [36.60, 59.26] | 51.43 [48.79, 53.98] | 58.76 [54.66, 62.09] | 55.63 [49.68, 60.42] | 50.84 [45.07, 55.75] |
| hi | 53.99 [46.26, 62.89] | 51.00 [40.99, 63.79] | 48.81 [40.41, 58.66] | 58.62 [51.79, 65.92] | 49.12 [46.28, 51.80] | 47.41 [41.25, 55.81] |
| mr | 51.58 [46.38, 56.54] | 59.74 [53.12, 65.27] | 46.48 [42.05, 51.61] | 55.75 [49.85, 61.28] | 56.30 [51.78, 59.95] | 51.39 [48.68, 54.63] |

#### Ministral: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.Ministral.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 57.52 [54.54, 60.45] | 60.07 [55.18, 65.31] | 56.45 [48.77, 62.79] | 56.60 [49.67, 65.80] | 47.44 [39.62, 55.87] | 56.96 [45.45, 65.73] |
| es | 57.09 [52.91, 60.89] | 57.33 [53.72, 60.47] | 51.27 [45.96, 58.15] | 49.95 [47.06, 52.51] | 45.13 [41.23, 49.30] | 46.01 [39.99, 54.36] |
| de | 54.26 [51.29, 57.06] | 52.78 [50.24, 55.36] | 54.90 [51.82, 57.35] | 57.19 [52.09, 61.24] | 47.65 [42.79, 53.85] | 53.22 [49.53, 56.45] |
| zh | 52.22 [50.54, 54.33] | 50.89 [48.83, 53.45] | 50.81 [49.23, 52.72] | 59.95 [55.52, 63.71] | 58.83 [50.71, 64.84] | 52.12 [50.44, 53.83] |
| hi | 49.99 [46.13, 54.22] | 49.30 [43.57, 53.95] | 51.04 [45.17, 55.52] | 57.97 [53.66, 63.86] | 56.96 [53.82, 60.72] | 51.26 [45.83, 55.83] |
| mr | 54.45 [48.67, 58.74] | 55.48 [50.32, 60.05] | 56.08 [50.24, 60.53] | 53.23 [47.27, 59.01] | 48.93 [44.70, 53.30] | 53.86 [50.47, 56.76] |

#### Gemma: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.Gemma.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 62.47 [59.52, 65.16] | 54.78 [50.62, 59.46] | 59.27 [54.78, 62.66] | 57.67 [54.40, 62.45] | 53.64 [47.23, 59.51] | 55.47 [49.16, 61.40] |
| es | 59.03 [55.53, 62.26] | 60.65 [57.31, 63.95] | 51.53 [47.31, 56.62] | 61.45 [55.79, 66.16] | 58.19 [53.70, 61.70] | 57.15 [52.42, 61.12] |
| de | 55.48 [53.19, 58.35] | 56.10 [50.57, 62.03] | 59.12 [55.75, 61.96] | 60.04 [56.72, 63.31] | 59.17 [56.19, 61.89] | 57.74 [54.07, 61.76] |
| zh | 63.03 [59.39, 66.61] | 59.15 [53.53, 64.65] | 59.75 [54.15, 64.73] | 60.50 [57.40, 63.52] | 60.59 [55.08, 64.58] | 55.03 [51.55, 59.47] |
| hi | 60.49 [55.06, 64.66] | 59.86 [51.48, 66.17] | 64.18 [56.35, 69.91] | 59.10 [54.23, 62.76] | 58.54 [55.08, 62.04] | 53.32 [46.68, 59.08] |
| mr | 55.65 [51.99, 60.27] | 55.51 [51.37, 59.93] | 56.69 [49.86, 61.80] | 51.29 [47.10, 56.51] | 51.65 [47.62, 57.07] | 56.08 [52.85, 59.30] |

#### Character control: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.char.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 51.41 [48.45, 54.88] | 49.77 [45.84, 54.18] | 50.93 [47.71, 53.45] | 50.57 [49.52, 51.66] | 50.69 [49.18, 51.96] | 50.44 [48.34, 52.02] |
| es | 50.91 [47.73, 53.72] | 54.90 [52.21, 58.23] | 46.99 [42.87, 52.01] | 50.56 [49.93, 51.19] | 50.91 [49.91, 52.01] | 50.59 [49.52, 51.48] |
| de | 50.77 [46.97, 53.85] | 49.81 [45.38, 54.82] | 49.37 [46.24, 52.70] | 50.71 [49.85, 51.77] | 50.84 [49.66, 52.17] | 50.96 [49.83, 52.04] |
| zh | 42.41 [33.92, 54.32] | 45.90 [37.54, 57.48] | 41.87 [35.14, 50.28] | 51.39 [46.13, 56.70] | 38.64 [28.49, 53.54] | 36.45 [24.87, 53.09] |
| hi | 52.07 [47.51, 56.99] | 50.13 [49.83, 50.42] | 49.43 [48.01, 51.10] | 50.07 [49.79, 50.38] | 51.82 [48.06, 54.79] | 54.94 [52.64, 57.14] |
| mr | 49.07 [47.82, 50.47] | 49.46 [48.77, 50.29] | 48.92 [47.37, 50.38] | 49.48 [48.82, 50.41] | 46.23 [42.27, 51.62] | 50.62 [47.06, 53.99] |

#### Word control: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.word.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 49.83 [46.53, 54.07] | 50.19 [43.21, 56.67] | 49.75 [44.11, 56.97] | 50.07 [48.80, 51.82] | 50.64 [49.46, 51.97] | 50.69 [50.02, 51.35] |
| es | 50.55 [47.14, 53.90] | 51.01 [47.79, 54.81] | 50.41 [43.91, 58.30] | 50.24 [49.37, 51.42] | 50.67 [49.78, 51.66] | 50.59 [50.19, 51.09] |
| de | 45.47 [39.05, 53.96] | 52.49 [45.95, 59.18] | 50.55 [47.25, 54.71] | 34.61 [26.19, 47.39] | 35.05 [26.93, 47.29] | 33.60 [24.87, 46.82] |
| zh | 46.35 [39.32, 55.99] | 49.55 [45.60, 53.94] | 47.34 [40.03, 55.91] | 51.36 [48.16, 54.66] | 48.47 [40.84, 57.52] | 47.37 [39.41, 56.87] |
| hi | 48.41 [45.58, 52.30] | 50.37 [49.63, 51.06] | 49.30 [41.24, 58.67] | 49.76 [48.60, 51.29] | 53.38 [50.59, 56.76] | 51.75 [48.87, 54.14] |
| mr | 44.10 [36.44, 54.61] | 42.72 [35.12, 51.15] | 45.47 [40.84, 50.69] | 41.74 [31.52, 53.69] | 47.29 [43.56, 52.47] | 53.20 [49.87, 56.79] |

#### Surface control: held-out source–target matrix (2026-09-30)

**FACT (2026-09-30):** Rows are training source languages; columns are evaluation target languages. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families.surface.holdout.matrix`).

| Source → target | en | es | de | zh | hi | mr |
|---|---|---|---|---|---|---|
| en | 45.98 [37.89, 55.25] | 44.28 [35.40, 54.47] | 46.94 [43.93, 50.49] | 39.69 [33.42, 47.62] | 52.65 [50.67, 54.69] | 47.80 [44.43, 51.44] |
| es | 45.32 [37.84, 54.04] | 45.37 [37.19, 55.21] | 49.22 [45.56, 53.45] | 49.53 [48.03, 50.69] | 49.73 [46.70, 53.56] | 48.18 [45.07, 52.14] |
| de | 49.36 [42.31, 56.70] | 49.46 [42.62, 56.89] | 45.81 [36.70, 57.54] | 42.47 [33.59, 53.79] | 52.07 [50.50, 53.90] | 44.31 [35.48, 55.10] |
| zh | 44.85 [39.98, 49.83] | 45.92 [42.62, 49.97] | 45.00 [39.56, 51.79] | 44.97 [37.57, 54.71] | 50.42 [48.10, 52.81] | 43.20 [37.59, 51.12] |
| hi | 53.24 [51.06, 55.40] | 52.97 [50.68, 55.20] | 53.05 [49.97, 56.07] | 42.28 [33.29, 53.19] | 49.23 [39.81, 60.52] | 53.17 [50.84, 55.35] |
| mr | 46.47 [38.35, 57.12] | 47.92 [39.63, 58.09] | 44.90 [35.19, 57.26] | 49.56 [48.91, 50.44] | 52.12 [50.84, 53.76] | 44.85 [35.17, 56.95] |

#### Held-out cell means (2026-09-30)

**FACT (2026-09-30):** Values are balanced accuracy (%) [pointwise 95% interval]. Means equally weight 36 cells, six same-language cells, or 30 cross-language cells. The matched development cohort contains 4,901 different statements from the same 58 documents, with label counts 2,402 / 2,499. Its predictions are recomputed from primary rows; the original 66-document development matrices retained in the JSON are context only and are not this cohort. The subtraction uses paired manifesto weights across the two cohorts. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method.matched_development_items`, `matched_development_class_counts`, `families.*.holdout`, `full_66_manifesto_development_context_only`).

| Family | All 36 cells | Same-language: 6 cells | Cross-language: 30 cells |
|---|---|---|---|
| OLMo | 50.39 [49.04, 51.98] | 54.36 [52.61, 55.94] | 49.59 [48.21, 51.34] |
| Qwen3.5 | 53.70 [52.39, 55.35] | 55.88 [53.70, 57.82] | 53.26 [51.91, 55.07] |
| Ministral | 53.48 [52.06, 54.68] | 56.75 [54.24, 58.98] | 52.82 [51.50, 53.98] |
| Gemma | 57.76 [55.94, 59.32] | 59.56 [57.42, 61.51] | 57.40 [55.58, 58.93] |
| Character control | 49.17 [47.09, 51.92] | 51.59 [49.02, 54.29] | 48.68 [46.55, 51.57] |
| Word control | 47.90 [44.69, 52.28] | 51.55 [49.02, 54.59] | 47.17 [43.67, 52.03] |
| Surface control | 47.56 [43.48, 52.74] | 46.04 [37.57, 56.47] | 47.87 [44.63, 52.05] |

#### Development cell means on the same 58 documents (2026-09-30)

**FACT (2026-09-30):** Values are balanced accuracy (%) [pointwise 95% interval]. Means equally weight 36 cells, six same-language cells, or 30 cross-language cells. The matched development cohort contains 4,901 different statements from the same 58 documents, with label counts 2,402 / 2,499. Its predictions are recomputed from primary rows; the original 66-document development matrices retained in the JSON are context only and are not this cohort. The subtraction uses paired manifesto weights across the two cohorts. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method.matched_development_items`, `matched_development_class_counts`, `families.*.matched_development`, `full_66_manifesto_development_context_only`).

| Family | All 36 cells | Same-language: 6 cells | Cross-language: 30 cells |
|---|---|---|---|
| OLMo | 53.96 [53.32, 54.64] | 58.93 [57.90, 60.05] | 52.97 [52.36, 53.59] |
| Qwen3.5 | 58.64 [57.71, 59.56] | 65.99 [64.93, 67.09] | 57.17 [56.19, 58.17] |
| Ministral | 59.47 [58.70, 60.21] | 69.81 [68.41, 71.18] | 57.40 [56.69, 58.10] |
| Gemma | 65.02 [64.02, 66.02] | 69.31 [68.26, 70.35] | 64.16 [63.13, 65.18] |
| Character control | 54.43 [53.46, 55.42] | 67.37 [65.81, 68.90] | 51.85 [50.87, 52.86] |
| Word control | 52.57 [50.61, 54.60] | 65.70 [64.12, 67.36] | 49.94 [47.75, 52.29] |
| Surface control | 48.66 [46.91, 50.71] | 47.38 [44.52, 50.74] | 48.91 [47.37, 50.73] |

#### Held-out minus matched-development cell means (2026-09-30)

**FACT (2026-09-30):** Values are percentage points [pointwise 95% interval]. Means equally weight 36 cells, six same-language cells, or 30 cross-language cells. The matched development cohort contains 4,901 different statements from the same 58 documents, with label counts 2,402 / 2,499. Its predictions are recomputed from primary rows; the original 66-document development matrices retained in the JSON are context only and are not this cohort. The subtraction uses paired manifesto weights across the two cohorts. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method.matched_development_items`, `matched_development_class_counts`, `families.*.holdout_minus_matched_development_pp`, `full_66_manifesto_development_context_only`).

| Family | All 36 cells | Same-language: 6 cells | Cross-language: 30 cells |
|---|---|---|---|
| OLMo | -3.57 [-5.02, -2.07] | -4.57 [-6.61, -2.88] | -3.37 [-4.82, -1.73] |
| Qwen3.5 | -4.94 [-6.18, -3.44] | -10.11 [-12.48, -7.90] | -3.91 [-5.09, -2.32] |
| Ministral | -5.99 [-7.69, -4.50] | -13.06 [-16.02, -10.42] | -4.58 [-6.14, -3.23] |
| Gemma | -7.26 [-9.12, -5.59] | -9.75 [-11.86, -7.84] | -6.76 [-8.63, -5.08] |
| Character control | -5.27 [-7.26, -2.61] | -15.79 [-18.51, -13.10] | -3.16 [-5.23, -0.37] |
| Word control | -4.67 [-7.55, -0.87] | -14.14 [-16.65, -11.26] | -2.77 [-5.96, 1.35] |
| Surface control | -1.09 [-4.77, 3.69] | -1.35 [-8.91, 8.41] | -1.04 [-3.99, 2.80] |

#### Paired probe-minus-control cross-language differences (2026-09-30)

**FACT (2026-09-30):** Each difference subtracts the control mean from the probe mean over the 30 held-out cross-language cells, within each shared manifesto draw. Values are percentage points [pointwise 95% interval]. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`probe_minus_control.*.off_diagonal_mean`).

| Probe minus control | Cross-language mean difference |
|---|---|
| OLMo minus char | 0.91 [-1.20, 2.58] |
| OLMo minus word | 2.43 [-1.17, 5.34] |
| OLMo minus surface | 1.72 [-1.56, 4.63] |
| Qwen3.5 minus char | 4.58 [2.43, 6.21] |
| Qwen3.5 minus word | 6.10 [1.50, 9.55] |
| Qwen3.5 minus surface | 5.40 [1.90, 8.04] |
| Ministral minus char | 4.14 [0.59, 6.85] |
| Ministral minus word | 5.65 [0.62, 9.50] |
| Ministral minus surface | 4.95 [0.26, 8.72] |
| Gemma minus char | 8.72 [4.79, 11.76] |
| Gemma minus word | 10.23 [4.55, 14.57] |
| Gemma minus surface | 9.53 [4.50, 13.61] |

**LIMITATION (2026-09-30):** These tables report the category-derived-label measurements and their descriptive differences. The matched-document cohorts contain different statements/categories; the differences do not estimate a paired same-statement or causal effect. Intervals condition on saved layers, classifiers, and splits, exclude retraining uncertainty, and are pointwise rather than simultaneous or multiplicity-adjusted. No sentence-ideology, party-independence, model-ranking, or wider category-population claim is made from these four fixed categories. Source: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method.interval`, `conditional_on`, `difference`, `limitations`); frozen protocol in §16.

#### Exact models, runtime, and verification (2026-09-30)

**FACT (2026-09-30):** All four model runs used their exact pinned revisions, recorded extraction batch size 8, raw final-token hidden states stored as float16 and cast to float32 for scoring, and the existing tokenizer settings. This Qwen model is Qwen3.5. Each smoke and full run checked eight original-order development items across 36 language pairs (288 checks): guesses matched exactly and maximum absolute margin error was 0.0; the unchanged tolerances were `atol=0.001`, `rtol=0.0001`. Sources: [OLMo full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/OLMo/summary.json); [Qwen3.5 full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Qwen3.5/summary.json); [Ministral full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Ministral/summary.json); [Gemma full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Gemma/summary.json); [OLMo smoke summary](../../../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/OLMo/summary.json); [Qwen3.5 smoke summary](../../../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/Qwen3.5/summary.json); [Ministral smoke summary](../../../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/Ministral/summary.json); [Gemma smoke summary](../../../gcp-workspace/rile_v2_category_holdout/smoke-batch8-20260930-luna/Gemma/summary.json) (`model`, `model_revision`, `batch_size`, `feature`, `dev_parity`).

| Family | Exact model | Revision | Smoke seconds | Full seconds |
|---|---|---|---|---|
| OLMo | `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | 108.48 | 412.14 |
| Qwen3.5 | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 131.89 | 469.57 |
| Ministral | `mistralai/Ministral-8B-Instruct-2410` | `2f494a194c5b980dfb9772cb92d26cbb671fce5a` | 116.49 | 313.55 |
| Gemma | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | 132.82 | 344.51 |

**FACT (2026-09-30):** The corrected driver interval was 34 minutes 19 seconds: 2026-10-01 00:09:48–00:44:07 UTC, corresponding to 2026-09-30 17:09:48–17:44:07 Pacific daylight time. It covers the corrected smoke/full sequence and wrapper overhead. Model runtimes above come from individual summaries. The local seven-family bootstrap took 6.81 seconds. Sources: [corrected driver status](../../../gcp-workspace/rile_v2_category_holdout/diagnostic-batch8-20261001-luna/driver_status_batch8_v4.tsv); [OLMo full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/OLMo/summary.json); [Qwen3.5 full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Qwen3.5/summary.json); [Ministral full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Ministral/summary.json); [Gemma full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Gemma/summary.json); [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`elapsed_seconds`).

**LIMITATION (2026-09-30):** Driver duration is not total VM running time or a cost measurement. Total VM uptime and incurred cost for this sequence are not established by the evaluation artifacts. Source: driver-status scope and the run-summary timing fields cited above.

**FACT (2026-09-30):** The model environment recorded Python 3.10.12, NumPy 2.2.6, PyTorch 2.13.0+cu130, Transformers 5.14.1, and bfloat16 execution. The CPU bootstrap recorded Python 3.12.7 and NumPy 2.5.3. Saved summary/prediction hashes, frozen parameter hashes, and original selection-summary hashes are retained in the model summaries and bootstrap provenance. Sources: [OLMo full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/OLMo/summary.json); [Qwen3.5 full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Qwen3.5/summary.json); [Ministral full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Ministral/summary.json); [Gemma full summary](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/Gemma/summary.json); [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`runtime`, `provenance`).

**FACT (2026-09-30):** The analysis checked all 252 held-out cell point estimates and all 252 original full-development cell point estimates against summaries to tolerance 1e-12; it separately recomputed development on the matched 58 documents. It validated item/document/fold/label alignment, saved selected layers, and prediction hashes before resampling. The following source/helper and shared-weight hashes are recorded. Sources: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`method.point_checks`, `weights_sha256`, `code_sha256`, `helper_sha256`, `provenance`); [analysis runner](../src/bootstrap_rile_category_holdout.py).

| Provenance field | SHA-256 |
|---|---|
| Evaluator | `bb366b23cdf72002e3eae8eb7d7006536016ec4bf9dbd72a823651979451f9d0` |
| Feature-extraction helper | `c125bdc1ee6fee74dcc1858ccf9f0cd79843058f1b50e59bc774a103530beaa8` |
| Bootstrap runner | `f975c97b062e43cfd392526ccb01e7db55e86c833233dc5303b101c20b642119` |
| Bootstrap transfer helper | `6ed5068458cf0689ed0ee69b366d3762d8ca5169a9957c2f11fd41470d5e759b` |
| Shared bootstrap weights | `79c7a1c5c773416669b04732012c80baf59c80e055ebfb843a272a6a94e24341` |

**FACT (2026-09-30):** All seven full matrices, source/target means, cellwise cohort differences, and probe-minus-control differences are retained in the numerical artifact. Sources: [complete bootstrap JSON](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json) (`families`, `probe_minus_control`); [complete HTML matrices](../../../gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/tables.html). Earlier failed diagnostics were preserved; recorded issue closures are in [bugs-squashed](../../../docs/bugs-squashed.md).

### 16.8 Category errors and paired model disagreements (2026-09-30)

**FACT (2026-09-30):** This CPU-only analysis uses the frozen four-model predictions for 1,087 items in 36 source–target cells. It rechecked all 144 balanced-accuracy points against both model summaries and the seven-family bootstrap JSON within 1e-12, verified prediction/input hashes, item/manifesto/fold/label/category coverage, and each saved selected layer. The 10,000 shared 58-manifesto bootstrap weights (seed 20260930) matched the upstream SHA-256 `79c7a1c5c773416669b04732012c80baf59c80e055ebfb843a272a6a94e24341`; zero draws were discarded. All four categories had 10,000 valid draws. A three-draw fixture with one absent-category draw verified that the percentile interval uses only valid draws. The repaired CPU run took 2.02 seconds. Sources: [repaired error-analysis JSON](../../../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/error_analysis.json) (`method`, `provenance`, `categories`); [analysis runner](../src/analyze_rile_holdout_errors.py); [frozen model summaries](../../../gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna/).

**FACT / METRIC DEFINITION (2026-09-30):** Each category has only one category-derived label, so the category metric is the pooled fraction correct relative to that label, not balanced accuracy. The table's cross-language estimate averages the 30 off-diagonal source–target cell fractions; the denominator is 30 × the category item count. The error-share column is that category's wrong cross-language item-cell predictions divided by the model's total wrong predictions across all 1,087 items × 30 cells. Items recur across language cells, so these counts are repeated predictions, not independent item observations. Brackets give pointwise 95% percentile intervals using the shared manifesto draws.

| Category | Items (share of holdout) | Model | Cross-language correct % [95% interval] | Wrong predictions / category cross predictions | Share of model cross errors |
|---|---:|---|---:|---:|---:|
| 107 | 287 (26.4%) | OLMo | 48.25 [46.50, 49.81] | 4,456 / 8,610 | 27.1% |
| 107 | 287 (26.4%) | Qwen3.5 | 37.24 [34.35, 40.15] | 5,404 / 8,610 | 34.4% |
| 107 | 287 (26.4%) | Ministral | 48.11 [46.22, 49.77] | 4,468 / 8,610 | 28.7% |
| 107 | 287 (26.4%) | Gemma | 50.65 [48.04, 53.75] | 4,249 / 8,610 | 30.2% |
| 506 | 306 (28.2%) | OLMo | 48.93 [47.37, 50.37] | 4,688 / 9,180 | 28.5% |
| 506 | 306 (28.2%) | Qwen3.5 | 38.25 [36.23, 40.39] | 5,669 / 9,180 | 36.1% |
| 506 | 306 (28.2%) | Ministral | 43.78 [41.53, 46.14] | 5,161 / 9,180 | 33.1% |
| 506 | 306 (28.2%) | Gemma | 52.98 [50.69, 55.42] | 4,316 / 9,180 | 30.7% |
| 601 | 271 (24.9%) | OLMo | 51.06 [48.53, 53.96] | 3,979 / 8,130 | 24.2% |
| 601 | 271 (24.9%) | Qwen3.5 | 68.87 [65.82, 72.74] | 2,531 / 8,130 | 16.1% |
| 601 | 271 (24.9%) | Ministral | 59.16 [56.56, 60.69] | 3,320 / 8,130 | 21.3% |
| 601 | 271 (24.9%) | Gemma | 62.61 [57.47, 65.72] | 3,040 / 8,130 | 21.6% |
| 603 | 223 (20.5%) | OLMo | 50.01 [46.16, 55.06] | 3,344 / 6,690 | 20.3% |
| 603 | 223 (20.5%) | Qwen3.5 | 68.65 [64.99, 73.85] | 2,097 / 6,690 | 13.4% |
| 603 | 223 (20.5%) | Ministral | 60.51 [57.66, 62.84] | 2,642 / 6,690 | 16.9% |
| 603 | 223 (20.5%) | Gemma | 63.35 [60.21, 65.41] | 2,452 / 6,690 | 17.4% |

**FACT (2026-09-30):** Paired disagreement is the fraction of aligned item predictions on which two models output different labels, summarized across the 30 cross-language cells. Since the task is binary and both models use the same reference label, disagreement is also the fraction where exactly one of the pair is correct. The intervals use shared manifesto resampling and the weighted item count for each draw.

| Model pair | Cross-language disagreement % [95% interval] |
|---|---:|
| OLMo vs Qwen3.5 | 49.05 [47.31, 51.14] |
| OLMo vs Ministral | 46.31 [43.90, 48.68] |
| OLMo vs Gemma | 49.25 [48.38, 50.07] |
| Qwen3.5 vs Ministral | 44.59 [43.53, 45.73] |
| Qwen3.5 vs Gemma | 45.68 [44.63, 46.90] |
| Ministral vs Gemma | 47.63 [46.80, 48.53] |

**FACT (2026-09-30):** For the exhaustive four-model partition across category items × 36 cells, unanimous-correct / unanimous-wrong / three-to-one / two-to-two counts were: category 107, 682 / 1,169 / 5,102 / 3,379; 506, 663 / 1,114 / 5,597 / 3,642; 601, 1,800 / 349 / 4,719 / 2,888; and 603, 1,473 / 280 / 3,882 / 2,393. The repaired output adds four category-specific 6×6 partition matrices and their 36-cell, diagonal-6-cell, and cross-language-30-cell totals. For every cell, the four counts sum to that category's item count; diagonal and cross-language totals sum to 6× and 30× that count, respectively. The full category and disagreement matrices, aggregates, row-level text-free review candidates, and provenance are in the [repaired HTML tables](../../../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/tables.html), [JSON](../../../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/error_analysis.json), and [compressed candidate rows](../../../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/manual_review_candidates.jsonl.gz). The candidate file has 34,514 rows and contains IDs, categories, folds, language pairs, and correctness flags, with no statement text. The first-pass output remains preserved in the sibling `error-analysis-20260930-luna/` folder.

**FACT (2026-09-30):** Repaired script SHA-256 is `716f3b0da3b646b1e494f1525ad041b6b0b548864a428abb4d83fd12b0097cf1`; repaired JSON SHA-256 is `2160e133e4150961d9eb0b4741bc490fb8beada77a327192d1b68cbb545efa6a`.

**LIMITATION (2026-09-30):** These are descriptive, fixed-category results conditional on saved probes, selected layers, and folds. Intervals are pointwise, not multiplicity-adjusted; repeated language-cell predictions are not independent samples. They support no significance, causal, model-ranking, sentence-level ideology, party-independence, or category-population inference. Sources: [repaired error-analysis JSON](../../../gcp-workspace/rile_v2_category_holdout/error-analysis-20260930-luna-repaired/error_analysis.json) (`method.limitations`, `categories`, `pairwise_model_disagreement`, `four_model_correctness_partitions`, `four_model_correctness_partitions_by_category`).
