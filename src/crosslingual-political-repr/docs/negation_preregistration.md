# Negation Confound: Pre-Registered Predictions

Last updated: 2026-09-15

Written **2026-09-15, before any subset analysis was run.** Nothing below may be changed after
the subset results are seen. Any later change must be added as a dated amendment, with the
original text kept.

## Why this test exists

In English, 242 of the 287 statements that contain a negation word are labelled −1. A rule with no
model ("contains a negation word → −1, else +1") scores 777 / 1,160 ≈ **67%** on the ±1 statements.
A probe that only detects negation cannot beat this rule by much, so cross-lingual transfer near 67% is
fully explainable by negation.

Observed cross-lingual (off-diagonal) accuracy at the reference layers: Gemma 2 9B 84.7%, Qwen 3.5 9B
80.7%, OLMo 3 7B 70.6%, Ministral 8B 68.1% (`current_raw`, `docs/rq2_findings.md`).

## Fixed definitions

- **Data source:** per-item predictions in `artifacts/results/extraction_diagnostics_<model>.json.gz`,
  condition `current_raw`. Use the lane that reproduces the published `current_raw` baseline. Checking
  the full-set baseline is allowed, since it is already published. Qwen uses Layer 12 (`..._layer12.json.gz`).
- **Models / layers:** OLMo 17, Ministral 31, Gemma 23, Qwen 12.
- **Negation words (English, whole words, case-insensitive):** not, no, never, n't, nor, none, nobody,
  nothing, neither, without, cannot.
- **Pair rule:** a question is *negation-marked* if **either** of its ±1 English statements contains a
  negation word; otherwise it is *clean*. The same split is used for every language, because the
  translations are aligned. Neutral (0) statements are excluded.
- **Metric:** per-statement accuracy, pooled over the 30 off-diagonal source→target cells, reported per
  model for the full, negation-marked and clean subsets. Diagonal accuracy is also reported.
- **Uncertainty:** 95% bootstrap intervals, 5,000 resamples over question IDs.
- **Chance on the clean subset:** 50% (each pair contributes one +1 and one −1 statement).

## Predictions

- **P1:** Gemma and Qwen keep clean-subset off-diagonal accuracy clearly above chance: the lower
  bound of the 95% interval is above 60% for both.
- **P2:** OLMo and Ministral fall to ≤ 58% clean-subset off-diagonal accuracy (point estimate).
- **P3:** The drop from full to clean is larger for OLMo and Ministral than for Gemma and Qwen.

**Falsified if:** P1 fails for either Gemma or Qwen. P2 fails if either OLMo or Ministral stays above 58%.
P3 fails if the ordering does not hold. Every outcome is reported regardless of direction.

## Declared secondary checks (reported, but not used to judge P1–P3)

1. **Target-language negation:** repeat the split using negation words in each target language
   (es: no, nunca, ni, sin; de: nicht, kein*, nie, ohne; zh: 不, 没, 無/无, 非, 未; hi: नहीं, न, मत;
   mr: नाही, न, नको). This catches translations that add a negation absent from the English.
2. **Broad negative wording:** a second list (too little, too much, less, fail*, harm*, worse)
   used as an exploratory check. It is decided now, but not treated as a prediction.
3. **Retrain on clean pairs only:** fit probes on clean pairs, then evaluate cross-lingual transfer.
   This needs re-extracted activations (none are stored locally).

## Known loophole this does not close

Checking literal negation words does not rule out a broader "negative framing" feature. Secondary
check 2 is a first look at that. It is not a full control.

---

## Amendment 1 — 2026-09-15 (still before any subset analysis)

Added after a review of other possible confounds. No subset result had been computed when this was written.

1. **Lane:** `fair` lane, condition `current_raw` (C chosen by inner grouped cross-validation). This is the lane
   behind the published baseline off-diagonal means (Gemma 84.67, Qwen 80.68, OLMo 70.64, Ministral 68.13).
2. **Label encoding:** `label = int(polarity == 1)`. Within each question, the stored rows are −1 then +1
   (`src/run_extraction_diagnostics.py::load_data`).
3. **Metrics:** report accuracy, AUC (from `decision_score`), and **pairwise accuracy**. Pairwise accuracy is
   the share of pairs where the +1 statement gets a higher `decision_score` than the −1 statement; ties count
   as half. It is not affected by a probe's threshold shifting in a new language. **P1–P3 are judged on
   pairwise accuracy.** The thresholds (60% lower bound, 58% point estimate) are unchanged; chance is still 50%.
   Accuracy is reported alongside.
4. **P4 (sentiment):** score each English ±1 statement with `cardiffnlp/twitter-xlm-roberta-base-sentiment`
   (argmax over negative / neutral / positive). A pair is *sentiment-matched* if both sides get the same argmax label.
   Prediction: on pairs that are **clean and sentiment-matched**, Gemma and Qwen both have a pairwise off-diagonal
   95% lower bound above 58%. Falsified if either does not. The label × sentiment table is also reported.
5. **Why the labels may be skewed:** the ±1 direction was chosen by Gemini during conversion
   (`src/convert_to_statements.py`), with no rule for which side is +1. This is audited by hand on
   50 random pairs (Phase 5) and reported regardless of outcome.

---

## Amendment 2 — 2026-09-15 (written after P1–P4 were computed)

This amendment corrects the motivating numbers and records what has already been run. It changes no
definition, threshold, or verdict.

1. **Motivating numbers came from an earlier, shorter regex.** The counts in "Why this test exists" (287 negated
   statements, 242 of them labelled −1, rule baseline 777 / 1,160) were computed with a regex that lacked
   "cannot" and the n't pattern. The code (`src/negation_reslice.py`, `NEGATION`) uses exactly the written list
   under "Fixed definitions". With that list: 301 negated statements, 251 labelled −1, rule baseline
   781 / 1,160 = 67.3%.
2. **Which list the results use.** P1–P4 were judged on the written list (301 / 251 / 781). Robustness to the regex
   variant (five variants, recomputed on the real artifacts): the clean subset has 311–319 questions (319 with the
   earlier regex), clean off-diagonal pairwise changes by at most 0.12 percentage points, and the P1–P3 verdicts
   do not change.
3. **Results already computed** (`artifacts/results/negation_reslice.json`, `artifacts/results/sentiment_check.json`):
   P1 true, P2 false, P3 true, P4 true. The P3 ordering is thin: Ministral's full−clean drop (0.0280) exceeds
   Qwen's (0.0268) by 0.0012.
4. **P4 threshold.** The 58% threshold in P4 was set relative to chance (50%), not relative to a tone-only baseline.
   On the clean and sentiment-matched subset (84 questions), a tone-only pairwise baseline (English valence =
   P(positive) − P(negative)) scores 0.679. So P4 being true does not show that the probes beat tone. Amendment 3
   tests that directly.
5. **Bootstrap change.** From this amendment on, every bootstrap interval uses a fresh random generator seeded with
   `BOOT_SEED` = 20260915, so the same quantity gets the same interval in every file. This moved interval endpoints
   in `negation_reslice.json` by at most 0.0016 (checked against the committed file) and in `sentiment_check.json`
   by at most 0.0036 (that file had not been committed, so this comes from the re-run only). Point estimates and
   verdicts did not change.

---

## Amendment 3 — 2026-09-15 (Step 1: tone control; written before `src/tone_control.py` is run on real data)

Question: does the probe carry information about the ±1 label beyond the tone (sentiment) of the statement?
Implemented in `src/tone_control.py`; output `artifacts/results/tone_control.json`.

**Unit:** question (580 ±1 pairs). All quantities are per-question differences: +1 side minus −1 side.

**Probe signal per question:** over the 30 off-diagonal source→target cells (fair lane, `current_raw`, same guarded
loader as P1–P4): (a) pairwise outcome averaged over the 30 cells; (b) margin = mean over the 30 cells of
decision_score(+1) − decision_score(−1). Probe scores are already out-of-fold for each question.

**Tone:** English valence = P(positive) − P(negative) from `cardiffnlp/twitter-xlm-roberta-base-sentiment`
(revision `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8`, truncation at 512 tokens). Tone pairwise per question:
1 if valence(+1) > valence(−1), 0.5 if equal, else 0. Negation flag: the written list above, per statement.

**Checks:**

1. **Head-to-head.** Per question: probe pairwise (mean over 30 cells) minus tone pairwise. Mean with a paired
   bootstrap 95% interval (5,000 resamples of questions, seed 20260915). Subsets: all 580 questions, and clean.
2. **Tone-controlled prediction (primary).** For each question, x = features(+1 side) − features(−1 side), with
   features: English valence, negation flag, probe margin. Symmetrize: x with y = 1 and −x with y = 0.
   `LogisticRegression(C=1.0, fit_intercept=False)` (scikit-learn default solver). Each feature is divided by its
   standard deviation on the training folds only (the symmetrized mean is exactly 0, so no centering is needed).
   5-fold `GroupKFold` by question (no shuffling), so both symmetrized rows of a question are in the same fold.
   Model A = valence + negation; model B = valence + negation + probe margin. For each question, held-out log-loss on
   the y = 1 orientation, log(1 + exp(−w·x)). Improvement = logloss_A − logloss_B per question; mean with a paired
   bootstrap 95% interval (same resampled questions for A and B). Also reported: held-out pairwise accuracy of A and B
   (w·x > 0 counts 1, = 0 counts 0.5), and standardized coefficients from a fit on all questions (descriptive only).
3. **Tone-wrong pairs.** Questions with valence(−1) > valence(+1): probe off-diagonal pairwise with bootstrap 95%
   interval and n. Also reported for the near-tie set |valence(+1) − valence(−1)| < 0.10.
4. **Robustness.** Repeat checks 1–3 with:
   - **(a)** a second tone model, `nlptown/bert-base-multilingual-uncased-sentiment`
     (revision `8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d`), on English. Valence = expected stars rescaled from
     [1, 5] to [−1, 1].
   - **(b)** target-language tone: the cardiff model scores all 6 languages. Checks 1 and 3 become per-cell: for
     cell s→t, tone is from language t. In check 1, only the probe-minus-tone difference is reported for this variant. In check 3, each (question, cell) with tone wrong in language t is included,
     pooled over cells, with a bootstrap over questions. In check 2, the valence feature becomes the mean valence
     difference over the 5 non-English languages (es, de, zh, hi, mr). The negation flag stays English.

**Coverage caveat:** the cardiff model was trained on ar, en, fr, de, hi, it, es, pt (not zh or mr). nlptown was
trained on en, nl, de, fr, it, es (used here on English only). In variant 4b, the cardiff tone scores for zh and mr
come from languages outside its training set and may be noisier.

**Decision rule (per model):** "probe carries information beyond tone" if and only if all three hold:
(i) the check-2 log-loss improvement 95% interval is above 0 with cardiff English tone; (ii) the check-3
tone-wrong probe pairwise lower bound is above 0.50; (iii) the check-2 improvement interval stays above 0 under both
robustness variants 4a and 4b. Otherwise the result is reported as "not established". All outcomes are reported
regardless of direction.

**Commit record:** `src/tone_control.py` refuses to run while this file has uncommitted changes, and records the
file's commit hash in `preregistration_amendment_commit`. The same guard is used by every later amendment's script.

---

## Amendment 4 — 2026-09-15 (exploratory; written after Step 1 was run, before this check)

Step 1 found a beyond-tone signal for Gemma only. Inspecting examples raised a specific alternative: on the
tone-wrong pairs, the probe may simply prefer the statement **without** a negation word. In two examples the
probe picked the non-negated side both times, which was right once (question 73) and wrong once (question 501).
Check 2 of Amendment 3 controls for negation, but check 3 does not.

**Check (exploratory, not a pre-registered prediction):** repeat Amendment 3's check 3 on the subset of
tone-wrong questions that are also *clean* (no negation word on either English side, same list as the original
definitions). Report per model: off-diagonal probe pairwise, 95% bootstrap interval (5,000, seed 20260915,
questions resampled), and n. Same for the near-tie subset (|Δvalence| < 0.10) restricted to clean questions.

**Reading of the result:** a model's beyond-tone signal is called *robust to negation* if the lower bound on the
clean tone-wrong subset stays above 0.50. If Gemma's lower bound falls to or below 0.50, its Step 1 result is
reported as possibly explained by negation preference rather than stance. Implemented in `src/clean_tone_wrong.py`;
output `artifacts/results/clean_tone_wrong.json`. All outcomes are reported regardless of direction.
