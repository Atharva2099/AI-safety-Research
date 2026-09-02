# LessWrong Blog Working Notes

Last updated: 2026-09-01

## Purpose

Running notes for a possible LessWrong post about the cross-lingual political-representation experiment.

This document is a planning and evidence checklist. It is not the post itself.

The post must discuss only this project.

## AI-writing constraint

LessWrong’s current policy has a stricter standard for first-time writers than for established users.

The safe interpretation for this post is:

- The submitted prose must be written by the human author.
- AI-generated draft text must not be copied, lightly edited, humanized, or submitted as human writing.
- AI may assist with research, source discovery, factual verification, outlining, and criticism.
- Research or arguments discovered with AI assistance may be used, but the author must understand them and express them independently.
- AI should not supply publication-ready replacements during the final review.
- Every claim, number, interpretation, and citation must be understood and verified by the author.

The existing AI-generated Draft 1 is reference material only. It must not become the submission through editing or paraphrasing.

### Allowed workflow for this post

1. Research relevant LessWrong posts and external literature.
2. Maintain an evidence table and outline in this planning document.
3. The author writes each section independently.
4. AI reviews the human draft by identifying:
   - unsupported claims;
   - reasoning gaps;
   - unclear terms;
   - missing counterarguments;
   - factual or numerical inconsistencies;
   - passages that resemble generic AI prose.
5. The author decides how to revise and writes the replacement wording.
6. Perform a final source and evidence audit without AI rewriting the prose.

Do not run the submission through an AI humanizer. Substantial AI revision would count as LLM output under the current policy.

## LessWrong guidance

For a new user’s first post:

- State the main point in the opening paragraphs.
- Give the strongest evidence early.
- Explain why the result is relevant to LessWrong.
- Communicate a clear model, argument, or body of evidence.
- Address important existing arguments rather than unknowingly restarting an old debate.
- Explain rather than persuade.
- Be curious about alternative explanations and what could make the interpretation wrong.
- Avoid unmarked AI-written or AI-coauthored prose.

### Primary policy sources

- [Policy for LLM Writing on LessWrong](https://www.lesswrong.com/posts/KXujJjnmP85u8eM6B/policy-for-llm-writing-on-lesswrong)
- [New LessWrong Editor and March 2026 LLM-policy update](https://www.lesswrong.com/posts/nQWavk9mnwcv6ScMR/new-lesswrong-editor-also-an-update-to-our-llm-policy)
- [LessWrong FAQ](https://www.lesswrong.com/faq)

The March 2026 policy defines “LLM output” as:

- text written entirely by an LLM;
- human text substantially edited by an LLM;
- LLM text later edited by a human.

It excludes human-written text that uses facts or arguments found with AI assistance, provided the human does not borrow the AI’s wording.

## Redundancy and related-work audit

Current conclusion: no direct LessWrong duplicate was found, but the search is not exhaustive.

The post appears publishable if it explicitly distinguishes its contribution from existing work and does not claim broad novelty beyond the evidence.

### Closely related LessWrong posts

#### Models have linear representations of what tasks they like

- Link: [Models have linear representations of what tasks they like](https://www.lesswrong.com/posts/pxC2RAeoBrvK8ivMf/models-have-linear-representations-of-what-tasks-they-like-1)
- Published: 2026-03-05
- Overlap:
  - residual-stream linear probes;
  - tests whether a learned direction generalizes out of distribution;
  - explicitly asks whether probes capture an evaluative representation or descriptive task features;
  - includes causal steering.
- Difference:
  - studies task preferences rather than political-statement polarity;
  - does not test transfer across six languages;
  - does not perform our token-extraction, punctuation, mean-pooling, and vector-normalization comparison.
- Lesson for our post:
  - acknowledge that probe success can arise from descriptive features;
  - do not equate decodability with an abstract or causal representation.

#### Making Linear Probes Interpretable

- Link: [Making Linear Probes Interpretable](https://www.lesswrong.com/posts/voNMRijPWkwcQ4ufB/making-linear-probes-interpretable)
- Published: 2025-12-18
- Overlap:
  - trains linear probes on model activations;
  - focuses on discovering which features make a probe work;
  - uses final-token activations.
- Difference:
  - studies sparse probes over sparse-autoencoder features;
  - does not test multilingual transfer or extraction sensitivity.
- Lesson for our post:
  - high probe accuracy does not explain which underlying features carry the signal.

#### Detecting Strategic Deception Using Linear Probes

- Link: [Detecting Strategic Deception Using Linear Probes](https://www.lesswrong.com/posts/9pGbTz6c78PGwJein/detecting-strategic-deception-using-linear-probes)
- Published: 2025-02-06
- Overlap:
  - uses logistic-regression probes on internal activations;
  - tests transfer from simple training data to more realistic settings;
  - discusses false positives and topic-related activation.
- Difference:
  - studies honesty and deception rather than political statements;
  - does not test cross-language representation transfer.
- Lesson for our post:
  - probe performance can reflect related subject matter rather than the intended latent property.

### What this project contributes

The defensible contribution is the combination of:

- matched political statements in six languages;
- cross-language probe transfer;
- layerwise linear decodability;
- four representation-extraction strategies;
- raw and L2-normalized variants;
- text-surface and scalar-vector-magnitude controls;
- explicit evidence that extraction sensitivity differs by model.

Avoid saying that this is the first study of multilingual model representations. The LessWrong search does not establish novelty across the wider literature.

## Central question

If a linear probe learns to distinguish opposing political statements in one language, can the same fitted probe make that distinction in another language?

The experiment tests cross-language linear decodability. It does not directly test:

- political beliefs;
- ideological understanding;
- a universal multilingual political concept;
- causal use of the decoded direction;
- complete coordinate alignment between languages.

## Proposed story

### 1. Motivation

- Begin with one matched statement in two languages.
- Ask whether a boundary learned in one language transfers unchanged to another.
- Explain why this matters for interpreting multilingual model representations.
- State the strongest bounded result near the beginning.

### 2. Constructing the data

- Identify the Pew and World Values Survey source material.
- Explain how questions were filtered.
- Explain how declarative statements and six language versions were generated.
- State the number of questions, statements, labels, and languages.
- Disclose the use of Gemini in dataset construction.
- State clearly whether translations received independent human validation.

Primary evidence: [dataset.md](dataset.md).

### 3. Initial final-token analysis

- Define a hidden-state vector.
- Define a linear probe.
- Explain grouped five-fold cross-validation by question ID.
- Show where within-language accuracy peaked across layers.
- Explain why peak decodability does not identify where a representation formed.
- Introduce the cross-language transfer matrix.

Primary evidence: [rq1_findings.md](rq1_findings.md) and [rq2_findings.md](rq2_findings.md).

### 4. The measurement concern

The original analysis used the final non-padding token.

Possible confounds or dependencies:

- terminal punctuation;
- tokenizer differences across languages and scripts;
- vector magnitude;
- whether information is concentrated at one token or distributed across the statement;
- common patterns introduced during generated translation.

Describe these as plausible alternative explanations, not as proven flaws.

### 5. Follow-up extraction tests

Compare:

- current final token;
- content token before terminal punctuation;
- mean pooling;
- a fresh forward pass after stripping terminal punctuation.

For each, compare:

- raw activation vectors;
- L2-normalized vectors.

Explain L2 normalization as dividing a vector by its length, preserving direction while removing overall magnitude.

Present each methodological choice beside the result it produced.

### 6. Results

The central pattern is model-specific sensitivity:

- Gemma and Qwen favor the original final-token extraction over mean pooling and punctuation stripping.
- OLMo improves modestly after punctuation stripping and normalization.
- Ministral is harmed by normalizing the original final token but improves when the stripped representation is normalized.

Do not present one extraction method as universally correct.

### 7. Controls and counterarguments

Address:

- the earlier pooled 61.48% surface-control figure is retracted because it combined incompatible controls;
- corrected character 3–5-gram TF-IDF accuracy is 81.42% within language but 51.67% across languages;
- direct character transfer is concentrated in English↔Spanish and Hindi↔Marathi;
- the n-gram control does not test ordinary multilingual semantic alignment;
- surface information exists in the dataset, but direct character overlap does not reproduce the activation-transfer matrix;
- scalar vector magnitude alone performed near chance;
- held-out question IDs are not the same as held-out political topics;
- generated translations may preserve systematic phrasing;
- Qwen’s historical Layer 12 matrix is reproduced exactly when extraction batch size is restored from 16 to 8;
- parity discrepancies for the other three models remain unresolved;
- decodability does not imply causality.

### 8. Revised conclusion

The result supports cross-language linear decodability of political-statement polarity for these models, layers, languages, and generated statements.

The extraction analysis shows that measured transfer depends on how the representation is read out, with different models responding differently.

End with experiments that would change the interpretation:

- independently validated translations;
- topic-held-out evaluation;
- stronger lexical and tokenizer controls;
- deliberately mismatched cross-language pairs;
- additional language families and models;
- causal intervention on the decoded direction.

## Introduction positioning

Do not use AI-generated prose from this section in the submission.

The introduction should establish:

1. Linear probes are already used on LessWrong to study internal model properties.
2. Existing work warns that probe success may reflect descriptive or surface features.
3. This project asks a different question: whether a political-statement boundary transfers across six languages.
4. The project’s most useful result is not simply that transfer occurs, but that its measured strength depends on token extraction and normalization.

## Claims requiring special care

| Tempting claim | Safer interpretation |
|---|---|
| The models share a political concept across languages. | A linear boundary trained in one language retains predictive information in another. |
| Political information emerges at the peak layer. | Linear decodability reaches its measured maximum at that layer. |
| Surface features do not explain the result. | Simple surface baselines perform worse, but surface-feature use has not been ruled out. |
| Vector magnitude does not matter. | Magnitude alone is near chance, while normalization changes transfer in some conditions. |
| Punctuation causes the transfer. | Removing punctuation changes measured transfer, with different effects across models. |
| We discovered a universal extraction method. | No extraction method performs best for every model. |
| This is the first multilingual representation study. | No direct LessWrong duplicate was found in a non-exhaustive search. |

## Facts to resolve before writing

- Exact Gemini model used for statement generation.
- Exact model or process used for translation.
- Whether translations were independently checked.
- Operational meaning of positive, negative, and neutral polarity.
- Exact layer-selection rule used for the reported transfer matrices.
- Whether aggregate transfer is an unweighted mean over matrix cells or pooled over predictions.
- Cause and interpretation of the legacy parity discrepancy.
- Final plot for the eight extraction conditions.

## Human-author checklist

Before submitting:

- [ ] I wrote the submitted prose myself.
- [ ] I did not copy or paraphrase the AI-generated Draft 1.
- [ ] I understand every method and statistical comparison.
- [ ] I independently checked every reported number.
- [ ] I read the related LessWrong posts cited above.
- [ ] I addressed the descriptive-feature and surface-cue objections.
- [ ] I explained why the post is relevant to LessWrong.
- [ ] The opening states the main finding and its strongest limitation.
- [ ] Every plot has a plain-language interpretation.
- [ ] Measured results are separated from interpretation.
- [ ] The conclusion does not imply causality or universal shared geometry.
- [ ] Any remaining AI-produced material is handled according to the current LessWrong policy.
