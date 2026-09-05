# Are Political Probes Polyglots?

Last updated: 2026-09-05

## Publication record

- Medium (live): https://medium.com/@atharva2099/are-political-probes-polyglots-58c1fe462183
- LessWrong: https://www.lesswrong.com/posts/L42d794BKhoiczpdN/are-political-probes-polyglots
- LessWrong status: rejected at automated gate for new accounts — "No LLM generated, assisted/co-written, or edited work." Moderation message stated this includes content the author wrote themselves and had an LLM edit for style/consistency/clarity where a non-trivial portion of resulting text came from the LLM. Post was not judged on experiments/numbers. Account age at posting: ~5 days.
- Dataset: https://huggingface.co/datasets/Atharva2099/multilingual-political-statements
- Code: https://github.com/Atharva2099/AI-safety-Research

## Post (verbatim, Medium version)

# Are Political Probes Polyglots?

### Testing cross-language transfer across six languages, four models, and eight activation-extraction methods

I set out to study how different model families represent political positions across languages, and whether a classifier trained on one language would continue to work in another.

I am new to interpretability research, and I wanted to begin with a question that was small, relatively unexplored, and familiar to me. I decided to test whether opposing responses to political survey questions could be decoded from model activations across languages, and whether the same probes would hold cross-lingually.

## The Question

Consider two statements:

> "Germany has too much influence in the European Union."
> "Germany has too little influence in the European Union."

A classifier can learn to separate these opposing responses using vectors from a model's residual stream. But what happens when the statements are translated into German, Spanish, Mandarin Chinese, Hindi, and Marathi? If we train a classifier in one language and apply it unchanged in another, does it still work?

If the classifier still worked in another language, one possible explanation would be that the model represents the distinction between opposing political statements similarly across languages. But there are simpler explanations. The classifier might be picking up patterns created by translation, sentence structure, punctuation, or the way each language is split into tokens.

So I narrowed the question down to:

**If a linear classifier learns to separate opposing political statements in one language, can it make the same distinction in another language without being retrained?**

I tested this across six languages and four models. Mean cross-language accuracy ranged from roughly 62% to 85% across all extraction conditions discussed later. But these scores changed depending on how we read the model's activations. Using the final token, averaging all tokens, removing punctuation, or rescaling every vector to the same length affected each model differently.

## Dataset Construction

I started with the `Anthropic/llm_global_opinions` dataset on Hugging Face. It contains 2,556 questions drawn from the Pew Global Attitudes Survey and the World Values Survey.

The original dataset contains questions and response options, but I needed standalone statements with polarity labels. I processed every question using `gemini-3.5-flash-lite`, converting suitable response options into short declarative statements expressing opposing policy positions.

I excluded questions that asked for factual predictions, ratings of politicians or parties, vague feelings or perceptions, or responses without clear opposing policy positions. This left 580 suitable questions: 494 with paired responses and 86 with a substantive neutral response.

For each retained question, the two opposing statements received labels `+1` and `-1`. A `0` label was included only when the original question provided an explicit middle or neutral response.

This produced 1,246 English statements:

- 580 with polarity `+1`
- 580 with polarity `-1`
- 86 with polarity `0`

I retained the neutral statements in the dataset, but the probe experiments used only the `+1` and `-1` statements.

I then processed the English statements with the same Gemini model to produce versions in Spanish (`es`), German (`de`), Simplified Mandarin Chinese (`zh`), Hindi (`hi`), and Marathi (`mr`), giving me six languages including English. The translation process was designed to preserve the meaning, polarity, scope, and strength of each statement.

The derived dataset is available under the original license on Hugging Face.

All sentences and translations were manually reviewed with my limited language familiarity of English, German, Hindi and Marathi. Independent fluent speaker review, especially for Spanish and Mandarin, remains to be done.

## Where is polarity readable?

I started by extracting the final non-padding token's residual-stream vector at every transformer layer. I then trained a logistic-regression probe to predict whether each statement had positive or negative polarity.

I evaluated the probe using grouped five-fold cross-validation, grouped by question ID so there was no data leak during evaluation.

The best English-language results were:

- OLMo 3 7B, Layer 17: 81.03%
- Qwen 3.5 9B, Layers 12-13: 85.52%
- Gemma 2 9B, Layer 23: 86.21%
- Ministral 8B, Layer 31: 86.55%

The peak occurred at a different layer for each model. In every case, the final layer performed worse than the best-performing layer.

This does not tell me when the model formed the distinction captured by the probe. It also does not show that later layers erased political information. It only identifies where this particular linear probe achieved its highest accuracy.

## Testing cross-lingual transfer

The first analysis identified where polarity was most readable in English, but that layer was not necessarily preferred by every other language. Before testing transfer, I therefore repeated the layer sweep for all six languages in each model.

The preferred layer varied substantially in some models. For example, Ministral peaked at Layer 9 for Hindi, Layer 20 for Mandarin, and Layers 31-33 for the remaining languages. Gemma's language-specific peaks ranged from Layer 13 for Marathi to Layer 25 for Spanish.

Preferred layers (en, es, de, zh, hi, mr):

- OLMo 3 7B: 17, 10, 17, 17, 15, 18
- Qwen 3.5 9B: 14, 14, 14, 12, 16, 10
- Gemma 2 9B: 15, 25, 22, 18, 24, 13
- Ministral 8B: 31, 32, 31, 20, 9, 33

These preferred layers come from the multilingual sweep. They can differ from the earlier English-only results because the two analyses used different cross-validation procedures. For example, Qwen's English-only sweep tied at Layers 12-13, while its English result in the multilingual sweep peaked at Layer 14.

Rather than using the preferred English layer for every source language, I trained each probe at that language's preferred layer. I then applied it unchanged to all six languages in the same model.

## What the transfer heatmap actually shows

On average, all four models transfer well above the 50% random chance baseline, but cross-lingual linear decodability varies substantially across architectures. Gemma 2 and Qwen 3.5 maintain a high transfer accuracy across all languages (averaging ~81.6% and ~78.3% respectively), whereas OLMo 3 (~68.6%) and Ministral 8B (~67.4%) experience a much steeper drop off compared to the diagonal.

There is also clear directional asymmetry. In Ministral, a probe trained on Spanish at Layer 32 reaches 70.9% on Hindi statements, but a probe trained on Hindi at Layer 9 reaches only 53.5% when evaluated on Spanish statements.

To check whether the transfer result depended on one particular layer, I repeated the full 6x6 transfer matrix at every unique language-specific peak. Layer choice changed the exact scores but not the broad ordering between models. Gemma's mean cross-language accuracy ranged from 80.0% to 82.3%, while Qwen ranged from 77.0% to 78.4%. OLMo was more sensitive, ranging from 65.1% to 68.9%, as was Ministral, which ranged from 63.0% to 67.8%.

## What is the final token measuring?

After observing these transfer scores, I wanted to stress-test my core assumptions. What if anchoring the evaluation entirely on the final token was misleading? What if the probe was simply looking at the final punctuation and making a decision instead of the broader semantics?

The issues with only auditing the final token are:

- Punctuation artifacts: different languages and tokenizers handle terminal punctuation differently (e.g. English `.`, Hindi `।`, Chinese `。`). Is the probe reading political stance, or memorising punctuation-related token features?
- Vector magnitude vs. direction: linear classifiers rely on both vector direction and norm. Is transfer driven by the actual direction the vector is pointing, or just by its magnitude?
- Information distribution: assuming political meaning is saturated in the final token is a naive assumption; it could very well be distributed across the sentence.

To test this, I compared eight ways of extracting an activation. I used the final token, the last non-punctuation token, the mean across all content tokens, or the final token after removing punctuation from the end of the sentence. For each method, I tested both the raw residual-stream vector and an L2-normalised version.

Changing the extraction method and layer at the same time would make the results difficult to interpret. I therefore fixed one reference layer for each model: Layer 17 for OLMo, Layer 12 for Qwen, Layer 23 for Gemma, and Layer 31 for Ministral.

I used the English-only peak layers as fixed reference points. For Qwen, Layers 12 and 13 tied, so I selected Layer 12 using an earliest-layer tie rule. These are reference points for comparing extraction methods, not universally optimal layers.

## Diagnostic findings

Raw residual-stream vectors, mean cross-language accuracy:

- Gemma 2 9B: final 84.67%, content 80.27%, mean 76.01%, stripped 80.28%
- Qwen 3.5 9B: final 80.68%, content 71.57%, mean 68.63%, stripped 71.48%
- OLMo 3 7B: final 70.64%, content 71.45%, mean 68.50%, stripped 72.07%
- Ministral 8B: final 68.13%, content 68.11%, mean 67.64%, stripped 68.18%

L2-normalised residual-stream vectors, mean cross-language accuracy:

- Gemma 2 9B: final 85.11%, content 80.91%, mean 75.17%, stripped 80.90%
- Qwen 3.5 9B: final 81.48%, content 73.00%, mean 69.99%, stripped 73.04%
- OLMo 3 7B: final 72.50%, content 72.80%, mean 70.50%, stripped 73.28%
- Ministral 8B: final 62.22%, content 70.28%, mean 67.14%, stripped 70.27%

Gemma and Qwen performed best with the final-token extraction. Averaging the content-token vectors reduced accuracy by 8.7 and 12.1 percentage points, respectively. Stripping punctuation from the end of each sentence reduced accuracy by 4.4 points for Gemma and 9.2 points for Qwen.

OLMo improved after stripping punctuation by 1.4 points and after L2-normalising the final-token vector by 1.9 points. Its best condition, `stripped_l2`, was 2.6 points above its baseline. In contrast, normalising Ministral's final-token vector reduced transfer by 5.9 points. This suggests that no single extraction method works best across all four models.

## Could simpler features explain the result?

The controls help me understand what the probes are using to make their decisions. Are they truly separating the statements based on political stance, or are they picking up word or sentence-level patterns that happen to track the labels? Since the statements were generated, they may contain recurring patterns or structures that I missed. I used these controls to test some of these possible shortcuts. L2 normalisation tests whether vector length matters, mean pooling checks whether the result depends on the final token, and the other extraction methods test the effects of token choice and punctuation. Strong transfer alone does not justify saying that the probes have identified political stance.

The eight extraction methods test where the signal sits inside the models' activations, but they don't check whether the raw text itself contains any shortcuts. To test the text itself, I trained character 3-5-gram TF-IDF classifiers separately for each language and used logistic regression for classification. I evaluated them using question-grouped five-fold cross-validation.

The character n-gram models had an average within-language accuracy of 81.42% (English 83.6%, Hindi 83.3%, Marathi 81.6%, German 80.8%, Spanish 79.7%, and Mandarin 79.7%). Upon cross-language testing of these models, the average dropped to 51.67%, basically random chance. It only showed slight transfer between related languages: English and Spanish scored between 61% and 63%, while Marathi and Hindi scored between 59% and 61%. Between completely different scripts like English to Mandarin or Hindi, the scores were a flat 50%.

The internal activation probes transferred at 62% to 85% across entirely different scripts, whereas character n-grams failed completely at 51.7%. This confirms that the activation probes are not just picking up on surface character overlap across languages.

Ruling out character overlap doesn't prove the probes are catching actual political beliefs. The models could just have strong multilingual alignment across the board, or the translations from Gemini might share repeated semantic patterns. The next natural test is using off-the-shelf sentence embeddings to set a stronger baseline.

## What this work still misses

1. Synthetic and translated data: all sentences were generated and translated by Gemini 3.5 Flash Lite. While I checked them to the best of my ability across four languages, the dataset still lacks independent fluent review for Spanish and Mandarin. Real-world political speech is also a lot messier than these binary pairs.
2. Decodability isn't causality: probing only proves that the vectors are linearly separable. It doesn't prove the model "understands" political concepts, or that it uses this direction when thinking through a response.
3. Extraction sensitivity: as seen in the diagnostics above, simple choices like switching from the final token to mean pooling can drop accuracy by 8 to 12 points. Even something as dumb as changing the batch size from 8 to 16 flips border predictions.

## What this leaves us with

My main takeaway is that cross-lingual linear alignment exists and is genuinely strong in models like Gemma 2 and Qwen 3.5, but much weaker in OLMo 3 and Ministral 8B. How we extract representations matters just as much as which layer we choose.

## Open questions

1. Cross-lingual activation steering: does adding the English direction during generation in Hindi or Mandarin actually tilt the model's stance on that issue? Because the probe was trained on statement polarity rather than a unified Left/Right axis, steering would test whether polarity itself transfers causally across languages.
2. External embedding baselines: running a dedicated multilingual embedding model (like LaBSE or text-embedding-3) on these exact statements would show whether internal activations do anything special, or if standard text embeddings transfer just as well.
3. Evaluating on natural human datasets: testing on human-authored political debates across languages would strip out LLM translation artifacts and give a much truer test of cross-lingual political representation.
