"""Tone (sentiment) scores for the +/-1 statements, cached per model in artifacts/results/tone_scores_<slug>.json.

Cache layout: {"model", "revision", "labels", "max_length", "versions", "langs": {lang: {"text_sha256", "probs"}}},
where probs[i] belongs to row i of load_data(...)[lang] (-1 then +1 per question). Scoring runs once per language.
"""
import hashlib
import json

import numpy as np

from src.metrics import RES

# Revisions pinned 2026-09-15 (huggingface_hub model_info(...).sha).
# cardiff is the model used for tone-matching; the others are independent validators trained on
# different domains (product reviews, movie reviews, mixed English sources, tweets).
TONE_MODELS = {
    "cardiff": ("cardiffnlp/twitter-xlm-roberta-base-sentiment", "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"),
    "nlptown": ("nlptown/bert-base-multilingual-uncased-sentiment", "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"),
    "sst2": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0fa89d2f80546fda750413ed43d93601a13"),
    "siebert": ("siebert/sentiment-roberta-large-english", "74cea614e245b0832c770ec9aa51bd58df965b9c"),
    "twitter_rob": ("cardiffnlp/twitter-roberta-base-sentiment-latest", "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"),
    # German / Spanish validators (sst2, siebert and twitter_rob are English-only).
    "german": ("oliverguhr/german-sentiment-bert", "b1177ff59e305c966836ba2825d3dc2efc53f125"),
    "spanish": ("pysentimiento/robertuito-sentiment-analysis", "a2cc0f67ebd705c55191e25a05ba23d885fcc09b"),
    "xlmr_multi": ("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", "82107f4ccba672c9ab1eee538e727e462cad21de"),
}
# Which validators can be trusted on which language (a model scoring text it was never trained on
# proves nothing). cardiff is excluded everywhere: it is the model used for matching.
VALIDATORS_BY_LANG = {
    "en": ["nlptown", "sst2", "siebert", "twitter_rob", "xlmr_multi"],
    "de": ["nlptown", "german", "xlmr_multi"],
    "es": ["nlptown", "spanish", "xlmr_multi"],
}
MAX_LENGTH = 512


def score_texts(key, texts, batch_size=32):
    """[n, k] softmax probabilities and the model's label names (read from its config)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name, rev = TONE_MODELS[key]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev).eval()
    names = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    # Models differ in how many positions they have (robertuito holds 130, BERT 512). Truncating at a
    # fixed 512 crashes the short ones, so cap at what this model actually supports; -2 leaves room
    # for the special tokens that RoBERTa-family position offsets consume.
    limit = getattr(model.config, "max_position_embeddings", MAX_LENGTH)
    max_len = max(16, min(MAX_LENGTH, limit - 2))
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            out.append(model(**enc).logits.softmax(-1).numpy())
    return np.concatenate(out), names


def versions():
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__}


def tone_probs(key, lang, texts):
    """Cached probabilities for one language's statements (row order as given). Scores and caches on first use."""
    path = RES / f"tone_scores_{key}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "model": TONE_MODELS[key][0], "revision": TONE_MODELS[key][1], "max_length": MAX_LENGTH, "langs": {}}
    assert cache["revision"] == TONE_MODELS[key][1], f"{path.name}: cached revision differs from pinned revision"
    sha = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    if lang not in cache["langs"]:
        probs, names = score_texts(key, texts)
        cache.update(labels=names, versions=versions())
        cache["langs"][lang] = {"text_sha256": sha, "probs": probs.tolist()}
        path.write_text(json.dumps(cache) + "\n", encoding="utf-8")
    assert cache["langs"][lang]["text_sha256"] == sha, f"{path.name}: {lang} statements changed since scoring"
    return np.array(cache["langs"][lang]["probs"]), cache["labels"]


def valence(key, probs, names):
    """Signed tone in [-1, 1], however the model labels its classes.

    positive/negative labels (2- or 3-class): P(positive) - P(negative).
    star labels (nlptown): expected stars rescaled from [1, 5] to [-1, 1].
    """
    alias = {"neg": "negative", "pos": "positive", "neu": "neutral"}
    norm = [alias.get(n, n) for n in names]
    if "positive" in norm and "negative" in norm:
        return probs[:, norm.index("positive")] - probs[:, norm.index("negative")]
    if not all(n.split()[0].isdigit() for n in norm):
        raise ValueError(f"{key}: cannot read tone from labels {names}; add them to the alias map")
    stars = np.array([int(n.split()[0]) for n in norm])
    return (probs @ stars - 3) / 2


def cache_meta(key):
    """Model, revision, max_length and library versions recorded in the cache (for output JSONs)."""
    c = json.loads((RES / f"tone_scores_{key}.json").read_text(encoding="utf-8"))
    return {k: c[k] for k in ("model", "revision", "max_length", "versions")}
