"""
Export the trained TF-IDF + Logistic Regression pipeline to a dependency-free
JSON model (`web/model.json`) so the Vercel serverless demo can run inference in
pure Python — no numpy / scikit-learn / torch needed at request time.

The export stores, per vocabulary n-gram, its IDF weight and the logistic-
regression coefficient, plus the intercept. Inference re-implements exactly what
`TfidfVectorizer(sublinear_tf=True, norm="l2")` + `LogisticRegression` do:

    tf'      = 1 + ln(tf)                       (sublinear term frequency)
    v[gram]  = tf' * idf[gram]
    v        = v / ||v||_2                       (L2 normalisation)
    score    = intercept + Σ v[gram] * coef[gram]
    p(misinfo) = sigmoid(score)

A parity check against scikit-learn's own predict_proba is run before writing.
"""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from preprocess import prepare, clean
from train_classical import make_vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = os.path.dirname(os.path.dirname(__file__))
# Colocated with the serverless function so Vercel bundles it automatically.
OUT = os.path.join(ROOT, "api", "model.json")
TOKEN_RE = re.compile(r"\b\w\w+\b")


def build_ngrams(text: str, ngram_max: int = 2):
    tokens = TOKEN_RE.findall(clean(text))
    grams = list(tokens)
    for n in range(2, ngram_max + 1):
        grams += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams


def python_predict_proba(text, vocab_idf, coef, intercept):
    counts = {}
    for g in build_ngrams(text):
        if g in vocab_idf:
            counts[g] = counts.get(g, 0) + 1
    vec = {g: (1.0 + math.log(c)) * vocab_idf[g] for g, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    score = intercept + sum((v / norm) * coef[g] for g, v in vec.items())
    return 1.0 / (1.0 + math.exp(-score))


def main():
    train_df, test_df = prepare()
    pipe = Pipeline([
        ("tfidf", make_vectorizer()),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    pipe.fit(train_df["clean_text"], train_df["label"])

    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    idf = vec.idf_
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    vocab = vec.vocabulary_  # ngram -> column index

    vocab_idf = {g: float(idf[i]) for g, i in vocab.items()}
    coef = {g: float(coefs[i]) for g, i in vocab.items()}

    # ---- parity check against scikit-learn -------------------------------
    sk = pipe.predict_proba(test_df["clean_text"])[:, 1]
    py = np.array([python_predict_proba(t, vocab_idf, coef, intercept)
                   for t in test_df["text"]])
    max_diff = float(np.max(np.abs(sk - py)))
    print(f"Parity check vs scikit-learn: max |Δ p(misinfo)| = {max_diff:.2e}")
    assert max_diff < 1e-6, "pure-python inference does not match scikit-learn!"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model = {
        "model": "tfidf(1,2)+logreg",
        "task": "nigerian-misinformation-detection",
        "labels": {"0": "credible", "1": "misinformation"},
        "intercept": intercept,
        "ngram_max": 2,
        "vocab": {g: [vocab_idf[g], coef[g]] for g in vocab},
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(model, fh)
    kb = os.path.getsize(OUT) / 1024
    print(f"Wrote {OUT} ({len(vocab)} n-grams, {kb:.1f} KB)")


if __name__ == "__main__":
    main()
