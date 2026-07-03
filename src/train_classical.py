"""
Classical (non-neural) baselines for Nigerian misinformation detection.

Each model is a scikit-learn Pipeline of a TF-IDF vectoriser (word + char
n-grams) feeding a linear or tree classifier:

    * Logistic Regression
    * Linear SVM (calibrated for probabilities)
    * Multinomial Naive Bayes
    * Random Forest

Cross-validated on the training split, then evaluated once on the held-out
test split. Results are written to results/metrics/ and results/figures/.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import prepare
from evaluate import (
    compute_metrics,
    save_confusion,
    save_metrics,
    save_comparison_plot,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "results", "models")


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced"), cv=3)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=0.3)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight="balanced",
                random_state=42, n_jobs=-1)),
        ]),
    }


def get_scores(pipe, X):
    """Return positive-class scores for ROC-AUC, whatever the estimator exposes."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return None


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_df, test_df = prepare()
    Xtr, ytr = train_df["clean_text"].tolist(), train_df["label"].values
    Xte, yte = test_df["clean_text"].tolist(), test_df["label"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"Training on {len(Xtr)} examples, testing on {len(Xte)}.\n")
    for name, pipe in build_models().items():
        cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro")
        pipe.fit(Xtr, ytr)
        y_pred = pipe.predict(Xte)
        y_score = get_scores(pipe, Xte)
        m = compute_metrics(yte, y_pred, y_score)
        m["cv_macro_f1_mean"] = round(float(np.mean(cv_f1)), 4)
        m["cv_macro_f1_std"] = round(float(np.std(cv_f1)), 4)
        results[name] = m
        save_confusion(yte, y_pred, name)
        joblib.dump(pipe, os.path.join(
            MODEL_DIR, f"{name.lower().replace(' ', '_')}.joblib"))
        print(f"{name:22s} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
              f"macroF1={m['macro_f1']:.3f}  cvF1={m['cv_macro_f1_mean']:.3f}"
              f"±{m['cv_macro_f1_std']:.3f}")

    save_metrics(results, "classical_metrics.json")
    save_comparison_plot(results)
    return results


if __name__ == "__main__":
    main()
