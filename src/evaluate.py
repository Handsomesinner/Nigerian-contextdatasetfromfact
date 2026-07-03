"""Shared evaluation utilities: metrics, confusion matrices and plots."""

from __future__ import annotations

import json
import os
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(ROOT, "results", "figures")
METRIC_DIR = os.path.join(ROOT, "results", "metrics")
LABELS = ["credible", "misinformation"]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                    y_score: Sequence[float] | None = None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    out = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except ValueError:
            out["roc_auc"] = None
    return out


def save_confusion(y_true, y_pred, model_name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=LABELS)
    ax.set_yticks([0, 1], labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_metrics(all_metrics: Dict[str, Dict], filename: str = "all_metrics.json") -> str:
    os.makedirs(METRIC_DIR, exist_ok=True)
    path = os.path.join(METRIC_DIR, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    return path


def save_comparison_plot(all_metrics: Dict[str, Dict]) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    models = list(all_metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(models)), 4.5))
    for i, mk in enumerate(metric_keys):
        vals = [all_metrics[m].get(mk, 0) or 0 for m in models]
        ax.bar(x + (i - 1.5) * width, vals, width, label=mk)
    ax.set_xticks(x, labels=models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classifier comparison — Nigerian misinformation detection")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "model_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
