#!/usr/bin/env python3
"""
End-to-end pipeline runner for:

    A Machine Learning Approach to Detecting Misinformation in Nigerian Social
    Media Content.

Stages
------
1. Build the curated Nigerian-context labelled dataset from fact-checking themes.
2. Train + evaluate classical baselines (LogReg, SVM, Naive Bayes, RandomForest).
3. Train + evaluate a transformer classifier:
      - preferred: fine-tune pretrained DistilBERT (needs huggingface.co access)
      - fallback : a from-scratch PyTorch Transformer encoder (always runnable)
4. Aggregate all results into a combined comparison table + figure and a report.

Run:  python run.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import write_csv, build              # noqa: E402
from evaluate import save_comparison_plot, METRIC_DIR  # noqa: E402

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")


def banner(msg):
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72)


def try_hf_transformer():
    """Attempt pretrained fine-tuning; return metrics dict or None if unavailable."""
    banner("STAGE 3a — Transformer (fine-tune pretrained DistilBERT)")
    proc = subprocess.run(
        [sys.executable, os.path.join(SRC, "train_transformer.py"),
         "--epochs", "5"],
        capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(proc.stderr[-800:])
        print("[run] Pretrained fine-tuning unavailable in this environment "
              "(model hub blocked) — using the from-scratch transformer instead.")
        return None
    path = os.path.join(METRIC_DIR, "transformer_metrics.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def main():
    banner("STAGE 1 — Build Nigerian-context labelled dataset")
    write_csv()
    recs = build()
    print(f"Dataset: {len(recs)} records "
          f"({sum(r.label for r in recs)} misinformation / "
          f"{sum(1 - r.label for r in recs)} credible)")

    banner("STAGE 2 — Classical baselines")
    from train_classical import main as run_classical
    classical = run_classical()

    # Stage 3: transformer (pretrained if possible, else from-scratch)
    hf = try_hf_transformer()

    banner("STAGE 3b — Transformer (from scratch, PyTorch)")
    from train_transformer_scratch import run as run_scratch
    scratch = run_scratch()

    # Aggregate
    banner("STAGE 4 — Combined comparison")
    combined = dict(classical)
    if hf:
        # transformer_metrics.json is keyed by model name
        for k, v in hf.items():
            combined["DistilBERT (fine-tuned)"] = v
    combined["Transformer (scratch)"] = scratch

    with open(os.path.join(METRIC_DIR, "combined_metrics.json"), "w") as fh:
        json.dump(combined, fh, indent=2)
    save_comparison_plot({k: v for k, v in combined.items()})

    # Pretty table
    keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
    header = f"{'Model':28s} " + " ".join(f"{k:>9s}" for k in keys)
    print("\n" + header)
    print("-" * len(header))
    lines = [header, "-" * len(header)]
    for name, m in combined.items():
        row = f"{name:28s} " + " ".join(
            f"{(m.get(k) if m.get(k) is not None else 0):9.3f}" for k in keys)
        print(row)
        lines.append(row)

    with open(os.path.join(ROOT, "results", "summary_table.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    best = max(combined.items(), key=lambda kv: kv[1].get("macro_f1", 0))
    print(f"\nBest model by macro-F1: {best[0]} "
          f"(macro-F1 = {best[1].get('macro_f1'):.3f})")
    print("\nArtifacts written to results/  (metrics/, figures/, summary_table.txt)")

    # Keep the deployed web demo's model in sync with the trained pipeline.
    banner("STAGE 5 — Export dependency-free model for the web demo")
    from export_web_model import main as export_web
    export_web()


if __name__ == "__main__":
    main()
