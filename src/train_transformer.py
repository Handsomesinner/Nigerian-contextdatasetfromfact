"""
Transformer baseline: fine-tune a pretrained encoder for binary misinformation
detection on the Nigerian-context corpus.

Default checkpoint is DistilBERT (`distilbert-base-uncased`) — small enough to
fine-tune on CPU in a few minutes. Override with --model, e.g.
`microsoft/MiniLM-L12-H384-uncased` or `prajjwal1/bert-tiny`.

If the pretrained weights cannot be downloaded (offline / restricted network),
the script fails gracefully with a clear message so the classical pipeline still
constitutes a complete run.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from preprocess import prepare
from evaluate import compute_metrics, save_confusion, save_metrics

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_OUT = os.path.join(ROOT, "results", "models", "transformer")


def run(model_name: str, epochs: int, batch_size: int, max_len: int):
    try:
        import torch
        from datasets import Dataset  # optional; we fall back to manual if absent
    except Exception:
        Dataset = None  # noqa: N806
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(42)
    torch.manual_seed(42)

    train_df, test_df = prepare()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={0: "credible", 1: "misinformation"},
        label2id={"credible": 0, "misinformation": 1},
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                list(texts), truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    train_ds = TextDataset(train_df["text"], train_df["label"])
    test_ds = TextDataset(test_df["text"], test_df["label"])

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return compute_metrics(labels, preds)

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_OUT, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=hf_metrics,
    )

    trainer.train()

    logits = trainer.predict(test_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)
    y_true = test_df["label"].values

    metrics = compute_metrics(y_true, preds, probs)
    metrics["model_name"] = model_name
    metrics["epochs"] = epochs

    save_confusion(y_true, preds, "DistilBERT")
    save_metrics({f"Transformer ({model_name})": metrics}, "transformer_metrics.json")

    os.makedirs(MODEL_OUT, exist_ok=True)
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)

    print("\nTransformer test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()
    try:
        run(args.model, args.epochs, args.batch_size, args.max_len)
    except Exception as exc:  # pragma: no cover
        print(f"\n[transformer] Could not complete fine-tuning: {exc}", file=sys.stderr)
        print("[transformer] The classical pipeline still produced full results.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
