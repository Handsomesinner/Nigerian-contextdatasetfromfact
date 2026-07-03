"""
Transformer-based classifier trained FROM SCRATCH (no pretrained weights).

Why from scratch?  The recommended approach is to fine-tune a pretrained encoder
(DistilBERT) via `src/train_transformer.py`. In this execution environment the
model hub (huggingface.co) is blocked by the network egress policy, so pretrained
weights cannot be downloaded. To still provide an *executed* transformer point in
the comparison, this script builds a compact Transformer encoder in pure PyTorch
and trains it on the corpus.

Architecture:  word-piece-free whitespace tokenizer  ->  token embedding +
learned positional embedding  ->  N stacked multi-head self-attention encoder
layers  ->  mean pooling  ->  linear classification head.

NOTE ON EXPECTATIONS:  a transformer trained from scratch on a few hundred short
examples cannot match a fine-tuned pretrained model and often *under*-performs
the TF-IDF baselines. That contrast is itself an informative result: transformers
need either large-scale pretraining or much larger datasets to shine. The
comparison is reported honestly.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from preprocess import prepare
from evaluate import compute_metrics, save_confusion, save_metrics

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_OUT = os.path.join(ROOT, "results", "models", "transformer_scratch.pt")

TOKEN_RE = re.compile(r"[a-z0-9']+")
PAD, UNK = "<pad>", "<unk>"
SEED = 42


def tokenize(text: str):
    return TOKEN_RE.findall(str(text).lower())


class Vocab:
    def __init__(self, texts, min_freq=1, max_size=8000):
        counter = Counter(tok for t in texts for tok in tokenize(t))
        self.itos = [PAD, UNK] + [
            w for w, c in counter.most_common(max_size) if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.stoi.get(tok, 1) for tok in tokenize(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids


class TextDS(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.x = torch.tensor([vocab.encode(t, max_len) for t in texts],
                              dtype=torch.long)
        self.y = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2,
                 dim_ff=256, max_len=64, dropout=0.2, n_classes=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)
        self.max_len = max_len

    def forward(self, x):
        pad_mask = x == 0  # (B, L) True where padding
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.tok_emb(x) * math.sqrt(self.tok_emb.embedding_dim) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # mean-pool over non-pad tokens
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.head(self.dropout(pooled))


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run(epochs=40, batch_size=16, max_len=64, lr=3e-4, d_model=128, layers=2):
    set_seed()
    device = torch.device("cpu")
    train_df, test_df = prepare()

    vocab = Vocab(train_df["clean_text"].tolist())
    train_ds = TextDS(train_df["clean_text"], train_df["label"], vocab, max_len)
    test_ds = TextDS(test_df["clean_text"], test_df["label"], vocab, max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = TransformerClassifier(len(vocab), d_model=d_model, layers=layers,
                                  max_len=max_len).to(device)

    # class weights for the mild imbalance
    counts = np.bincount(train_df["label"].values, minlength=2)
    weights = torch.tensor(counts.sum() / (2.0 * np.clip(counts, 1, None)),
                           dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"From-scratch Transformer: {n_params:,} params, vocab={len(vocab)}, "
          f"{epochs} epochs on {len(train_ds)} examples.")

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * len(xb)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  train_loss={tot / len(train_ds):.4f}")

    # evaluation
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            all_logits.append(model(xb))
            all_y.append(yb)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=1).numpy()[:, 1]
    preds = logits.argmax(1).numpy()

    metrics = compute_metrics(y_true, preds, probs)
    metrics["architecture"] = "from-scratch transformer encoder"
    metrics["parameters"] = int(n_params)
    metrics["epochs"] = epochs

    save_confusion(y_true, preds, "Transformer (scratch)")
    save_metrics({"Transformer (scratch)": metrics}, "transformer_scratch_metrics.json")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos}, MODEL_OUT)

    print("\nFrom-scratch Transformer test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len,
        lr=args.lr, layers=args.layers)


if __name__ == "__main__":
    main()
