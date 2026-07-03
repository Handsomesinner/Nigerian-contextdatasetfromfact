"""
Text preprocessing and train/test splitting for the Nigerian misinformation corpus.

Two views of the text are produced:

    clean_text  : light normalisation (lowercase, URL/@handle/emoji stripping,
                  whitespace collapse) used by the classical TF-IDF models.
    text        : the original raw text, used by the transformer (its own
                  tokenizer handles casing / sub-words).
"""

from __future__ import annotations

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "data", "raw", "nigerian_factcheck_dataset.csv")
PROC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s'#]")
MULTISPACE_RE = re.compile(r"\s+")

RANDOM_STATE = 42


def clean(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def load_dataframe(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean)
    return df


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified split on the binary label so class balance is preserved."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare(test_size: float = 0.2):
    os.makedirs(PROC_DIR, exist_ok=True)
    df = load_dataframe()
    train_df, test_df = split(df, test_size=test_size)
    train_df.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = prepare()
    print(f"train: {len(train_df)}  test: {len(test_df)}")
    print("train label balance:\n", train_df["label"].value_counts().to_string())
    print("test label balance:\n", test_df["label"].value_counts().to_string())
