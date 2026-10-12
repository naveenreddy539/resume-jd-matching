"""Data access script: load the Resume-JD Match dataset and create clean train / validation / test splits.

Reproduces notebooks 01 and 02:
  1. Load facehuggerapoorv/resume-jd-match from Hugging Face (6,241 train / 1,759 test rows)
  2. Split each record into JD and resume, and drop the label embedded in the text
  3. Clean the text (rules 1-5)
  4. Train only: remove pairs with conflicting labels, then exact duplicates
  5. Validation split by JD (StratifiedGroupKFold, ~1/7 of JDs, random_state=42)
  6. Save train_clean / val_clean / test_clean parquet files to data/

Run from the project root:   python -m src.data_loader
Expected: train 5,285 | val 943 | test 1,759 rows; JDs 240 / 40 / 71; JD overlap 0.
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocess import SPLIT_PATTERN, clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET = "facehuggerapoorv/resume-jd-match"


def load_raw():
    from datasets import load_dataset
    ds = load_dataset(DATASET)
    return ds["train"].to_pandas(), ds["test"].to_pandas()


def split_text(df):
    parts = df["text"].str.extract(SPLIT_PATTERN, flags=re.S)
    parts.columns = ["jd", "resume", "embedded_label"]
    out = df.join(parts)
    unparsed = out["jd"].isna().sum()
    if unparsed:
        raise ValueError(f"{unparsed} rows did not match the expected format")
    return out[["jd", "resume", "label"]]          # embedded label is dropped


def remove_label_noise(train):
    conflict = train.groupby(["jd", "resume"])["label"].transform("nunique") > 1
    train = train[~conflict]
    dups = train.duplicated(subset=["jd", "resume", "label"])
    return train[~dups].reset_index(drop=True), int(conflict.sum()), int(dups.sum())


def jd_grouped_split(train):
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    groups = train["jd"].factorize()[0]
    tr_idx, val_idx = next(sgkf.split(train, train["label"], groups))
    return train.iloc[tr_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)


def main():
    raw_train, raw_test = load_raw()
    print("Raw:", raw_train.shape, raw_test.shape)

    train, test = split_text(raw_train), split_text(raw_test)
    for df in (train, test):
        df["jd"] = df["jd"].map(clean_text)
        df["resume"] = df["resume"].map(clean_text)

    train, n_conflict, n_dups = remove_label_noise(train)          # test is never deduplicated
    print(f"Removed {n_conflict} conflicting rows and {n_dups} duplicate rows from train")

    train_final, val = jd_grouped_split(train)
    print("Rows -> train:", len(train_final), "| val:", len(val), "| test:", len(test))
    print("JDs  -> train:", train_final["jd"].nunique(), "| val:", val["jd"].nunique(),
          "| test:", test["jd"].nunique(), "| JD overlap train/val:", int(val["jd"].isin(train_final["jd"]).sum()))

    DATA_DIR.mkdir(exist_ok=True)
    train_final.to_parquet(DATA_DIR / "train_clean.parquet", index=False)
    val.to_parquet(DATA_DIR / "val_clean.parquet", index=False)
    test.to_parquet(DATA_DIR / "test_clean.parquet", index=False)
    print("Saved to", DATA_DIR)


if __name__ == "__main__":
    main()
