"""
Stage 1: Data Engineering
- download Adult Income
- clean (?, NaN, outliers)
- train/test split
"""
import os
import logging
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

NUMERIC_COLS = ["age", "fnlwgt", "education-num",
                "capital-gain", "capital-loss", "hours-per-week"]


def download_raw():
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "adult.csv")
    if not os.path.exists(path):
        log.info("Downloading Adult Income dataset...")
        urllib.request.urlretrieve(TRAIN_URL, path)
    else:
        log.info("Raw dataset already exists, skipping download.")
    return path


def load_raw(path: str) -> pd.DataFrame:
    log.info("Loading raw data...")
    df = pd.read_csv(path, header=None, names=COLUMNS,
                     na_values="?", skipinitialspace=True)
    log.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning data...")
    # strip whitespace in object columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # target to binary
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df = df.dropna(subset=["income"])

    # drop rows with too many NaNs (workclass/occupation/native-country)
    df = df.dropna()

    # outliers: IQR on age and hours-per-week
    for col in ["age", "hours-per-week"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        log.info(f"  {col}: removed {before - len(df)} outliers")

    # drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"  dropped {before - len(df)} duplicates")

    log.info(f"After cleaning: {len(df)} rows")
    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df["income"])
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    log.info(f"Saved train={len(train)}, test={len(test)} -> {PROCESSED_DIR}")


if __name__ == "__main__":
    raw_path = download_raw()
    df = load_raw(raw_path)
    df = clean(df)
    split_and_save(df)