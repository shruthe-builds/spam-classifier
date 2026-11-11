"""
download_data.py
-----------------
Downloads the SMS Spam Collection dataset (same data referenced by the
UCI SMS Spam Collection / Kaggle spam datasets in the project brief)
and saves it as data/spam.csv in the format the rest of the project expects:
columns -> label, message
"""

import os
import pandas as pd
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
RAW_PATH = "data/spam_raw.csv"
CLEAN_PATH = "data/spam.csv"


def download():
    os.makedirs("data", exist_ok=True)
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    print(f"Saved raw file to {RAW_PATH}")

    # The raw file has extra empty columns and latin-1 encoding
    df = pd.read_csv(RAW_PATH, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    df.dropna(inplace=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEAN_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    download()
