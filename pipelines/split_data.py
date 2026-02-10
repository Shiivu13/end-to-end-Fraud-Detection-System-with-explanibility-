# pipelines/split_data.py
# Day 3 Practical - Step 1
# Read processed preview (or full processed file) and create stratified train/val/test splits.
# Saves CSVs to data/processed/train.csv, val.csv, test.csv

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
# If you have the full processed file, change this path to processed.csv
INPUT = ROOT / "data" / "processed" / "processed.csv"
OUT_DIR = ROOT / "data" / "processed"

def main():
    print("Running split_data.py ...")
    if not INPUT.exists():
        print(f"ERROR: processed input not found at {INPUT}")
        print("Please run pipelines/preprocess.py first (it creates processed_preview.csv) or update INPUT path.")
        return

    df = pd.read_csv(INPUT)
    if 'Class' not in df.columns:
        print("ERROR: target column 'Class' not found in processed file.")
        print("Columns found:", list(df.columns))
        return

    # Ensure we have at least some rows
    if len(df) < 10:
        print("ERROR: processed file too small:", len(df))
        return

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Stratified 70/15/15 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss.split(X, y))
    train = df.iloc[train_idx].reset_index(drop=True)
    temp = df.iloc[temp_idx].reset_index(drop=True)

    # split temp into val/test 50/50 -> each 15% of original
    X_temp = temp.drop(columns=['Class'])
    y_temp = temp['Class']
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    val = temp.iloc[val_idx].reset_index(drop=True)
    test = temp.iloc[test_idx].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.csv"
    val_path = OUT_DIR / "val.csv"
    test_path = OUT_DIR / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved splits -> train: {train_path} ({len(train)} rows), val: {val_path} ({len(val)} rows), test: {test_path} ({len(test)} rows)")
    print("\nClass distribution (train):")
    print(train['Class'].value_counts(), "\n")
    print("Class distribution (val):")
    print(val['Class'].value_counts(), "\n")
    print("Class distribution (test):")
    print(test['Class'].value_counts(), "\n")

if __name__ == "__main__":
    main()