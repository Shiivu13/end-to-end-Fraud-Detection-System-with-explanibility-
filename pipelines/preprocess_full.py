# pipelines/preprocess_full.py
# Small helper: create a full processed.csv from raw (same features as preview)

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "creditcard.csv"
OUT = ROOT / "data" / "processed" / "processed.csv"

def safe_load(path):
    if not path.exists():
        raise FileNotFoundError(f"Put creditcard.csv in {path.parent}")
    return pd.read_csv(path)

def make_features(df):
    df = df.sort_values("Time").reset_index(drop=True)
    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["is_night"] = df["hour"].isin(range(0,6)).astype(int)
    df["time_diff"] = df["Time"].diff().fillna(0)
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)
    times = df["Time"].to_numpy()
    left_idx = np.searchsorted(times, times - 60, side="left")
    df["count_1min"] = (np.arange(len(times)) - left_idx + 1).astype(int)
    df["rolling_log_amount_5"] = df["log_amount"].rolling(window=5, min_periods=1).mean()
    vcols = [c for c in df.columns if c.startswith("V")]
    keep = vcols + ["Time", "hour", "is_night", "time_diff",
                    "Amount", "log_amount", "amount_zscore",
                    "count_1min", "rolling_log_amount_5", "Class"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]

def main():
    df = safe_load(RAW)
    processed = make_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUT, index=False)
    print(f"Wrote full processed file: {OUT}  (rows={len(processed)})")
    print("Class distribution:")
    print(processed['Class'].value_counts())

if __name__ == "__main__":
    main()