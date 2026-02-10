# pipelines/preprocess.py
# Day 2 Practical - Step: sort by Time, add basic features, save processed preview

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
RAW = ROOT / "data" / "raw" / "creditcard.csv"
OUT = ROOT / "data" / "processed" / "processed_preview.csv"

def safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Put creditcard.csv in data/raw/")
    return pd.read_csv(path)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure sorted by Time so time-based features make sense
    df = df.sort_values("Time").reset_index(drop=True)

    # Hour of day (approx)
    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)

    # Is night flag (0/1)
    df["is_night"] = df["hour"].isin(range(0,6)).astype(int)

    # Time difference since previous transaction (seconds)
    df["time_diff"] = df["Time"].diff().fillna(0)

    # Log transform amount
    df["log_amount"] = np.log1p(df["Amount"])

    # Global amount z-score (useful baseline)
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)

    # Count of transactions in the last 60 seconds (global/session-like)
    times = df["Time"].to_numpy()
    left_idx = np.searchsorted(times, times - 60, side="left")
    df["count_1min"] = (np.arange(len(times)) - left_idx + 1).astype(int)

    # Rolling mean of log_amount over last 5 transactions
    df["rolling_log_amount_5"] = df["log_amount"].rolling(window=5, min_periods=1).mean()

    # Keep target and some columns; include V1..V28 if present
    vcols = [c for c in df.columns if c.startswith("V")]
    keep = vcols + ["Time", "hour", "is_night", "time_diff",
                    "Amount", "log_amount", "amount_zscore",
                    "count_1min", "rolling_log_amount_5", "Class"]
    # If some columns missing, intersect
    keep = [c for c in keep if c in df.columns]
    return df[keep]
def main():
    try:
        df = safe_load(RAW)
    except FileNotFoundError as e:
        print(str(e))
        return
    processed = make_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Save only a preview (first 1000 rows) so file is small for quick checks
    processed.head(1000).to_csv(OUT, index=False)
    print(f"Processed preview saved to: {OUT}")
    print("Preview shape:", processed.head(1000).shape)
    # Print simple summary
    print("\nColumns saved:", list(processed.columns))
    print("\nClass distribution (counts) on preview:")
    print(processed['Class'].value_counts())
    print("\nSample rows:")
    print(processed.head(6).to_string(index=False))

if __name__ == "__main__":
    main()
