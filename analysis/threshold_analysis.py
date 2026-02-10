import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.pkl"
VAL_PATH = ROOT / "data" / "processed" / "val.csv"

print("Loading model and validation data...")

model = joblib.load(MODEL_PATH)
val = pd.read_csv(VAL_PATH)

X_val = val.drop(columns=["Class"])
y_val = val["Class"]

scores = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, scores)

print("Total thresholds:", len(thresholds))

# ---- Inspect some threshold points ----
for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
    print(
        f"Threshold={thresholds[i]:.4f} | "
        f"Precision={precision[i]:.4f} | "
        f"Recall={recall[i]:.4f}"
    )

# ---- threshold selection logic ----
target_recall = 0.25
candidates = []

for p, r, t in zip(precision, recall, thresholds):
    if r >= target_recall and t > 0:
        candidates.append((t, p, r))

if not candidates:
    print("\nNo threshold satisfies target recall.")
else:
    chosen_threshold, chosen_precision, chosen_recall = max(
        candidates, key=lambda x: x[1]
    )

    print("\nChosen Threshold (business-safe):")
    print(f"Threshold={chosen_threshold:.6f}")
    print(f"Precision={chosen_precision:.4f}")
    print(f"Recall={chosen_recall:.4f}")