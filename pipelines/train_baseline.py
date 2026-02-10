# pipelines/train_baseline.py
# Train a baseline LightGBM fraud model with explainability

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
import lightgbm as lgb
import joblib
import shap

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VAL_PATH = ROOT / "data" / "processed" / "val.csv"
OUT_DIR = ROOT / "models"
MODEL_PATH = OUT_DIR / "lightgbm_baseline.pkl"


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def precision_at_k(y_true, y_scores, k):
    idx = np.argsort(y_scores)[::-1][:k]
    return y_true.values[idx].sum() / len(idx)


def pr_auc(y_true, y_scores):
    p, r, _ = precision_recall_curve(y_true, y_scores)
    return auc(r, p)


def explain_with_words(shap_df, top_n=3):
    """
    Convert top SHAP features into a human-readable explanation.
    """
    top_features = shap_df.head(top_n)

    reasons = []
    for _, row in top_features.iterrows():
        fname = row["feature"]
        sval = row["shap_value"]

        if sval > 0:
            reasons.append(f"{fname} increased fraud risk")
        else:
            reasons.append(f"{fname} reduced fraud risk")

    explanation = (
        "The model flagged this transaction as fraud mainly because "
        + ", ".join(reasons)
        + "."
    )
    return explanation


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------
def main():
    print("Loading data...")
    train = load_df(TRAIN_PATH)
    val = load_df(VAL_PATH)

    y_train = train["Class"]
    X_train = train.drop(columns=["Class"])
    y_val = val["Class"]
    X_val = val.drop(columns=["Class"])

    print("Training LightGBM baseline model...")

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=200,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30)],
    )

    print("Best iteration (trees used):", model.best_iteration_)

    # --------------------------------------------------
    # Validation predictions
    # --------------------------------------------------
    val_scores = model.predict_proba(X_val)[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    pr = pr_auc(y_val, val_scores)
    k = int(y_val.sum()) if int(y_val.sum()) > 0 else 100
    p_at_k = precision_at_k(y_val, val_scores, k)

    preds_05 = (val_scores >= 0.5).astype(int)
    prec_05 = precision_score(y_val, preds_05, zero_division=0)
    rec_05 = recall_score(y_val, preds_05)

    print(f"PR-AUC (val): {pr:.4f}")
    print(f"Precision@k (k={k}): {p_at_k:.4f}")
    print(f"Precision@0.5: {prec_05:.4f} | Recall@0.5: {rec_05:.4f}")

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "pr_auc": float(pr),
        "precision_at_k": float(p_at_k),
        "precision_at_0.5": float(prec_05),
        "recall_at_0.5": float(rec_05),
        "k": int(k),
    }

    metrics_path = OUT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {metrics_path}")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"Saved baseline model to: {MODEL_PATH}")

    # --------------------------------------------------
    # Feature importance plot
    # --------------------------------------------------
    importances = model.feature_importances_
    feature_names = X_train.columns

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (LightGBM)")
    plt.tight_layout()

    fi_path = OUT_DIR / "feature_importance.png"
    plt.savefig(fi_path)
    plt.close()

    print(f"Saved feature importance plot to: {fi_path}")

    # --------------------------------------------------
    # SHAP explanation (single example)
    # --------------------------------------------------
    explainer = shap.TreeExplainer(model)

    sample_X = X_val.iloc[[0]]
    shap_values = explainer.shap_values(sample_X)

    shap_df = (
        pd.DataFrame(
            {
                "feature": sample_X.columns,
                "shap_value": shap_values[0],
            }
        )
        .sort_values(by="shap_value", ascending=False)
    )

    print("\nTop features pushing prediction towards FRAUD:")
    print(shap_df.head(5))

    print("\nTop features pushing prediction towards NORMAL:")
    print(shap_df.tail(5))

    explanation = explain_with_words(shap_df)
    print("\nHuman-readable explanation:")
    print(explanation)


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()