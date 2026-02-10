# pipelines/explain_transaction.py
# Explain a single transaction using ML + SHAP

import joblib
import pandas as pd
import shap
from pathlib import Path

# -------------------------
# Paths & configuration
# -------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.pkl"

# Business-driven fraud threshold (LOCKED)
FRAUD_THRESHOLD = 0.0010


# -------------------------
# Main explanation function
# -------------------------

def explain_transaction(X_row: pd.DataFrame):
    """
    Takes a single-row DataFrame (one transaction)
    and returns prediction + explanation.
    """

    # 1️⃣ Load trained model
    model = joblib.load(MODEL_PATH)

    # 2️⃣ Predict fraud probability
    # predict_proba returns: [prob_not_fraud, prob_fraud]
    prob = model.predict_proba(X_row)[0, 1]

    # 3️⃣ Apply business threshold
    prediction = int(prob >= FRAUD_THRESHOLD)

    # 4️⃣ SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)[0]

    shap_df = (
        pd.DataFrame({
            "feature": X_row.columns,
            "shap_value": shap_values
        })
        .sort_values(by="shap_value", ascending=False)
    )

    # 5️⃣ Human-readable explanation (top 3 features)
    top_features = shap_df.head(3)
    reasons = []

    for _, row in top_features.iterrows():
        reasons.append(f"{row['feature']} increased fraud risk")

    explanation_text = (
        "The transaction was flagged as potentially fraudulent because "
        + ", ".join(reasons)
        + "."
    )

    # 6️⃣ Final response
    result = {
        "prediction": prediction,
        "probability": round(float(prob), 6),
        "threshold_used": FRAUD_THRESHOLD,
        "explanation": explanation_text,
    }

    return result