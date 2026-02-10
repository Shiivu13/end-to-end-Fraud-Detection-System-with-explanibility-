# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd
from pathlib import Path

# ------------------------
# App init
# ------------------------
app = FastAPI(title="Fraud Detection API")

# ------------------------
# Paths & constants
# ------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.pkl"

FRAUD_THRESHOLD = 0.001  # business-driven threshold

# ------------------------
# Request schema
# ------------------------
class TransactionRequest(BaseModel):
    features: Dict[str, float]

# ------------------------
# Load model once (important)
# ------------------------
model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = model.feature_name_

# ------------------------
# Routes
# ------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: TransactionRequest):
    features = request.features

    # Convert input to DataFrame
    X = pd.DataFrame([features])

    # Handle missing features
    missing_features = []
    for col in EXPECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0.0
            missing_features.append(col)

    # Reorder columns to match training
    X = X[EXPECTED_FEATURES]

    # Predict
    prob = model.predict_proba(X)[0, 1]
    prediction = int(prob >= FRAUD_THRESHOLD)

    return {
        "prediction": prediction,
        "probability": round(float(prob), 6),
        "threshold_used": FRAUD_THRESHOLD,
        "missing_features_filled": missing_features,
        "explanation": "Prediction based on learned transaction patterns."
    }