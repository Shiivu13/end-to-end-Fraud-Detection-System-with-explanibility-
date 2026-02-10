import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------------
# Setup & Paths
# ------------------------
st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

# Determine the project root relative to this file
# frontend/app.py -> parent is frontend -> parent is project root
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.pkl"

FRAUD_THRESHOLD = 0.001  # business-driven threshold

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ------------------------
# UI
# ------------------------
st.title("Fraud Detection Demo")

if model is None:
    st.warning("Please ensure the model exists in the 'models' directory.")
else:
    EXPECTED_FEATURES = model.feature_name_

    st.subheader("Enter transaction features")
    num_features = 5
    input_data = {}

    col1, col2 = st.columns(2)
    
    with col1:
        for i in range(1, num_features + 1):
            input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    with col2:
        input_data["Amount"] = st.number_input("Amount", value=0.0)

    if st.button("Check Fraud"):
        # ------------------------
        # Prediction Logic
        # ------------------------
        try:
            # Convert input to DataFrame
            X = pd.DataFrame([input_data])

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

            # ------------------------
            # Display Results
            # ------------------------
            st.subheader("Result")
            
            if prediction == 1:
                st.error(f"**Prediction:** Fraud")
            else:
                st.success(f"**Prediction:** Not Fraud")
                
            st.write(f"**Probability:** {prob:.6f}")
            st.write(f"**Threshold used:** {FRAUD_THRESHOLD}")
            
            st.write("**Explanation:** Prediction based on learned transaction patterns.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    st.write("---")
    st.caption("This demo runs the fraud detection model directly in the app.")
