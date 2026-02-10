import streamlit as st
import pandas as pd
import joblib
import shap
from pathlib import Path

# ------------------------
# Setup & Paths
# ------------------------
st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

# Determine the project root relative to this file
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.pkl"

FRAUD_THRESHOLD = 0.001  # business-driven threshold

# ------------------------
# Helper Functions
# ------------------------
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
            reasons.append(f"**{fname}** increased fraud risk")
        else:
            reasons.append(f"**{fname}** reduced fraud risk")

    explanation = (
        "The model flagged this transaction as fraud mainly because "
        + ", ".join(reasons)
        + "."
    )
    return explanation

# ------------------------
# Load Model & Explainer
# ------------------------
@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_resources()

# ------------------------
# UI
# ------------------------
st.title("Fraud Detection Demo")

if model is None:
    st.error(f"Model file not found at: {MODEL_PATH}")
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
            for col in EXPECTED_FEATURES:
                if col not in X.columns:
                    X[col] = 0.0

            # Reorder columns to match training
            X = X[EXPECTED_FEATURES]

            # Predict
            prob = model.predict_proba(X)[0, 1]
            prediction = int(prob >= FRAUD_THRESHOLD)

            # ------------------------
            # SHAP Explanation
            # ------------------------
            # shap_values returns a matrix (n_samples, n_features) or list of matrices
            shap_values = explainer.shap_values(X)
            
            # Handle different return types of shap_values (list vs array)
            if isinstance(shap_values, list):
                # Binary classification often returns list [class0_contribs, class1_contribs]
                # We care about class 1 (Fraud)
                feature_contribs = shap_values[1][0]
            else:
                feature_contribs = shap_values[0]

            shap_df = (
                pd.DataFrame(
                    {
                        "feature": X.columns,
                        "shap_value": feature_contribs,
                    }
                )
                .sort_values(by="shap_value", ascending=False)
            )

            explanation_text = explain_with_words(shap_df)

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
            
            st.markdown(f"**Explanation:** {explanation_text}")
            
            with st.expander("See detailed feature contributions"):
                st.dataframe(shap_df.style.background_gradient(cmap="coolwarm", subset=["shap_value"]))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    st.write("---")
    st.caption("This demo runs the fraud detection model directly in the app.")
