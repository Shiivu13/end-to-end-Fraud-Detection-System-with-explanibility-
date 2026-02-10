import streamlit as st
import pandas as pd
import joblib
import shap
from pathlib import Path

# ------------------------
# Setup & Paths
# ------------------------
st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightgbm_baseline.pkl"

FRAUD_THRESHOLD = 0.001

# ------------------------
# Feature Mapping (Human Readable)
# ------------------------
# Since V1-V28 are anonymized PCA features, we simulate meaningful names for the demo
FEATURE_DESC = {
    "V1": "User Login Behavior",
    "V2": "Location Consistency",
    "V3": "Device Integrity",
    "V4": "Transaction Velocity",
    "V5": "Network Security Score",
    "Amount": "Transaction Amount"
}

# ------------------------
# Helper Functions
# ------------------------
def explain_with_words(shap_df, top_n=3):
    """
    Convert top SHAP features into a human-readable explanation using friendly names.
    """
    top_features = shap_df.head(top_n)

    reasons = []
    for _, row in top_features.iterrows():
        fname = row["feature"]
        friendly_name = FEATURE_DESC.get(fname, fname) # Fallback to code if no desc
        sval = row["shap_value"]

        if sval > 0:
            reasons.append(f"**{friendly_name}** ({fname}) caused suspicion")
        else:
            reasons.append(f"**{friendly_name}** ({fname}) was normal")

    explanation = (
        "The model flagged this transaction as fraud mainly because "
        + ", ".join(reasons)
        + "."
    )
    return explanation

# ------------------------
# Load Resources
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

    st.subheader("Enter transaction details")
    # Using 5 features as per original demo
    num_features = 5
    input_data = {}

    col1, col2 = st.columns(2)
    
    with col1:
        for i in range(1, num_features + 1):
            feat_name = f"V{i}"
            desc = FEATURE_DESC.get(feat_name, feat_name)
            input_data[feat_name] = st.number_input(f"{desc} (V{i})", value=0.0)

    with col2:
        input_data["Amount"] = st.number_input("Transaction Amount", value=0.0)

    if st.button("Check Fraud Risk"):
        try:
            # Prepare Input
            X = pd.DataFrame([input_data])
            for col in EXPECTED_FEATURES:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[EXPECTED_FEATURES]

            # Predict
            prob = model.predict_proba(X)[0, 1]
            prediction = int(prob >= FRAUD_THRESHOLD)

            # Explain
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                feature_contribs = shap_values[1][0]
            else:
                feature_contribs = shap_values[0]

            shap_df = (
                pd.DataFrame({"feature": X.columns, "shap_value": feature_contribs})
                .sort_values(by="shap_value", ascending=False)
            )

            explanation_text = explain_with_words(shap_df)

            # Display
            st.subheader("Analysis Result")
            
            if prediction == 1:
                st.error(f"🚨 **FRAUD DETECTED**")
            else:
                st.success(f"✅ **Legitimate Transaction**")
                
            st.write(f"**Risk Score:** {prob:.2%}")
            st.info(f"💡 **Reasoning:** {explanation_text}")
            
            with st.expander("View Technical Details"):
                # Add friendly names to dataframe for display
                shap_df["description"] = shap_df["feature"].map(FEATURE_DESC)
                st.dataframe(shap_df[["description", "feature", "shap_value"]].style.background_gradient(cmap="coolwarm", subset=["shap_value"]))

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.write("---")
    st.caption("Note: 'V' features are anonymized banking parameters. Labels are simulated for this demo.")
