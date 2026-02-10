import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

st.title("Fraud Detection Demo")

st.subheader("Enter transaction features")
num_features =5
input_data={}

for i in range(1, num_features+1):
    input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
input_data["Amount"] = st.number_input("Amount", value=0.0)

if st.button("Check Fraud"):
    try:
        payload={
             "features": input_data
        }
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result=response.json()

            st.subheader("Result")
            st.write(f"**Prediction:**{'Fraud' if result['prediction']==1 else 'Not Fraud'}")
            st.write(f"**Probability:** {result['probability']}")
            st.write(f"**Threshold used:**{result['threshold_used']}")
            st.write("**Explanation:** {result['explanation']}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
                st.error(f"Connection Error: {e}")

st.write("This demo allows you to test the fraud detection API")
st.info("Backend API must be running on http://localhost:8000")


