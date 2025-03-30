# streamlit_frontend.py (Streamlit Frontend)

import streamlit as st
import requests
import json

st.title("Credit Card Fraud Detection")

# Input fields for credit card transaction data (adjust as needed)
input_features = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Time"
]

input_values = []
for feature in input_features:
    value = st.number_input(f"Enter {feature}")
    input_values.append(value)

if st.button("Predict"):
    data = {"data": input_values}
    headers = {'Content-type': 'application/json'}
    try:
        response = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data), headers=headers)
        result = response.json()
        if "prediction" in result:
            prediction = result['prediction']
            if prediction == 1:
                st.error("Fraudulent Transaction Detected!")
            else:
                st.success("Legitimate Transaction.")
        elif "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.error("Unexpected response from backend.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")