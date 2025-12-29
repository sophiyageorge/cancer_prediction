"""
Streamlit UI for Cancer Disease Prediction.

This application collects user input, sends it to the
FastAPI backend, and displays the prediction result.
"""

import streamlit as st
import requests

st.title("Cancer Disease Prediction")

# Get input from user
feature_values = []
for i in range(30):  # 30 features in the dataset
    value = st.number_input(f"Feature {i + 1}", value=0.0)
    feature_values.append(value)

if st.button("Predict"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": feature_values},
            timeout=5,  # seconds
        )
        response.raise_for_status()
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")
