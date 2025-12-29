import streamlit as st
import requests

st.title("Cancer Disease Prediction")

# Get input from user
feature_values = []
for i in range(30):  # 30 features in the dataset
    val = st.number_input(f"Feature {i+1}", value=0.0)
    feature_values.append(val)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": feature_values}
    )
    result = response.json()
    st.success(f"Prediction: {result['prediction']}")
