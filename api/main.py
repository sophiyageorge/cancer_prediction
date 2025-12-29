"""
Cancer Prediction API.

This module defines a FastAPI application that exposes
an endpoint for predicting cancer based on patient features.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import predict

app = FastAPI(title="Cancer Prediction API")

class PatientData(BaseModel):
    """
    Schema for patient input data.

    Attributes:
        features (list): List of numerical features used for prediction.
    """
    features: list

@app.post("/predict")
def make_prediction(data: PatientData):
    """
    Generate a cancer prediction based on input features.

    Args:
        data (PatientData): Patient feature data.

    Returns:
        dict: Prediction result.
    """
    prediction = predict(data.features)
    return {"prediction": prediction}
