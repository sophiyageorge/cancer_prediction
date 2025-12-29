"""
Prediction module.

This module loads the trained model and scaler
and provides a prediction interface for inference.
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model():
    """
    Load and cache the trained model and scaler.

    Returns:
        tuple: Loaded model and scaler objects.

    Raises:
        FileNotFoundError: If model or scaler files are missing.
    """
    if not hasattr(load_model, "model") or not hasattr(load_model, "scaler"):
        model_path = os.path.join(BASE_DIR, "model.joblib")
        scaler_path = os.path.join(BASE_DIR, "scaler.joblib")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Model or scaler not found. Train the model first."
            )

        load_model.model = joblib.load(model_path)
        load_model.scaler = joblib.load(scaler_path)

    return load_model.model, load_model.scaler


def predict(input_data: list) -> str:
    """
    Predict cancer type based on input features.

    Args:
        input_data (list): List of numerical feature values.

    Returns:
        str: Prediction result ('Malignant' or 'Benign').
    """
    model, scaler = load_model()

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    return "Malignant" if prediction[0] == 0 else "Benign"
