import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_model = None
_scaler = None

def load_model():
    global _model, _scaler

    if _model is None or _scaler is None:
        model_path = os.path.join(BASE_DIR, "model.joblib")
        scaler_path = os.path.join(BASE_DIR, "scaler.joblib")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler not found. Train the model first.")

        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)

    return _model, _scaler


def predict(input_data: list):
    model, scaler = load_model()

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    return "Malignant" if prediction[0] == 0 else "Benign"
