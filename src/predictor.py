import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))

def predict(input_data: list):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return "Malignant" if prediction[0] == 0 else "Benign"

if __name__ == "__main__":
    sample_input = [0.1]*30
    print(predict(sample_input))
