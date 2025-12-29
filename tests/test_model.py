import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from model import train_model
from predictor import predict

def test_prediction_output():
    model, scaler = train_model()
    # Sample input from first row
    sample_input = [5.1, 3.5, 1.4, 0.2] + [0]*26
    result = predict(sample_input)
    assert result in ["Malignant", "Benign"]
