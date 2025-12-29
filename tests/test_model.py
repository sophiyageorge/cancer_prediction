from src.model import train_model
from src.predictor import predict

def test_prediction_output():
    model, scaler = train_model()
    # Sample input from first row
    sample_input = [5.1, 3.5, 1.4, 0.2] + [0]*26
    result = predict(sample_input)
    assert result in ["Malignant", "Benign"]
