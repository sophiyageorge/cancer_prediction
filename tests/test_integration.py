
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from model import train_model
from predictor import predict


def test_full_pipeline():
    model, scaler = train_model()
    sample_input = [0.1]*30
    result = predict(sample_input)
    assert result in ["Malignant", "Benign"]
