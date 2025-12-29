"""
Unit tests for model training and prediction output.
"""
import sys
from pathlib import Path

# Add src directory to path for test imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# pylint: disable=import-error, wrong-import-position
from model import train_model
from predictor import predict

def test_prediction_output():
    """
    Test that model training completes and prediction returns valid output.
    """
    _, _ = train_model()
    # Sample input from first row
    sample_input = [5.1, 3.5, 1.4, 0.2] + [0]*26
    result = predict(sample_input)
    assert result in ["Malignant", "Benign"]
