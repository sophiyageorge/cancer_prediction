"""
Integration tests for the cancer prediction pipeline.

This module tests the full training and prediction workflow.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# pylint: disable=import-error, wrong-import-position
from model import train_model
from predictor import predict


def test_full_pipeline():
    """
    Test the complete ML pipeline from training to prediction.
    """
    _, _ = train_model()
    sample_input = [0.1]*30
    result = predict(sample_input)
    assert result in ["Malignant", "Benign"]
