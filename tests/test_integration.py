"""
Integration tests for the Cancer Prediction API.
Tests both ML pipeline and API endpoints.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Make src folder importable
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from main import app  # Your FastAPI app

client = TestClient(app)

def test_api_prediction():
    """Test the /predict endpoint of the Cancer Prediction API."""
    sample_input = {"features": [0.1]*30}
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Malignant", "Benign"]
