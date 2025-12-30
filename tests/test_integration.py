"""
Integration tests for the Cancer Prediction API.
Tests both ML pipeline and API endpoints.
"""

import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Make api folder importable
# sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))
# Add repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from api.main import app  # Your FastAPI app

client = TestClient(app)

def test_api_prediction():
    """Test the /predict endpoint of the Cancer Prediction API."""
    sample_input = {"features": [0.1]*30}
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Malignant", "Benign"]
