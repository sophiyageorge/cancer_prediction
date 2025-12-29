from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import predict

app = FastAPI(title="Cancer Prediction API")

class PatientData(BaseModel):
    features: list

@app.post("/predict")
def make_prediction(data: PatientData):
    prediction = predict(data.features)
    return {"prediction": prediction}
