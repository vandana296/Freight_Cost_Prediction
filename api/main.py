from fastapi import FastAPI
from src.predict import predict_cost

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Freight API running"}

@app.post("/predict")
def predict(data: dict):
    features = list(data.values())
    result = predict_cost(features)
    return {"predicted_cost": result}