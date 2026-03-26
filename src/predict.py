import joblib
import numpy as np
import os

# Load model (correct path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "freight_model.pkl")

model = joblib.load(model_path)


def predict_cost(features):
    """
    Predict freight cost from input features
    """

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    # Reverse log transform (IMPORTANT)
    return float(np.expm1(prediction[0]))