import joblib
import numpy as np

model = joblib.load("models/freight_model.pkl")

sample = np.array([[100, 500, 2000, 5, 300]])

prediction = model.predict(sample)

print("Predicted Cost:", np.expm1(prediction[0]))