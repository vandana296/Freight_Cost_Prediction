import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import predict_cost

st.set_page_config(page_title="Freight Cost Predictor")

st.title("🚚 Freight Cost Predictor")

price = st.number_input("Product Price")
weight = st.number_input("Weight (g)")
volume = st.number_input("Volume")
delivery_days = st.number_input("Delivery Days")
distance = st.number_input("Distance")

if st.button("Predict"):
    result = predict_cost([price, weight, volume, delivery_days, distance])
    st.success(f"Estimated Cost: ₹ {result:.2f}")