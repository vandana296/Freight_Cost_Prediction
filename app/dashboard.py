
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

import sys
import os

# Fix import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
#from src.feature_engineering import haversine

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



# ---- Title ----
st.title("🚚 Freight Cost Prediction Dashboard")
st.markdown("Interactive dashboard for Olist freight data")

# ---- Load Data ----
@st.cache_data
def load_data():
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")
    df = orders.merge(items, on="order_id").merge(products, on="product_id")
    
    # Feature engineering
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['volume'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    
    return df

df = load_data()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
min_weight = int(df['product_weight_g'].min())
max_weight = int(df['product_weight_g'].max())
weight_range = st.sidebar.slider("Product Weight (g)", min_weight, max_weight, (min_weight, max_weight))

min_price = float(df['price'].min())
max_price = float(df['price'].max())
price_range = st.sidebar.slider("Price", min_price, max_price, (min_price, max_price))

filtered_df = df[(df['product_weight_g'] >= weight_range[0]) & 
                 (df['product_weight_g'] <= weight_range[1]) &
                 (df['price'] >= price_range[0]) &
                 (df['price'] <= price_range[1])]

st.markdown(f"### Showing {filtered_df.shape[0]} orders after filter")

# ---- Interactive Plots ----
st.markdown("## 📊 Freight Cost Distribution")
fig1 = px.histogram(filtered_df, x="freight_value", nbins=50, title="Freight Cost Distribution")
st.plotly_chart(fig1, width = 'stretch')

st.markdown("## 📦 Top 10 Categories by Avg Freight Cost")
top_categories = filtered_df.groupby('product_category_name')['freight_value'].mean().sort_values().tail(10)
fig2 = px.bar(top_categories, x=top_categories.values, y=top_categories.index, orientation='h', title="Top 10 Categories by Avg Freight Cost")
st.plotly_chart(fig2, width = 'stretch')

st.markdown("## ⚡ Freight Cost vs Delivery Days")
fig3 = px.scatter(filtered_df, x="delivery_days", y="freight_value", trendline="ols", title="Freight Cost vs Delivery Days")
st.plotly_chart(fig3, width= 'stretch')

st.markdown("## 📏 Weight vs Freight Cost")
fig4 = px.scatter(filtered_df, x="product_weight_g", y="freight_value", color="delivery_days", title="Weight vs Freight Cost by Delivery Days")
st.plotly_chart(fig4, width='stretch')

# ---- Predict Freight Cost ----
st.markdown("## 🚀 Predict Freight Cost")

st.sidebar.header("Predictor Inputs")
price = st.sidebar.number_input("Price", float(df['price'].min()), float(df['price'].max()), float(df['price'].mean()))
weight = st.sidebar.number_input("Product Weight (g)", int(df['product_weight_g'].min()), int(df['product_weight_g'].max()), int(df['product_weight_g'].mean()))
volume = st.sidebar.number_input("Product Volume (cm³)", int(df['volume'].min()), int(df['volume'].max()), int(df['volume'].mean()))
delivery_days = st.sidebar.number_input("Delivery Days", 1, 30, 5)

if st.sidebar.button("Predict"):
    # Load model
    model = joblib.load("models/freight_model.pkl")
    features = np.array([[price, weight, volume, delivery_days]])
    pred = model.predict(features)[0]
    st.success(f"Predicted Freight Cost: ${pred:.2f}")