# src/train.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

import shap


def load_data():
    print("Loading data...")

    orders = pd.read_csv("data/olist_orders_dataset.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")

    # Merge datasets
    df = orders.merge(items, on="order_id")
    df = df.merge(products, on="product_id")

    return df


def feature_engineering(df):
    print("Performing feature engineering...")

    # Convert datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

    # Delivery days
    df['delivery_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    # Volume
    df['volume'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )

    # Drop missing
    df = df.dropna(subset=['freight_value', 'delivery_days'])

    return df


def prepare_data(df):
    print("Preparing data...")

    # Features and target
    X = df[['price', 'product_weight_g', 'volume', 'delivery_days']]
    y = np.log1p(df['freight_value'])  # log transform

    return X, y


def train_model(X, y):
    print("Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return model, X_train


def save_artifacts(model, X_train):
    print("Saving model and SHAP explainer...")

    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/freight_model.pkl")

    # SHAP explainer (fast version)
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, "models/shap_explainer.pkl")

    print("✅ Model and SHAP explainer saved!")


def main():
    df = load_data()
    df = feature_engineering(df)
    X, y = prepare_data(df)
    model, X_train = train_model(X, y)
    save_artifacts(model, X_train)


if __name__ == "__main__":
    main()