
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.data_loader import load_data
from src.feature_engineering import engineer_features

print("🔥 Script started")
def train_model():

    # 🔷 1. Load Data
    print("Loading data...")
    df = load_data()

    # 🔷 2. Feature Engineering
    print("Engineering features...")
    df = engineer_features(df)

    # 🔷 3. Handle Outliers
    print("Removing outliers...")
    df = df[df['cost'] < df['cost'].quantile(0.99)]

    # 🔷 4. Split Features & Target
    X = df.drop('cost', axis=1)
    y = df['cost']

    # 🔥 5. Log Transform Target (IMPORTANT)
    y = np.log1p(y)

    # 🔷 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔷 7. Model (XGBoost)
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 🔷 8. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Reverse log transform
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # 🔷 9. Save Model
    print("Saving model...")
    joblib.dump(model, "models/freight_model.pkl")

    print("✅ Training complete!")


if __name__ == "__main__":
    train_model()