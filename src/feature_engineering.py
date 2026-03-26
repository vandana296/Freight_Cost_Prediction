import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Perform feature engineering on merged Olist dataset
    """

    df = df.copy()

    # 🔷 1. Rename target column
    df.rename(columns={'freight_value': 'cost'}, inplace=True)

    # 🔷 2. Product Volume (IMPORTANT FEATURE)
    df['product_volume'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )

    # 🔷 3. Convert date columns
    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp'], errors='coerce'
    )
    df['order_delivered_customer_date'] = pd.to_datetime(
        df['order_delivered_customer_date'], errors='coerce'
    )

    # 🔷 4. Delivery Time Feature
    df['delivery_days'] = (
        df['order_delivered_customer_date'] -
        df['order_purchase_timestamp']
    ).dt.days

    # 🔷 5. Distance Proxy (Zip Code Difference)
    df['distance_proxy'] = abs(
        df['customer_zip_code_prefix'] -
        df['seller_zip_code_prefix']
    )

    # 🔷 6. Handle Missing Values
    df = df.dropna(subset=[
        'product_weight_g',
        'product_volume',
        'delivery_days',
        'distance_proxy',
        'cost'
    ])

    # 🔷 7. Select Final Features
    df = df[[
        'price',
        'product_weight_g',
        'product_volume',
        'delivery_days',
        'distance_proxy',
        'cost'
    ]]

    # 🔷 8. Remove invalid values
    df = df[df['delivery_days'] >= 0]
    df = df[df['product_weight_g'] > 0]

    return df