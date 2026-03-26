import pandas as pd

def load_data():
    orders = pd.read_csv(r"C:\Users\vinay\Freight_Cost_Prediction\data\olist_orders_dataset.csv")
    items = pd.read_csv(r"C:\Users\vinay\Freight_Cost_Prediction\data\olist_order_items_dataset.csv")
    products = pd.read_csv(r"C:\Users\vinay\Freight_Cost_Prediction\data\olist_products_dataset.csv")
    customers = pd.read_csv(r"C:\Users\vinay\Freight_Cost_Prediction\data\olist_customers_dataset.csv")
    sellers = pd.read_csv(r"C:\Users\vinay\Freight_Cost_Prediction\data\olist_sellers_dataset.csv")

    df = items.merge(orders, on='order_id')
    df = df.merge(products, on='product_id')
    df = df.merge(customers, on='customer_id')
    df = df.merge(sellers, on='seller_id')

    return df