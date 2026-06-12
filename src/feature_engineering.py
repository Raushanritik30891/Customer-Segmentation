"""
Feature Engineering Module
----------------------------
Builds customer-level RFM (Recency, Frequency, Monetary) features
and additional behavioral metrics for clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute RFM features per customer.
    - Recency:   days since last purchase
    - Frequency: number of unique invoices
    - Monetary:  total revenue
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    print(f"[INFO] RFM snapshot date: {snapshot_date.date()}")

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()

    return rfm


def build_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional customer-level behavioral features:
    - AvgOrderValue     : average revenue per invoice
    - AvgItemsPerOrder  : average items per invoice
    - UniqueProducts    : distinct SKUs purchased
    - ActiveDays        : distinct purchase days
    - AvgDaysBetweenOrders : purchasing cadence
    - PurchaseSpan      : days between first and last purchase
    - ReturnRate        : proxy (0 here as returns removed)
    - TopCountry        : primary country of purchase
    """
    # Per-invoice aggregation
    invoice_stats = df.groupby(["CustomerID", "InvoiceNo"]).agg(
        invoice_revenue=("Revenue", "sum"),
        invoice_items=("Quantity", "sum"),
    ).reset_index()

    customer_invoice = invoice_stats.groupby("CustomerID").agg(
        AvgOrderValue=("invoice_revenue", "mean"),
        AvgItemsPerOrder=("invoice_items", "mean"),
    ).reset_index()

    # Product diversity
    product_diversity = df.groupby("CustomerID")["StockCode"].nunique().reset_index()
    product_diversity.columns = ["CustomerID", "UniqueProducts"]

    # Active days and purchase span
    date_stats = df.groupby("CustomerID")["InvoiceDate"].agg(
        first_purchase="min",
        last_purchase="max",
        active_days="nunique",
    ).reset_index()
    date_stats.columns = ["CustomerID", "FirstPurchase", "LastPurchase", "ActiveDays"]
    date_stats["PurchaseSpan"] = (date_stats["LastPurchase"] - date_stats["FirstPurchase"]).dt.days

    # Average days between orders (cadence)
    def avg_days_between(dates):
        dates_sorted = sorted(dates)
        if len(dates_sorted) < 2:
            return 0.0
        diffs = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
        return float(np.mean(diffs))

    cadence = df.groupby("CustomerID")["InvoiceDate"].apply(avg_days_between).reset_index()
    cadence.columns = ["CustomerID", "AvgDaysBetweenOrders"]

    # Top country
    top_country = df.groupby("CustomerID")["Country"].agg(lambda x: x.mode()[0]).reset_index()
    top_country.columns = ["CustomerID", "TopCountry"]

    # Merge everything
    features = customer_invoice \
        .merge(product_diversity, on="CustomerID") \
        .merge(date_stats[["CustomerID", "ActiveDays", "PurchaseSpan"]], on="CustomerID") \
        .merge(cadence, on="CustomerID") \
        .merge(top_country, on="CustomerID")

    return features


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature builder: combines RFM + extra features.
    Returns a customer-level dataframe.
    """
    print("[INFO] Building RFM features...")
    rfm = build_rfm(df)

    print("[INFO] Building extra behavioral features...")
    extras = build_extra_features(df)

    features = rfm.merge(extras, on="CustomerID", how="left")

    # Derived scores
    features["CustomerValueScore"] = (
        features["Monetary"] * 0.5
        + features["Frequency"] * 10
        - features["Recency"] * 0.1
    )
    features["LoyaltyScore"] = features["ActiveDays"] / (features["Recency"] + 1)
    features["SpendIntensity"] = features["Monetary"] / (features["PurchaseSpan"] + 1)

    print(f"[INFO] Customer feature matrix shape: {features.shape}")
    return features


def get_feature_matrix(features: pd.DataFrame) -> tuple:
    """
    Select numeric features for clustering, apply StandardScaler.
    Returns (scaled_array, scaler, feature_cols, features_df_with_id).
    """
    clustering_cols = [
        "Recency", "Frequency", "Monetary",
        "AvgOrderValue", "AvgItemsPerOrder", "UniqueProducts",
        "ActiveDays", "PurchaseSpan", "AvgDaysBetweenOrders",
        "CustomerValueScore", "LoyaltyScore", "SpendIntensity",
    ]

    # Keep only columns that exist
    clustering_cols = [c for c in clustering_cols if c in features.columns]
    print(f"[INFO] Clustering on features: {clustering_cols}")

    X = features[clustering_cols].copy()

    # Clip extreme outliers at 99th percentile
    for col in X.columns:
        cap = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=cap)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, clustering_cols, features


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_data, clean_data

    df = clean_data(load_data("data/Online_Retail.xlsx"))
    features = build_customer_features(df)
    print(features.describe())
    features.to_csv("data/customer_features.csv", index=False)
    print("[INFO] Saved customer_features.csv")
