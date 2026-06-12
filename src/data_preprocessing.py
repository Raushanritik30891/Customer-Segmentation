"""
Data Preprocessing Module
--------------------------
Handles loading, cleaning, and validating the Online Retail dataset.
Produces a clean transaction-level dataset ready for feature engineering.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Online Retail xlsx dataset."""
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_excel(filepath, engine="openpyxl")
    print(f"[INFO] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Drop missing CustomerIDs (cannot segment without identity)
    2. Remove cancelled invoices (InvoiceNo starting with 'C')
    3. Remove negative Quantity / UnitPrice rows
    4. Drop duplicate rows
    5. Fix data types
    """
    print("\n[INFO] Starting data cleaning...")

    original_count = len(df)

    # 1. Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    print(f"  After dropping null CustomerID: {len(df):,} rows (removed {original_count - len(df):,})")

    # 2. Remove cancellations (InvoiceNo starts with 'C')
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    before = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    print(f"  After removing cancellations: {len(df):,} rows (removed {before - len(df):,})")

    # 3. Remove invalid Quantity and UnitPrice
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"  After removing invalid qty/price: {len(df):,} rows (removed {before - len(df):,})")

    # 4. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  After dropping duplicates: {len(df):,} rows (removed {before - len(df):,})")

    # 5. Fix data types
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Description"] = df["Description"].fillna("Unknown").astype(str)

    # 6. Add revenue column
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"\n[INFO] Clean dataset shape: {df.shape}")
    print(f"[INFO] Unique customers: {df['CustomerID'].nunique():,}")
    print(f"[INFO] Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")

    return df.reset_index(drop=True)


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dictionary of the cleaned dataset."""
    return {
        "total_rows": len(df),
        "unique_customers": df["CustomerID"].nunique(),
        "unique_products": df["StockCode"].nunique(),
        "unique_invoices": df["InvoiceNo"].nunique(),
        "unique_countries": df["Country"].nunique(),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "total_revenue": df["Revenue"].sum(),
        "avg_order_value": df.groupby("InvoiceNo")["Revenue"].sum().mean(),
    }


if __name__ == "__main__":
    df_raw = load_data("data/Online_Retail.xlsx")
    df_clean = clean_data(df_raw)
    summary = get_data_summary(df_clean)
    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    df_clean.to_csv("data/clean_transactions.csv", index=False)
    print("\n[INFO] Saved clean_transactions.csv")
