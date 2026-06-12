import argparse
import os
import sys
import joblib
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_data, clean_data, get_data_summary
from src.feature_engineering import build_customer_features, get_feature_matrix
from src.clustering import (
    elbow_analysis, find_optimal_k, train_kmeans,
    build_cluster_profiles, plot_elbow,
)
from src.visualization import generate_all_eda_charts
from src.business_insights import generate_insights, print_insights, build_revenue_opportunity


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Segmentation Training Pipeline")
    parser.add_argument("--data", default="data/Online_Retail.xlsx", help="Path to dataset")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--models", default="models", help="Model save directory")
    parser.add_argument("--k", type=int, default=None, help="Override number of clusters")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    print("\n" + "="*60)
    print("  CUSTOMER SEGMENTATION — TRAINING PIPELINE")
    print("="*60)

    # ── 1. Load & clean ──────────────────────────────────────────
    df_raw = load_data(args.data)
    df = clean_data(df_raw)
    summary = get_data_summary(df)

    print("\n[DATASET SUMMARY]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: £{v:,.2f}")
        else:
            print(f"  {k}: {v}")

    # Save clean transactions
    df.to_csv(f"{args.output}/clean_transactions.csv", index=False)

    # ── 2. Feature engineering ───────────────────────────────────
    features = build_customer_features(df)
    X_scaled, scaler, feat_cols, features = get_feature_matrix(features)

    # ── 3. Elbow analysis ────────────────────────────────────────
    elbow_results = elbow_analysis(X_scaled, k_range=range(2, 10))
    plot_elbow(elbow_results, optimal_k=args.k or find_optimal_k(elbow_results),
               save_path=f"{args.output}/elbow_plot.png")

    # ── 4. Train final model ─────────────────────────────────────
    optimal_k = args.k if args.k else find_optimal_k(elbow_results)
    model = train_kmeans(X_scaled, optimal_k)
    labels = model.labels_

    # ── 5. Cluster profiling ─────────────────────────────────────
    features, cluster_stats = build_cluster_profiles(features, labels)

    print("\n[CLUSTER PROFILES]")
    display_cols = ["SegmentName", "CustomerCount", "Recency_mean", "Frequency_mean", "Monetary_mean"]
    display_cols = [c for c in display_cols if c in cluster_stats.columns]
    print(cluster_stats[display_cols].to_string(index=False))

    # ── 6. Business insights ─────────────────────────────────────
    insights = generate_insights(cluster_stats)
    print_insights(insights)

    opp_table = build_revenue_opportunity(insights)
    print("\n[REVENUE OPPORTUNITY TABLE]")
    print(opp_table.to_string(index=False))
    opp_table.to_csv(f"{args.output}/revenue_opportunity.csv", index=False)

    # ── 7. Visualizations ────────────────────────────────────────
    generate_all_eda_charts(df, features, cluster_stats, output_dir=args.output)

    # ── 8. Save artifacts ────────────────────────────────────────
    joblib.dump(scaler, f"{args.models}/scaler.joblib")
    joblib.dump(model, f"{args.models}/kmeans_model.joblib")
    joblib.dump(feat_cols, f"{args.models}/feature_cols.joblib")

    cluster_stats.to_csv(f"{args.models}/cluster_summary.csv", index=False)
    features.to_csv(f"{args.output}/segmented_customers.csv", index=False)

    # Save insights as JSON for the dashboard
    insights_serializable = []
    for card in insights:
        card_clean = {k: (int(v) if isinstance(v, (int, )) else v) for k, v in card.items()}
        insights_serializable.append(card_clean)
    with open(f"{args.models}/insights.json", "w") as f:
        json.dump(insights_serializable, f, indent=2)

    # Save elbow results
    with open(f"{args.models}/elbow_results.json", "w") as f:
        json.dump(elbow_results, f, indent=2)

    print(f"\n[INFO] ✅ Training complete!")
    print(f"  Models saved to: {args.models}/")
    print(f"  Outputs saved to: {args.output}/")
    print(f"\n  Launch dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
