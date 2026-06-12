"""
Visualization Module
---------------------
Generates EDA and cluster visualizations using Plotly, Matplotlib, Seaborn.
All functions save .png files to the outputs/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Style config ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Set2


# ── Transaction-level EDA ───────────────────────────────────────────────────

def plot_revenue_over_time(df: pd.DataFrame, save_path: str = None):
    monthly = df.set_index("InvoiceDate").resample("M")["Revenue"].sum().reset_index()
    monthly.columns = ["Month", "Revenue"]
    fig = px.line(
        monthly, x="Month", y="Revenue",
        title="Monthly Revenue Trend",
        labels={"Revenue": "Revenue (£)", "Month": "Month"},
        template=PLOTLY_TEMPLATE,
        markers=True,
    )
    fig.update_traces(line_color="#2563EB", marker_color="#1D4ED8")
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_top_countries(df: pd.DataFrame, top_n: int = 10, save_path: str = None):
    country_rev = df.groupby("Country")["Revenue"].sum().nlargest(top_n).reset_index()
    fig = px.bar(
        country_rev, x="Revenue", y="Country", orientation="h",
        title=f"Top {top_n} Countries by Revenue",
        labels={"Revenue": "Revenue (£)"},
        template=PLOTLY_TEMPLATE,
        color="Revenue",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_rfm_distributions(features: pd.DataFrame, save_path: str = None):
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Recency (days)", "Frequency (orders)", "Monetary (£)"])
    cols = ["Recency", "Frequency", "Monetary"]
    colors = ["#3B82F6", "#10B981", "#F59E0B"]
    for i, (col, color) in enumerate(zip(cols, colors), 1):
        fig.add_trace(go.Histogram(x=features[col], marker_color=color, name=col, nbinsx=40), row=1, col=i)
    fig.update_layout(title="RFM Feature Distributions", template=PLOTLY_TEMPLATE, showlegend=False, height=400)
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_correlation_heatmap(features: pd.DataFrame, save_path: str = None):
    numeric_cols = features.select_dtypes(include=[np.number]).drop(columns=["CustomerID", "Cluster"], errors="ignore")
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Cluster visualizations ──────────────────────────────────────────────────

def plot_cluster_scatter(features: pd.DataFrame, save_path: str = None):
    """Recency vs Monetary scatter, colored by segment."""
    fig = px.scatter(
        features,
        x="Recency", y="Monetary",
        color="SegmentName",
        title="Customer Segments: Recency vs. Monetary Value",
        labels={"Recency": "Recency (days)", "Monetary": "Total Revenue (£)"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
        hover_data=["Frequency", "AvgOrderValue"] if "AvgOrderValue" in features.columns else ["Frequency"],
        opacity=0.7,
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_segment_distribution(cluster_stats: pd.DataFrame, save_path: str = None):
    fig = px.pie(
        cluster_stats,
        names="SegmentName",
        values="CustomerCount",
        title="Customer Segment Distribution",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_cluster_radar(cluster_stats: pd.DataFrame, save_path: str = None):
    """Radar / spider chart comparing segments across metrics."""
    metrics = ["Recency_mean", "Frequency_mean", "Monetary_mean",
               "AvgOrderValue_mean", "ActiveDays_mean", "LoyaltyScore_mean"]
    metrics = [m for m in metrics if m in cluster_stats.columns]

    if len(metrics) < 3:
        return None

    # Normalize each metric 0-1
    stats = cluster_stats.copy()
    for m in metrics:
        rng = stats[m].max() - stats[m].min()
        if rng == 0:
            stats[m + "_norm"] = 0.5
        else:
            stats[m + "_norm"] = (stats[m] - stats[m].min()) / rng
    # Recency is inverse (lower = better)
    if "Recency_mean_norm" in stats.columns:
        stats["Recency_mean_norm"] = 1 - stats["Recency_mean_norm"]

    norm_cols = [m + "_norm" for m in metrics]
    labels = [m.replace("_mean", "").replace("_", " ") for m in metrics]

    fig = go.Figure()
    for _, row in stats.iterrows():
        values = [row[c] for c in norm_cols]
        values += values[:1]  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=row["SegmentName"],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Profiles (Normalized)",
        template=PLOTLY_TEMPLATE,
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_cluster_bar_comparison(cluster_stats: pd.DataFrame, save_path: str = None):
    metrics = {
        "Monetary_mean": "Avg Revenue (£)",
        "Frequency_mean": "Avg Frequency",
        "Recency_mean": "Avg Recency (days)",
    }
    metrics = {k: v for k, v in metrics.items() if k in cluster_stats.columns}

    fig = make_subplots(rows=1, cols=len(metrics),
                        subplot_titles=list(metrics.values()))
    for i, (col, label) in enumerate(metrics.items(), 1):
        fig.add_trace(
            go.Bar(
                x=cluster_stats["SegmentName"],
                y=cluster_stats[col],
                name=label,
                marker_color=COLOR_SEQ[i % len(COLOR_SEQ)],
                showlegend=False,
            ),
            row=1, col=i,
        )

    fig.update_layout(
        title="Cluster Comparison Across Key Metrics",
        template=PLOTLY_TEMPLATE,
        height=450,
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_frequency_distribution(features: pd.DataFrame, save_path: str = None):
    fig = px.histogram(
        features, x="Frequency", color="SegmentName",
        nbins=30, barmode="overlay",
        title="Purchase Frequency Distribution by Segment",
        labels={"Frequency": "Number of Orders"},
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
        opacity=0.75,
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def generate_all_eda_charts(df: pd.DataFrame, features: pd.DataFrame,
                             cluster_stats: pd.DataFrame, output_dir: str = "outputs"):
    """Run all visualizations and save to output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Generating EDA and cluster visualizations...")
    plot_revenue_over_time(df, f"{output_dir}/revenue_trend.html")
    plot_top_countries(df, save_path=f"{output_dir}/top_countries.html")
    plot_rfm_distributions(features, f"{output_dir}/rfm_distributions.html")
    plot_correlation_heatmap(features, f"{output_dir}/correlation_heatmap.png")
    plot_cluster_scatter(features, f"{output_dir}/cluster_scatter.html")
    plot_segment_distribution(cluster_stats, f"{output_dir}/segment_distribution.html")
    plot_cluster_radar(cluster_stats, f"{output_dir}/cluster_radar.html")
    plot_cluster_bar_comparison(cluster_stats, f"{output_dir}/cluster_bar_comparison.html")
    plot_frequency_distribution(features, f"{output_dir}/frequency_distribution.html")
    print(f"[INFO] Saved all charts to {output_dir}/")
