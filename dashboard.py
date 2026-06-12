"""
dashboard.py — Streamlit Customer Segmentation Dashboard
----------------------------------------------------------
Sections:
  0. Sidebar: controls & model status
  1. Overview KPI cards
  2. Segment distribution
  3. RFM scatter & cluster profiles
  4. Detailed EDA (revenue trend, top countries, distributions)
  5. Business Insights & Revenue Opportunity
  6. Customer lookup tool

Run:
  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
COLOR_SEQ = px.colors.qualitative.Set2
PLOTLY_TEMPLATE = "plotly_white"

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {
      background: #F8FAFC;
      border-radius: 12px;
      padding: 18px 22px;
      border-left: 4px solid #2563EB;
      margin-bottom: 8px;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; color: #1E40AF; }
  .kpi-label { font-size: 0.85rem; color: #64748B; margin-top: 2px; }
  .segment-card {
      background: #F0FDF4;
      border-radius: 10px;
      padding: 14px 18px;
      border-left: 4px solid #10B981;
      margin-bottom: 6px;
  }
  .insight-header { font-size: 1.1rem; font-weight: 600; color: #1F2937; }
  .insight-tagline { font-size: 0.9rem; color: #6B7280; margin-bottom: 8px; }
  hr { border: 0; border-top: 1px solid #E5E7EB; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_segmented_customers():
    path = f"{OUTPUTS_DIR}/segmented_customers.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_transactions():
    path = f"{OUTPUTS_DIR}/clean_transactions.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    return df


@st.cache_data
def load_cluster_stats():
    path = f"{MODELS_DIR}/cluster_summary.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_insights():
    path = f"{MODELS_DIR}/insights.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_elbow():
    path = f"{MODELS_DIR}/elbow_results.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.markdown("## 🛒 Customer Segmentation")
    st.markdown("**Online Retail Analytics Dashboard**")
    st.markdown("---")

    customers = load_segmented_customers()
    if customers is None:
        st.error("⚠️ No model artifacts found.\n\nRun first:\n```\npython train.py\n```")
    else:
        st.success(f"✅ Model loaded  |  {len(customers):,} customers")

    st.markdown("---")
    st.markdown("**Navigation**")
    section = st.radio(
        "Go to section",
        ["📊 Overview", "🧩 Segments", "📈 EDA", "💡 Business Insights", "🔍 Customer Lookup"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Filters**")
    if customers is not None and "SegmentName" in customers.columns:
        all_segments = sorted(customers["SegmentName"].unique())
        selected_segments = st.multiselect("Filter by Segment", all_segments, default=all_segments)
    else:
        selected_segments = []


# ── Guard ─────────────────────────────────────────────────────────────────────
if customers is None:
    st.title("🛒 Customer Segmentation Dashboard")
    st.warning("### Model not trained yet\n\nPlease run `python train.py` first, then refresh this page.")
    st.code("python train.py", language="bash")
    st.stop()

# Apply segment filter
cluster_stats = load_cluster_stats()
insights = load_insights()
transactions = load_transactions()

if selected_segments:
    filtered = customers[customers["SegmentName"].isin(selected_segments)]
else:
    filtered = customers.copy()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section == "📊 Overview":
    st.title("📊 Customer Analytics Overview")

    # KPI row
    total_customers = len(filtered)
    total_segments = filtered["SegmentName"].nunique() if "SegmentName" in filtered.columns else 0
    avg_revenue = filtered["Monetary"].mean() if "Monetary" in filtered.columns else 0
    avg_frequency = filtered["Frequency"].mean() if "Frequency" in filtered.columns else 0

    premium_count = len(filtered[filtered["SegmentName"].str.contains("Premium", na=False)])
    at_risk_count = len(filtered[filtered["SegmentName"].str.contains("At Risk", na=False)])
    total_revenue = filtered["Monetary"].sum() if "Monetary" in filtered.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "👥", f"{total_customers:,}", "Total Customers"),
        (c2, "🧩", f"{total_segments}", "Segments"),
        (c3, "£", f"{avg_revenue:,.0f}", "Avg Customer Revenue"),
        (c4, "🔁", f"{avg_frequency:.1f}", "Avg Order Frequency"),
        (c5, "💎", f"{premium_count:,}", "Premium Customers"),
        (c6, "⚠️", f"{at_risk_count:,}", "At Risk Customers"),
    ]
    for col, icon, value, label in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-value">{icon} {value}</div>
              <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Scatter: Recency vs Monetary
        if "Recency" in filtered.columns and "Monetary" in filtered.columns:
            fig = px.scatter(
                filtered.sample(min(3000, len(filtered)), random_state=42),
                x="Recency", y="Monetary",
                color="SegmentName",
                title="Customer Map: Recency vs. Revenue",
                labels={"Recency": "Recency (days since last purchase)", "Monetary": "Total Revenue (£)"},
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=COLOR_SEQ,
                opacity=0.65,
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Donut: segment distribution
        if cluster_stats is not None:
            seg_filtered = cluster_stats[cluster_stats["SegmentName"].isin(selected_segments)]
            fig = px.pie(
                seg_filtered, names="SegmentName", values="CustomerCount",
                title="Segment Distribution",
                template=PLOTLY_TEMPLATE, hole=0.45,
                color_discrete_sequence=COLOR_SEQ, height=420,
            )
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

    # Revenue by segment bar
    if cluster_stats is not None:
        seg_filtered = cluster_stats[cluster_stats["SegmentName"].isin(selected_segments)].copy()
        seg_filtered["TotalRevenue"] = seg_filtered["CustomerCount"] * seg_filtered.get("Monetary_mean", 0)
        fig = px.bar(
            seg_filtered.sort_values("TotalRevenue", ascending=False),
            x="SegmentName", y="TotalRevenue",
            title="Total Revenue by Customer Segment",
            labels={"TotalRevenue": "Total Revenue (£)", "SegmentName": "Segment"},
            color="SegmentName",
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE,
            text_auto=".3s",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🧩 Segments":
    st.title("🧩 Customer Segment Profiles")

    if cluster_stats is not None:
        seg_filtered = cluster_stats[cluster_stats["SegmentName"].isin(selected_segments)]

        # Metrics table
        display_cols = [
            "SegmentName", "CustomerCount",
            "Recency_mean", "Frequency_mean", "Monetary_mean",
        ]
        display_cols = [c for c in display_cols if c in seg_filtered.columns]
        renamed = seg_filtered[display_cols].copy()
        renamed.columns = [
            c.replace("_mean", "").replace("_", " ") for c in display_cols
        ]
        st.dataframe(
            renamed.style.format({
                "Recency mean": "{:.0f} days",
                "Frequency mean": "{:.1f}",
                "Monetary mean": "£{:,.0f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Bar: avg monetary by segment
            if "Monetary_mean" in seg_filtered.columns:
                fig = px.bar(
                    seg_filtered.sort_values("Monetary_mean", ascending=False),
                    x="SegmentName", y="Monetary_mean",
                    title="Average Revenue per Customer",
                    labels={"Monetary_mean": "Avg Revenue (£)", "SegmentName": "Segment"},
                    color="SegmentName", color_discrete_sequence=COLOR_SEQ,
                    template=PLOTLY_TEMPLATE, text_auto=".3s",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bar: avg recency
            if "Recency_mean" in seg_filtered.columns:
                fig = px.bar(
                    seg_filtered.sort_values("Recency_mean"),
                    x="SegmentName", y="Recency_mean",
                    title="Average Recency (lower = more recent)",
                    labels={"Recency_mean": "Recency (days)", "SegmentName": "Segment"},
                    color="SegmentName", color_discrete_sequence=COLOR_SEQ,
                    template=PLOTLY_TEMPLATE, text_auto=".0f",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Elbow plot
        elbow = load_elbow()
        if elbow:
            st.markdown("### Optimal Cluster Selection")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(x=elbow["k"], y=elbow["inertia"], markers=True,
                               title="Elbow Method — Inertia",
                               labels={"x": "k", "y": "Inertia"},
                               template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.line(x=elbow["k"], y=elbow["silhouette"], markers=True,
                               title="Silhouette Score vs k",
                               labels={"x": "k", "y": "Silhouette Score"},
                               template=PLOTLY_TEMPLATE)
                fig.update_traces(line_color="#10B981")
                st.plotly_chart(fig, use_container_width=True)

    # RFM 3D scatter
    if "Recency" in filtered.columns:
        st.markdown("### 3D RFM Space")
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        fig = px.scatter_3d(
            sample, x="Recency", y="Frequency", z="Monetary",
            color="SegmentName",
            title="3D RFM Cluster View",
            labels={"Recency": "Recency", "Frequency": "Frequency", "Monetary": "Monetary"},
            color_discrete_sequence=COLOR_SEQ,
            opacity=0.6, height=600,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    if transactions is not None:
        # Monthly revenue
        monthly = transactions.set_index("InvoiceDate").resample("ME")["Revenue"].sum().reset_index()
        monthly.columns = ["Month", "Revenue"]
        fig = px.line(
            monthly, x="Month", y="Revenue",
            title="Monthly Revenue Trend",
            labels={"Revenue": "Revenue (£)"},
            template=PLOTLY_TEMPLATE, markers=True,
        )
        fig.update_traces(line_color="#2563EB")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            country_rev = transactions.groupby("Country")["Revenue"].sum().nlargest(12).reset_index()
            fig = px.bar(
                country_rev, x="Revenue", y="Country", orientation="h",
                title="Top 12 Countries by Revenue",
                labels={"Revenue": "Revenue (£)"},
                template=PLOTLY_TEMPLATE,
                color="Revenue", color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Day of week heatmap
            transactions["DayOfWeek"] = transactions["InvoiceDate"].dt.day_name()
            transactions["Hour"] = transactions["InvoiceDate"].dt.hour
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = transactions.groupby(["DayOfWeek", "Hour"])["Revenue"].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index="DayOfWeek", columns="Hour", values="Revenue").reindex(dow_order)
            fig = px.imshow(
                heatmap_pivot,
                title="Revenue Heatmap: Day of Week × Hour",
                labels={"x": "Hour", "y": "Day", "color": "Revenue (£)"},
                color_continuous_scale="Blues",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top products
        top_products = transactions.groupby("Description")["Revenue"].sum().nlargest(15).reset_index()
        fig = px.bar(
            top_products, x="Revenue", y="Description", orientation="h",
            title="Top 15 Products by Revenue",
            template=PLOTLY_TEMPLATE,
            color="Revenue", color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # RFM distributions
    if "Recency" in filtered.columns:
        st.markdown("### RFM Feature Distributions")
        c1, c2, c3 = st.columns(3)
        for col_name, col_widget, color in [
            ("Recency", c1, "#3B82F6"),
            ("Frequency", c2, "#10B981"),
            ("Monetary", c3, "#F59E0B"),
        ]:
            if col_name in filtered.columns:
                with col_widget:
                    fig = px.histogram(
                        filtered, x=col_name, nbins=40,
                        title=f"{col_name} Distribution",
                        template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[color],
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "💡 Business Insights":
    st.title("💡 Business Insights & Marketing Recommendations")

    if not insights:
        st.warning("No insights found. Run `python train.py` first.")
    else:
        filtered_insights = [i for i in insights if i["segment"] in selected_segments]
        if not filtered_insights:
            filtered_insights = insights

        # Revenue opportunity table
        opp_path = f"{OUTPUTS_DIR}/revenue_opportunity.csv"
        if os.path.exists(opp_path):
            st.markdown("### 💰 Revenue Opportunity Sizing")
            opp = pd.read_csv(opp_path)
            opp_filtered = opp[opp["Segment"].isin(selected_segments)] if selected_segments else opp
            st.dataframe(opp_filtered.style.format({
                "Current Revenue (£)": "£{:,.0f}",
                "Potential Uplift (£)": "£{:,.0f}",
            }), use_container_width=True)

            total_uplift = opp_filtered["Potential Uplift (£)"].sum() if len(opp_filtered) > 0 else 0
            st.metric("🚀 Total Revenue Uplift Potential", f"£{total_uplift:,.0f}")

        st.markdown("---")
        st.markdown("### 🎯 Segment-Level Marketing Strategies")

        for card in filtered_insights:
            with st.expander(f"{card['segment']} — {card['customer_count']:,} customers", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Recency", f"{card['avg_recency_days']} days")
                col2.metric("Avg Orders", f"{card['avg_frequency']}")
                col3.metric("Avg Revenue", f"£{card['avg_monetary']:,.0f}")
                col4.metric("Share of Customers", f"{card['pct_customers']}%")

                st.markdown(f"*{card['tagline']}*")
                st.markdown("**Recommended Actions:**")
                for i, s in enumerate(card["strategies"], 1):
                    st.markdown(f"{i}. {s}")

        # Radar chart for segment comparison
        if cluster_stats is not None:
            st.markdown("### 🕸️ Segment Profile Comparison")
            metrics = ["Recency_mean", "Frequency_mean", "Monetary_mean",
                       "AvgOrderValue_mean", "ActiveDays_mean"]
            metrics = [m for m in metrics if m in cluster_stats.columns]

            if len(metrics) >= 3:
                stats = cluster_stats[cluster_stats["SegmentName"].isin(selected_segments)].copy()
                for m in metrics:
                    rng = stats[m].max() - stats[m].min()
                    stats[m + "_n"] = (stats[m] - stats[m].min()) / rng if rng else 0.5
                if "Recency_mean_n" in stats.columns:
                    stats["Recency_mean_n"] = 1 - stats["Recency_mean_n"]
                norm_cols = [m + "_n" for m in metrics]
                labels = [m.replace("_mean", "").replace("_", " ") for m in metrics]
                fig = go.Figure()
                for _, row in stats.iterrows():
                    vals = [row[c] for c in norm_cols]
                    vals += vals[:1]
                    fig.add_trace(go.Scatterpolar(
                        r=vals, theta=labels + [labels[0]],
                        fill="toself", name=row["SegmentName"],
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Segment Profiles (Normalized 0–1)",
                    template=PLOTLY_TEMPLATE, height=500,
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CUSTOMER LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔍 Customer Lookup":
    st.title("🔍 Customer Lookup")

    customer_id_input = st.text_input("Enter Customer ID", placeholder="e.g. 17850")

    if customer_id_input:
        try:
            cid = int(customer_id_input)
            row = customers[customers["CustomerID"] == cid]
            if len(row) == 0:
                st.warning(f"Customer {cid} not found in segmented data.")
            else:
                r = row.iloc[0]
                st.success(f"Found customer **{cid}**")
                seg = r.get("SegmentName", "Unknown")
                st.markdown(f"### {seg}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recency", f"{r.get('Recency', 0):.0f} days")
                c2.metric("Frequency", f"{r.get('Frequency', 0):.0f} orders")
                c3.metric("Total Revenue", f"£{r.get('Monetary', 0):,.2f}")
                c4.metric("Avg Order Value", f"£{r.get('AvgOrderValue', 0):,.2f}")

                st.markdown("---")
                st.markdown("**All customer features:**")
                display_row = row.select_dtypes(include=[np.number]).T
                display_row.columns = ["Value"]
                st.dataframe(display_row.style.format("{:.2f}"), use_container_width=True)

                # Matching insight
                matched = next((i for i in insights if i["segment"] == seg), None)
                if matched:
                    st.markdown("---")
                    st.markdown(f"### 🎯 Recommended Actions for this Customer")
                    st.markdown(f"*{matched['tagline']}*")
                    for i, s in enumerate(matched["strategies"], 1):
                        st.markdown(f"{i}. {s}")
        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")

    else:
        # Show random sample
        st.markdown("### Sample Customers")
        sample = customers.sample(min(20, len(customers)), random_state=42)
        cols_show = [c for c in ["CustomerID", "SegmentName", "Recency", "Frequency",
                                   "Monetary", "AvgOrderValue", "UniqueProducts"] if c in sample.columns]
        st.dataframe(
            sample[cols_show].style.format({
                "Monetary": "£{:,.2f}",
                "AvgOrderValue": "£{:,.2f}",
                "Recency": "{:.0f}",
                "Frequency": "{:.0f}",
            }),
            use_container_width=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9CA3AF; font-size:0.8rem'>"
    "Customer Segmentation Analytics Dashboard · Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
