# 🛒 Customer Segmentation & Marketing Analytics

> A production-ready Machine Learning project that segments e-commerce customers using **RFM Analysis + KMeans Clustering** and surfaces actionable marketing insights through an interactive **Streamlit dashboard**.

---

## 📌 Project Overview

This project analyzes transactional data from an online retail store and clusters customers into meaningful segments (Premium, High Value, Loyal, Regular, Potential, At Risk) to enable targeted marketing strategies.

| Item | Detail |
|------|--------|
| **Dataset** | UCI Online Retail Dataset (541K transactions, 4,372 customers) |
| **Period** | Dec 2010 – Dec 2011 |
| **Algorithm** | KMeans Clustering |
| **Features** | RFM + 9 behavioral features |
| **Dashboard** | Streamlit + Plotly |

---

## 🎯 Problem Statement

A UK-based online retailer needs to move from one-size-fits-all marketing to personalized, segment-level campaigns. Without knowing who their most valuable customers are (or who is at risk of churning), every marketing pound is spent inefficiently.

**Goal:** Automatically segment customers and generate data-driven marketing strategies for each segment.

---

## 📂 Project Structure

```
customer_segmentation/
├── data/
│   └── Online_Retail.xlsx          # Raw dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Load, clean, validate
│   ├── feature_engineering.py      # RFM + behavioral features
│   ├── clustering.py               # KMeans, elbow, silhouette
│   ├── visualization.py            # EDA + cluster charts
│   └── business_insights.py        # Marketing recommendations
├── models/                         # Saved artifacts (after training)
│   ├── kmeans_model.joblib
│   ├── scaler.joblib
│   ├── feature_cols.joblib
│   ├── cluster_summary.csv
│   ├── insights.json
│   └── elbow_results.json
├── outputs/                        # Charts + processed data (after training)
│   ├── segmented_customers.csv
│   ├── clean_transactions.csv
│   ├── revenue_opportunity.csv
│   └── *.html  (interactive charts)
├── train.py                        # Master training script
├── dashboard.py                    # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧹 Data Cleaning Process

| Step | Action | Rows Removed |
|------|--------|-------------|
| Remove null CustomerID | Can't segment anonymous users | ~135K |
| Remove cancellations | InvoiceNo starts with 'C' | ~9K |
| Remove invalid qty/price | Quantity ≤ 0 or UnitPrice ≤ 0 | ~2K |
| Remove duplicates | Exact row duplicates | ~5K |
| **Result** | Clean dataset ready for analysis | ~392K rows |

---

## 🔬 Exploratory Data Analysis

EDA covers:
- **Revenue Trend** — monthly revenue line chart
- **Geographic Analysis** — top countries by revenue
- **Purchase Timing** — day-of-week × hour heatmap
- **Top Products** — bestsellers by revenue
- **RFM Distributions** — histograms for Recency, Frequency, Monetary

---

## ⚙️ Feature Engineering

Starting from raw transactions, the pipeline builds 12 customer-level features:

| Feature | Description |
|---------|-------------|
| **Recency** | Days since last purchase |
| **Frequency** | Unique invoices count |
| **Monetary** | Total revenue |
| **AvgOrderValue** | Mean revenue per invoice |
| **AvgItemsPerOrder** | Mean items per invoice |
| **UniqueProducts** | Distinct SKUs bought |
| **ActiveDays** | Distinct purchase days |
| **PurchaseSpan** | Days between first & last purchase |
| **AvgDaysBetweenOrders** | Purchase cadence |
| **CustomerValueScore** | Composite score (Monetary × 0.5 + Frequency × 10 − Recency × 0.1) |
| **LoyaltyScore** | ActiveDays / (Recency + 1) |
| **SpendIntensity** | Monetary / (PurchaseSpan + 1) |

All features are scaled with `StandardScaler` and clipped at the 99th percentile.

---

## 🤖 Clustering Methodology

1. **Feature Selection** — 12 RFM + behavioral features
2. **Scaling** — StandardScaler
3. **Elbow Method** — inertia for k = 2 to 9
4. **Silhouette Score** — evaluates cluster separation
5. **Optimal k Selection** — reconciles elbow + silhouette, clamped to [4–6] for business usability
6. **KMeans Training** — n_init=20, max_iter=500, random_state=42
7. **Segment Naming** — data-driven ranking by Monetary/Recency/Frequency

---

## 📊 Customer Segments

| Segment | Description |
|---------|-------------|
| 💎 **Premium** | Low recency, highest spend — VIP customers |
| 🏆 **High Value** | High spenders, not as frequent |
| 🔄 **Loyal** | Most frequent buyers |
| 📦 **Regular** | Average on all dimensions |
| 🌱 **Potential** | New or infrequent, moderate spend |
| ⚠️ **At Risk** | Long since last purchase — about to churn |

---

## 💡 Business Insights

Each segment receives 5 tailored marketing strategies. Examples:

- **Premium** → VIP loyalty program, concierge service, brand ambassador
- **At Risk** → Win-back campaigns, re-engagement surveys, limited-time offers
- **Loyal** → Points program, surprise discounts, referral bonuses
- **Potential** → Welcome series, first-repeat-purchase discount, retargeting

---

## 📱 Dashboard Overview

The Streamlit dashboard has 5 sections:

| Section | Content |
|---------|---------|
| 📊 Overview | KPI cards, scatter map, segment donut, revenue bar |
| 🧩 Segments | Metrics table, bar comparisons, elbow plots, 3D RFM |
| 📈 EDA | Revenue trend, country map, heatmap, distributions |
| 💡 Business Insights | Opportunity table, radar chart, strategy cards |
| 🔍 Customer Lookup | Individual customer profile + recommendations |

---

## 🔮 Future Improvements

- Add DBSCAN / Hierarchical clustering for comparison
- Integrate demographic / web clickstream data
- Build a real-time scoring API (FastAPI) for new customers
- Deploy to Streamlit Cloud / AWS
- Add email campaign trigger automation
- Track segment migration over time (cohort analysis)

---

## 🚀 Quick Start

```bash
# 1. Clone / set up
cd customer_segmentation
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train
python train.py

# 3. Dashboard
streamlit run dashboard.py
```

See **SETUP_GUIDE.md** for detailed instructions.
"# Customer-Segmentation"  git init git add . git commit -m "first commit" git branch -M main git remote add origin https://github.com/Raushanritik30891/Customer-Segmentation.git git push -u origin main
