# 🛠️ Setup & Installation Guide

Complete step-by-step guide to run the Customer Segmentation project from scratch.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 or higher |
| pip | latest |
| RAM | 4 GB minimum (8 GB recommended) |
| Disk | ~500 MB free |

---

## Step 1 — Get the Project

If cloning from Git:
```bash
git clone <your-repo-url>
cd customer_segmentation
```

Or if you downloaded the ZIP:
```bash
unzip customer_segmentation.zip
cd customer_segmentation
```

---

## Step 2 — Place the Dataset

Make sure the raw dataset is at:
```
customer_segmentation/
└── data/
    └── Online_Retail.xlsx      ← required
```

If the file has a different name, update the path in `train.py`:
```python
parser.add_argument("--data", default="data/Online_Retail.xlsx", ...)
```

---

## Step 3 — Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

---

## Step 4 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — KMeans clustering, StandardScaler
- `matplotlib`, `seaborn` — static charts
- `plotly` — interactive charts
- `streamlit` — dashboard
- `joblib` — model persistence
- `openpyxl` — xlsx reading

**Estimated install time:** 2–4 minutes.

---

## Step 5 — Train the Model

```bash
python train.py
```

### What this does:
1. Loads `data/Online_Retail.xlsx` (~540K transactions)
2. Cleans data (removes nulls, cancellations, invalid rows)
3. Builds RFM + 9 behavioral features per customer
4. Runs elbow + silhouette analysis (k = 2 to 9)
5. Trains final KMeans model
6. Generates cluster profiles & business insights
7. Creates interactive HTML charts
8. Saves all artifacts

### Expected output:
```
============================================================
  CUSTOMER SEGMENTATION — TRAINING PIPELINE
============================================================
[INFO] Loading dataset from: data/Online_Retail.xlsx
[INFO] Raw shape: (541909, 8)
[INFO] Starting data cleaning...
  After dropping null CustomerID: 406829 rows ...
  ...
[INFO] Running elbow + silhouette analysis...
  k=2  inertia=... silhouette=...
  ...
[CLUSTER PROFILES]
  SegmentName          CustomerCount  Recency_mean  Monetary_mean
  💎 Premium Customers   ...
  ...
[INFO] ✅ Training complete!
  Models saved to: models/
  Outputs saved to: outputs/
```

### Optional arguments:
```bash
# Override cluster count manually
python train.py --k 5

# Use a different dataset path
python train.py --data /path/to/your/data.xlsx

# Change output directories
python train.py --output my_outputs --models my_models
```

**Training time:** ~3–5 minutes (depends on hardware).

---

## Step 6 — Launch Dashboard

```bash
streamlit run dashboard.py
```

The browser opens automatically at `http://localhost:8501`

If not, open it manually.

### Dashboard sections:
- **📊 Overview** — KPI cards + segment map
- **🧩 Segments** — detailed cluster profiles + 3D RFM
- **📈 EDA** — revenue trends, countries, heatmaps
- **💡 Business Insights** — strategies per segment
- **🔍 Customer Lookup** — individual customer analysis

---

## Step 7 — Expected Output Files

After training, you'll find:

```
models/
├── kmeans_model.joblib      ← trained KMeans
├── scaler.joblib            ← fitted StandardScaler
├── feature_cols.joblib      ← feature names list
├── cluster_summary.csv      ← per-cluster statistics
├── insights.json            ← marketing strategies JSON
└── elbow_results.json       ← elbow/silhouette data

outputs/
├── segmented_customers.csv  ← all customers with segment labels
├── clean_transactions.csv   ← cleaned transaction data
├── revenue_opportunity.csv  ← revenue uplift sizing
├── elbow_plot.png           ← elbow + silhouette plot
├── cluster_scatter.html     ← interactive scatter
├── segment_distribution.html
├── cluster_radar.html
├── rfm_distributions.html
├── revenue_trend.html
├── top_countries.html
└── correlation_heatmap.png
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Make sure venv is activated and `pip install -r requirements.txt` ran |
| `FileNotFoundError: Online_Retail.xlsx` | Check the file is at `data/Online_Retail.xlsx` |
| Dashboard shows "Model not found" | Run `python train.py` first |
| Streamlit port in use | `streamlit run dashboard.py --server.port 8502` |
| Slow training | Normal — the dataset has 540K rows. Takes 3–5 min |
| `xlrd` error on xlsx | The file uses `.xlsx` not `.xls` — openpyxl handles it |

---

## Re-running the Pipeline

To re-train from scratch:
```bash
# Clear old artifacts (optional)
rm -rf models/* outputs/*

# Re-train
python train.py
```

---

## Running Individual Modules

```bash
# Test preprocessing alone
python src/data_preprocessing.py

# Test feature engineering
python src/feature_engineering.py

# Test clustering
python src/clustering.py

# Test insights
python src/business_insights.py
```
