# 🛒 Customer Segmentation & Marketing Analytics

<p align="center">
  <b>Machine Learning-Powered Customer Intelligence Platform</b><br>
  Segment customers, identify high-value audiences, detect churn risks, and generate actionable marketing strategies using RFM Analysis and K-Means Clustering.
</p>

<p align="center">
  <a href="https://customer-segmentation-fcyebxabxnmzepkwmcmgdf.streamlit.app/">🚀 Live Demo</a> •
  <a href="https://github.com/Raushanritik30891/Customer-Segmentation">💻 Source Code</a>
</p>

---

## 📖 Overview

Customer Segmentation & Marketing Analytics is an end-to-end Machine Learning project that transforms raw e-commerce transaction data into actionable business intelligence.

Using customer purchase history, the system automatically identifies meaningful customer groups based on spending patterns, purchasing frequency, and engagement behavior. These insights help businesses improve customer retention, increase marketing ROI, and maximize revenue opportunities.

---

## 🎯 Business Problem

Most businesses use generic marketing campaigns for all customers, leading to:

* Low conversion rates
* Poor customer retention
* High marketing costs
* Missed upselling opportunities

This project solves the problem by automatically identifying valuable customer segments and recommending targeted marketing actions for each group.

---

## 🚀 Key Features

### Customer Segmentation

* RFM Analysis (Recency, Frequency, Monetary)
* Advanced Behavioral Feature Engineering
* K-Means Clustering
* Automatic Segment Naming

### Analytics & Insights

* Customer Value Scoring
* Loyalty Analysis
* Revenue Opportunity Estimation
* Churn Risk Identification
* Segment Performance Comparison

### Interactive Dashboard

* Executive KPI Dashboard
* Customer Segment Analysis
* Interactive Visualizations
* Business Recommendation Engine
* Customer Lookup Tool

### Automated Recommendations

* Premium Customer Retention Strategies
* High-Value Customer Upsell Plans
* Customer Nurturing Campaigns
* Win-Back Strategies for At-Risk Customers

---

## 📊 Dataset

**Online Retail Dataset**

| Attribute    | Value               |
| ------------ | ------------------- |
| Transactions | 541,909             |
| Customers    | 4,372               |
| Products     | 3,600+              |
| Countries    | 37                  |
| Time Period  | Dec 2010 – Dec 2011 |

After cleaning and preprocessing:

* Removed anonymous customers
* Removed cancellations
* Removed invalid transactions
* Removed duplicates

Final dataset contains approximately **392,000+ clean transactions**.

---

## 🔬 Machine Learning Pipeline

### 1. Data Preprocessing

* Missing Value Handling
* Duplicate Removal
* Transaction Validation
* Revenue Calculation

### 2. Feature Engineering

Generated customer-level features:

* Recency
* Frequency
* Monetary
* Average Order Value
* Average Items Per Order
* Unique Products Purchased
* Active Purchase Days
* Purchase Span
* Average Days Between Orders
* Customer Value Score
* Loyalty Score
* Spend Intensity

### 3. Clustering

Algorithm Used:

* K-Means Clustering

Optimization Techniques:

* Elbow Method
* Silhouette Analysis
* Feature Scaling using StandardScaler

### 4. Segment Generation

Customers are automatically categorized into:

| Segment                 | Description                           |
| ----------------------- | ------------------------------------- |
| 💎 Premium Customers    | Highest revenue and engagement        |
| 🏆 High Value Customers | Strong spenders with growth potential |
| 🔄 Loyal Customers      | Frequent repeat buyers                |
| 📦 Regular Customers    | Average purchasing behavior           |
| 🌱 Potential Customers  | Early-stage customers                 |
| ⚠️ At Risk Customers    | Inactive customers likely to churn    |

---

## 📈 Dashboard Features

### Executive Overview

* Revenue KPIs
* Customer KPIs
* Segment Distribution
* Revenue Contribution Analysis

### Customer Analytics

* RFM Analysis
* Segment Comparison
* Cluster Visualization
* Revenue Breakdown

### Business Intelligence

* Revenue Opportunity Analysis
* Customer Lifetime Insights
* Marketing Recommendations
* Strategic Growth Suggestions

### Customer Lookup

* Individual Customer Profile
* Segment Assignment
* Customer Metrics
* Personalized Recommendations

---

## 🛠️ Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* K-Means Clustering

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Dashboard

* Streamlit

### Model Persistence

* Joblib

---

## 📊 Business Impact

This solution enables organizations to:

✔ Identify VIP customers

✔ Detect churn risks early

✔ Improve customer retention

✔ Optimize marketing campaigns

✔ Increase conversion rates

✔ Improve customer lifetime value

✔ Generate actionable business insights

---

## 🚀 Live Demo

**Streamlit Application**

https://customer-segmentation-fcyebxabxnmzepkwmcmgdf.streamlit.app/

---

## ⚡ Installation

```bash
git clone https://github.com/Raushanritik30891/Customer-Segmentation.git

cd Customer-Segmentation

pip install -r requirements.txt

python train.py

streamlit run dashboard.py
```

---

## 📌 Future Enhancements

* DBSCAN Clustering
* Hierarchical Clustering
* Real-Time Customer Scoring API
* Customer Lifetime Value Prediction
* Cohort Analysis
* Marketing Campaign Automation
* Cloud Deployment (AWS/GCP)

---

## 👨‍💻 Author

**Ritik Raushan**

Aspiring Data Scientist & Machine Learning Engineer

* Machine Learning
* Data Analytics
* Business Intelligence
* Customer Analytics

⭐ If you found this project useful, consider giving it a star.
