# 🛒 E-commerce Customer Intelligence & Revenue Optimization — Olist

> End-to-end analytics project using 100K+ real orders from the Olist Brazilian marketplace. Covers data engineering, exploratory analysis, ML churn prediction, interactive dashboards, and business storytelling.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![SQLite](https://img.shields.io/badge/SQLite-Database-07405e?style=flat-square&logo=sqlite)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f89939?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly_Dash-Dashboard-3f4f75?style=flat-square&logo=plotly)

---

## 📋 Problem Statement

Olist's marketplace serves thousands of sellers and hundreds of thousands of customers across Brazil, but most customers buy only once. **This project identifies the drivers of churn, segments customers by value, and quantifies the revenue opportunity from targeted retention campaigns** — giving the business a clear playbook to recover R$12.3M in at-risk revenue.

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [Olist Brazilian E-Commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| **Size** | ~50 MB, 100K+ orders |
| **Tables** | 9 relational tables (orders, customers, items, payments, reviews, products, sellers, geolocation, category translation) |
| **Period** | 2016 – 2018 |

## 🛠 Tools Used

| Category | Tools |
|----------|-------|
| **Languages** | Python · SQL |
| **Database** | SQLite (portable, standard SQL compatible with PostgreSQL) |
| **Analysis** | pandas · NumPy · matplotlib · seaborn |
| **ML** | scikit-learn (Random Forest) · SHAP |
| **Visualization** | Plotly · Plotly Dash (interactive dashboard) |
| **Geo** | Plotly choropleth maps |

## 🔑 Key Findings

| Finding | Impact |
|---------|--------|
| Champions (7% of customers) generate **13.1% of revenue** | High-value segment worth protecting |
| At-Risk segment: **22K customers, R$3.7M lifetime value** | Largest recovery opportunity |
| Late deliveries cause **1.6-point drop** in review scores | 8.1% late rate → direct NPS impact |
| Churn model (ROC-AUC: 1.00) → **R$12.3M revenue at risk** | Data-driven retention targeting |
| São Paulo = 41% of orders but **lowest AOV (R$142)** | Upsell opportunity |

## 📁 Project Structure

```
E-commerce/
├── README.md
├── requirements.txt
├── schema.sql                          # DDL for all 9 tables
├── .gitignore
│
├── olist_data/                         # Raw CSVs (gitignored)
├── data/
│   └── olist.db                        # SQLite database
│
├── notebooks/
│   ├── 01_data_cleaning.py             # Load, clean, audit data quality
│   ├── 02_eda.py                       # Exploratory data analysis (8 charts)
│   ├── 03_rfm_segmentation.py          # RFM scoring & customer segments
│   ├── 04_cohort_retention.py          # Cohort retention heatmap
│   ├── 05_churn_prediction.py          # ML churn model + SHAP + revenue-at-risk
│   ├── 06_product_analysis.py          # Product & category deep-dive
│   └── 07_seller_geo.py               # Seller scorecard & geo choropleth
│
├── sql/
│   └── key_queries.sql                 # 4 showcase SQL queries
│
├── dashboard/
│   └── app.py                          # 3-page Plotly Dash dashboard
│
└── outputs/
    ├── figures/                        # All generated charts (PNG)
    ├── rfm_segments.csv                # RFM segment data
    ├── seller_scorecard.csv            # Seller performance data
    ├── churn_predictions.csv           # Churn model predictions
    └── business_memo.md                # 1-page business memo
```

---

## 📈 Analysis Screenshots

### Monthly Revenue Trend & Order Volume
![Monthly revenue trend with dual-axis showing revenue growth and order volume](outputs/figures/02_monthly_revenue_trend.png)

### RFM Customer Segmentation
![RFM segment distribution donut chart and revenue by segment bar chart](outputs/figures/09_rfm_segments.png)

### Cohort Retention Heatmap
![Cohort retention heatmap showing month-over-month customer retention rates](outputs/figures/11_cohort_retention_heatmap.png)

### CLV Forecast & Customer Health Score
![Health score distribution and CLV forecast charts](outputs/figures/21_health_score_clv.png)

### Churn Prediction Model — ROC Curve & Confusion Matrix
![ROC curve and confusion matrix for the churn prediction Random Forest model](outputs/figures/18_churn_model_evaluation.png)

### Feature Importance & SHAP Explainability
![Feature importance chart showing which factors drive customer churn](outputs/figures/19_feature_importance.png)
![SHAP summary plot for churn model explainability](outputs/figures/20_shap_summary.png)

### Revenue by Brazilian State (Choropleth Map)
![Choropleth map of Brazil showing revenue distribution by state](outputs/figures/16_revenue_choropleth.png)

### Delivery Time vs Review Score
![Scatter plot showing negative correlation between delivery time and review scores](outputs/figures/14_delivery_vs_review.png)

### Category Seasonality Heatmap
![Heatmap showing monthly revenue patterns across top product categories](outputs/figures/15_category_seasonality.png)

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/E-commerce-Customer-Intelligence.git
cd E-commerce-Customer-Intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (requires Kaggle API key)
kaggle datasets download -d olistbr/brazilian-ecommerce -p olist_data/
unzip olist_data/brazilian-ecommerce.zip -d olist_data/

# 4. Run the full pipeline
python notebooks/01_data_cleaning.py     # Load & clean data → SQLite
python notebooks/02_eda.py               # Exploratory analysis
python notebooks/03_rfm_segmentation.py  # RFM segmentation
python notebooks/04_cohort_retention.py  # Cohort retention
python notebooks/05_churn_prediction.py  # Churn model + SHAP
python notebooks/06_product_analysis.py  # Product analysis
python notebooks/07_seller_geo.py        # Seller & geo analysis
python notebooks/08_clv_health_score.py  # CLV forecast + health score



# Or launch Dash dashboard
python dashboard/app.py
# Open http://localhost:8050
```


## 📝 Business Memo

See the full [Business Memo](outputs/business_memo.md) with data-driven findings, recommendations, and next steps.

**Headline metrics:**
- 💰 R$12.3M revenue at risk from churned customers
- 🎯 22K At-Risk customers recoverable with 10% voucher campaign
- 📦 8.1% late delivery rate directly linked to 1.6-point review drop
- 🏆 Champions segment (7%) drives 13.1% of all revenue
- 🏥 Customer Health Score: avg 30.9/100 — only 3% in "Excellent" tier

## 🏗 Differentiation Features

| Layer | Feature | 
|-------|---------|
| Base | EDA + Dashboard + Churn Model | 
| Factor 1 | Revenue impact numbers + Business memo | 
| Factor 1 | What-if scenario simulator | 
| Factor 2 | SHAP explainability | 
| Factor 2 | CLV forecast (simple + BG/NBD attempt) | 
| Factor 2 | Customer health score (composite 0-100) | 
| Factor 3 | Interactive Power BI Dashboard | 
| Factor 3 | Customer lookup tool | 
| Factor 3 | Auto-refresh simulation | 
| Factor 4 | AI-powered insight narrative | 
| Factor 4 | Natural language query interface | 

## 📄 License

This project uses the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) which is provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

