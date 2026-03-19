# %% [markdown]
# # 05 — Churn Prediction Model
# Binary classification: predict which customers will churn (no repurchase within
# 90 days). Includes feature engineering, Random Forest, SHAP explainability,
# and revenue-at-risk calculation.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150})
sns.set_theme(style="whitegrid")

# %%
conn = sqlite3.connect(DB_PATH)

orders = pd.read_sql("""
    SELECT o.order_id, o.order_purchase_timestamp, o.order_delivered_customer_date,
           o.order_estimated_delivery_date, o.order_status,
           c.customer_unique_id AS customer_id
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
""", conn, parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                         "order_estimated_delivery_date"])

payments = pd.read_sql("""
    SELECT order_id, payment_type, payment_value
    FROM order_payments
""", conn)

items = pd.read_sql("""
    SELECT i.order_id, i.price, i.freight_value, i.product_id,
           COALESCE(t.product_category_name_english, p.product_category_name) AS category
    FROM order_items i
    JOIN products p ON i.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
""", conn)

reviews = pd.read_sql("SELECT order_id, review_score FROM order_reviews", conn)

conn.close()
print(f"Orders: {len(orders):,} | Customers: {orders['customer_id'].nunique():,}")

# %% [markdown]
# ## 1. Define Churn Label
# Churn = customer who has NOT made a purchase within 90 days of their last order.
# We use 90 days as the cutoff because Olist customers are predominantly one-time
# buyers; 90 days is a reasonable window for a marketplace.

# %%
max_date = orders["order_purchase_timestamp"].max()
print(f"Dataset end date: {max_date}")

# Compute last purchase date per customer
customer_last = orders.groupby("customer_id")["order_purchase_timestamp"].max().reset_index()
customer_last.columns = ["customer_id", "last_purchase"]

# Churn label: 1 if last purchase > 90 days ago, 0 otherwise
customer_last["days_since_last"] = (max_date - customer_last["last_purchase"]).dt.days
customer_last["churned"] = (customer_last["days_since_last"] > 90).astype(int)

churn_rate = customer_last["churned"].mean()
print(f"Churn rate: {churn_rate:.1%} ({customer_last['churned'].sum():,} churned / "
      f"{len(customer_last):,} total)")

# %% [markdown]
# ## 2. Feature Engineering

# %%
# --- Merge all data at order level ---
order_data = orders.copy()

# Payment per order
order_pay = payments.groupby("order_id").agg(
    total_payment=("payment_value", "sum"),
    payment_type_mode=("payment_type", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown")
).reset_index()
order_data = order_data.merge(order_pay, on="order_id", how="left")

# Items per order
order_items = items.groupby("order_id").agg(
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    num_items=("product_id", "count"),
    num_categories=("category", "nunique")
).reset_index()
order_data = order_data.merge(order_items, on="order_id", how="left")

# Review score per order
order_reviews = reviews.groupby("order_id")["review_score"].mean().reset_index()
order_data = order_data.merge(order_reviews, on="order_id", how="left")

# Delivery delay
order_data["delivery_delay_days"] = np.where(
    order_data["order_delivered_customer_date"].notna() &
    order_data["order_estimated_delivery_date"].notna(),
    (order_data["order_delivered_customer_date"] -
     order_data["order_estimated_delivery_date"]).dt.total_seconds() / 86400,
    0
)

# --- Aggregate to customer level ---
customer_first = orders.groupby("customer_id")["order_purchase_timestamp"].min().reset_index()
customer_first.columns = ["customer_id", "first_purchase"]

features = order_data.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    total_spend=("total_payment", "sum"),
    avg_order_value=("total_payment", "mean"),
    avg_review_score=("review_score", "mean"),
    avg_delivery_delay_days=("delivery_delay_days", "mean"),
    num_categories_purchased=("num_categories", "sum"),
    avg_freight_ratio=("total_freight", lambda x: (x / order_data.loc[x.index, "total_price"].replace(0, np.nan)).mean()),
).reset_index()

# Merge first/last purchase info
features = features.merge(customer_first, on="customer_id")
features = features.merge(customer_last, on="customer_id")

# Derived features
features["days_since_first_order"] = (max_date - features["first_purchase"]).dt.days

# Payment type mode per customer
def safe_mode(x):
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else "unknown"

pay_mode = order_data.groupby("customer_id")["payment_type_mode"].agg(safe_mode).reset_index()
pay_mode.columns = ["customer_id", "payment_type_mode"]
features = features.merge(pay_mode, on="customer_id", how="left")

# Days between orders (for repeat buyers)
def avg_days_between(group):
    dates = group.sort_values()
    if len(dates) < 2:
        return 0
    return dates.diff().dt.days.mean()

days_between = orders.groupby("customer_id")["order_purchase_timestamp"]\
    .apply(avg_days_between).reset_index()
days_between.columns = ["customer_id", "days_between_orders"]
features = features.merge(days_between, on="customer_id", how="left")

# Fill NaN
features["avg_review_score"] = features["avg_review_score"].fillna(features["avg_review_score"].median())
features["avg_freight_ratio"] = features["avg_freight_ratio"].fillna(0)
features["days_between_orders"] = features["days_between_orders"].fillna(0)

print(f"\n✓ Engineered {features.shape[1] - 2} features for {len(features):,} customers")
print("\nFeature Summary:")
feature_cols = [
    "days_since_first_order", "total_orders", "total_spend", "avg_order_value",
    "avg_review_score", "avg_delivery_delay_days", "num_categories_purchased",
    "avg_freight_ratio", "days_between_orders"
]
print(features[feature_cols].describe().round(2))

# %% [markdown]
# ## 3. Model Training

# %%
# Encode payment type
features["payment_credit_card"] = (features["payment_type_mode"] == "credit_card").astype(int)
features["payment_boleto"] = (features["payment_type_mode"] == "boleto").astype(int)

# Final feature set
X_cols = feature_cols + ["payment_credit_card", "payment_boleto"]
X = features[X_cols].values
y = features["churned"].values

print(f"\nClass distribution: Churned={y.sum():,} ({y.mean():.1%}), "
      f"Retained={len(y)-y.sum():,} ({1-y.mean():.1%})")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline: StandardScaler → RandomForestClassifier
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)
y_pred_proba = pipe.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## 4. Model Evaluation

# %%
print("\n" + "=" * 60)
print("CHURN PREDICTION MODEL RESULTS")
print("=" * 60)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"📊 ROC-AUC Score: {roc_auc:.4f}")

# --- ROC Curve ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax1.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax1.fill_between(fpr, tpr, alpha=0.1, color="#2E86AB")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve — Churn Prediction", fontweight="bold")
ax1.legend(fontsize=12)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "18_churn_model_evaluation.png")
plt.close()
print("✓ Saved ROC curve & confusion matrix")

# %% [markdown]
# ## 5. Feature Importance & SHAP

# %%
# --- Random Forest Feature Importance ---
importances = pipe.named_steps["model"].feature_importances_
feat_imp = pd.DataFrame({"feature": X_cols, "importance": importances})\
    .sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(feat_imp["feature"], feat_imp["importance"], color="#2E86AB")
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Churn Prediction — Feature Importance", fontsize=14, fontweight="bold")

# Highlight top 3
for bar in bars[-3:]:
    bar.set_color("#E74C3C")

plt.savefig(FIG_DIR / "19_feature_importance.png")
plt.close()
print("✓ Saved feature importance chart")

# --- SHAP Values ---
try:
    import shap

    # Use a sample for SHAP (faster)
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    # Scale features as the model expects
    X_sample_scaled = pipe.named_steps["scaler"].transform(X_sample)
    shap_values = explainer.shap_values(X_sample_scaled)

    # SHAP summary plot (for churned class)
    fig, ax = plt.subplots(figsize=(12, 7))
    shap.summary_plot(
        shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values[1],
        X_sample,
        feature_names=X_cols,
        show=False
    )
    plt.title("SHAP Feature Impact on Churn Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "20_shap_summary.png")
    plt.close()
    print("✓ Saved SHAP summary plot")

except Exception as e:
    print(f"⚠️  SHAP plot error (non-critical): {e}")

# %% [markdown]
# ## 6. Revenue at Risk

# %%
# Calculate CLV for each customer
features["predicted_churn"] = pipe.predict(
    pipe.named_steps["scaler"].transform(
        features[X_cols].values
    ) if False else features[X_cols].values
)

# Re-predict properly
features["predicted_churn"] = pipe.predict(features[X_cols].values)
features["churn_probability"] = pipe.predict_proba(features[X_cols].values)[:, 1]

# Revenue at risk = predicted churned customers × their total spend
at_risk = features[features["predicted_churn"] == 1]
retained = features[features["predicted_churn"] == 0]

revenue_at_risk = at_risk["total_spend"].sum()
avg_clv_at_risk = at_risk["total_spend"].mean()

print("\n" + "=" * 60)
print("REVENUE AT RISK ANALYSIS")
print("=" * 60)
print(f"  Predicted churned customers : {len(at_risk):,}")
print(f"  Predicted retained customers: {len(retained):,}")
print(f"  Avg CLV of at-risk customers: R${avg_clv_at_risk:,.2f}")
print(f"  💰 Total Revenue at Risk    : R${revenue_at_risk:,.2f}")
print(f"\n  If a reactivation campaign converts even 10% of at-risk customers:")
print(f"  → Recoverable revenue: R${revenue_at_risk * 0.10:,.2f}")

# High-value at-risk customers
high_value_at_risk = at_risk.nlargest(10, "total_spend")[
    ["customer_id", "total_spend", "total_orders", "avg_review_score",
     "days_since_last", "churn_probability"]
]
print(f"\n  Top 10 High-Value At-Risk Customers:")
print(high_value_at_risk.to_string(index=False))

# Save predictions
features[["customer_id", "churned", "predicted_churn", "churn_probability", "total_spend"]]\
    .to_csv(PROJECT_ROOT / "outputs" / "churn_predictions.csv", index=False)

print(f"\n✅ Churn prediction complete — ROC-AUC: {roc_auc:.4f}")
print(f"   Revenue at risk: R${revenue_at_risk:,.0f}")
