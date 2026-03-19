# %% [markdown]
# # 08 — CLV Forecast & Customer Health Score
# Predict future customer lifetime value using BG/NBD model and build a
# composite 0–100 health score combining RFM + churn risk + CLV.

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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150})
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Prepare Customer Transaction Data

# %%
conn = sqlite3.connect(DB_PATH)

query = """
SELECT c.customer_unique_id AS customer_id,
       o.order_id,
       o.order_purchase_timestamp,
       p.payment_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
"""
df = pd.read_sql(query, conn, parse_dates=["order_purchase_timestamp"])
conn.close()

# Aggregate payments per order
orders = df.groupby(["customer_id", "order_id", "order_purchase_timestamp"])\
    ["payment_value"].sum().reset_index()

print(f"Customers: {orders['customer_id'].nunique():,} | Orders: {len(orders):,}")

# %% [markdown]
# ## 2. Build RFM-T Matrix for BG/NBD Model

# %%
max_date = orders["order_purchase_timestamp"].max()

# Per-customer summary statistics
customer_summary = orders.groupby("customer_id").agg(
    frequency=("order_id", lambda x: x.nunique() - 1),  # repeat purchases (BG/NBD convention)
    recency=("order_purchase_timestamp", lambda x: (x.max() - x.min()).days),
    T=("order_purchase_timestamp", lambda x: (max_date - x.min()).days),
    monetary_value=("payment_value", "mean"),
    total_spend=("payment_value", "sum"),
    first_purchase=("order_purchase_timestamp", "min"),
    last_purchase=("order_purchase_timestamp", "max"),
    order_count=("order_id", "nunique"),
).reset_index()

# Age in months
customer_summary["customer_age_months"] = customer_summary["T"] / 30.44

print(f"Repeat buyers: {(customer_summary['frequency'] > 0).sum():,} "
      f"({(customer_summary['frequency'] > 0).mean():.1%})")
print(f"One-time buyers: {(customer_summary['frequency'] == 0).sum():,}")

# %% [markdown]
# ## 3. BG/NBD Model — Predict Future Purchases

# %%
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    # Fit BG/NBD model (predicts purchase frequency)
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        customer_summary["frequency"],
        customer_summary["recency"],
        customer_summary["T"]
    )
    print(f"✓ BG/NBD model fitted")
    print(f"  Parameters: {bgf.params_}")

    # Predict purchases in next 90 days
    customer_summary["predicted_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90,
        customer_summary["frequency"],
        customer_summary["recency"],
        customer_summary["T"]
    )

    # Probability alive
    customer_summary["prob_alive"] = bgf.conditional_probability_alive(
        customer_summary["frequency"],
        customer_summary["recency"],
        customer_summary["T"]
    )

    print(f"  Avg predicted purchases (90d): {customer_summary['predicted_purchases_90d'].mean():.3f}")
    print(f"  Avg probability alive: {customer_summary['prob_alive'].mean():.2%}")

    # Fit Gamma-Gamma model for monetary value (only on repeat buyers)
    repeat_buyers = customer_summary[customer_summary["frequency"] > 0].copy()

    if len(repeat_buyers) > 100:
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])
        print(f"✓ Gamma-Gamma model fitted on {len(repeat_buyers):,} repeat buyers")

        # Predicted average profit per transaction
        repeat_buyers["predicted_avg_value"] = ggf.conditional_expected_average_profit(
            repeat_buyers["frequency"],
            repeat_buyers["monetary_value"]
        )

        # CLV for next 12 months (BG/NBD + Gamma-Gamma)
        repeat_buyers["predicted_clv_12m"] = ggf.customer_lifetime_value(
            bgf,
            repeat_buyers["frequency"],
            repeat_buyers["recency"],
            repeat_buyers["T"],
            repeat_buyers["monetary_value"],
            time=12,  # 12 months
            freq="D"
        )

        # Merge back
        customer_summary = customer_summary.merge(
            repeat_buyers[["customer_id", "predicted_avg_value", "predicted_clv_12m"]],
            on="customer_id", how="left"
        )
    else:
        customer_summary["predicted_avg_value"] = np.nan
        customer_summary["predicted_clv_12m"] = np.nan

    bgf_success = True

except Exception as e:
    print(f"⚠️  BG/NBD model error: {e}")
    print("  → Falling back to simple CLV calculation")
    bgf_success = False
    customer_summary["predicted_purchases_90d"] = np.nan
    customer_summary["prob_alive"] = np.nan
    customer_summary["predicted_avg_value"] = np.nan
    customer_summary["predicted_clv_12m"] = np.nan

# %% [markdown]
# ## 4. Simple CLV Forecast (always calculated as baseline)

# %%
# Simple CLV = avg_order_value × purchase_rate_per_month × 12
customer_summary["simple_clv_annual"] = np.where(
    customer_summary["customer_age_months"] > 0,
    (customer_summary["monetary_value"]) *
    (customer_summary["order_count"] / customer_summary["customer_age_months"]) * 12,
    customer_summary["monetary_value"]
)

# Use BG/NBD CLV if available, otherwise use simple CLV
customer_summary["clv_forecast"] = customer_summary["predicted_clv_12m"].fillna(
    customer_summary["simple_clv_annual"]
)

print(f"\n📊 CLV FORECAST SUMMARY:")
print(f"  Mean CLV (12-month): R${customer_summary['clv_forecast'].mean():,.2f}")
print(f"  Median CLV:          R${customer_summary['clv_forecast'].median():,.2f}")
print(f"  Max CLV:             R${customer_summary['clv_forecast'].max():,.2f}")
print(f"  Total predicted CLV: R${customer_summary['clv_forecast'].sum():,.0f}")

# %% [markdown]
# ## 5. Customer Health Score (Composite 0–100)

# %%
# Load RFM segments and churn predictions
rfm = pd.read_csv(OUTPUT_DIR / "rfm_segments.csv")
churn = pd.read_csv(OUTPUT_DIR / "churn_predictions.csv")

# Merge everything
health = customer_summary[["customer_id", "clv_forecast", "prob_alive",
                            "predicted_purchases_90d", "total_spend",
                            "order_count", "frequency"]].copy()
health = health.merge(
    rfm[["customer_id", "R_score", "F_score", "M_score", "Segment", "Recency", "Monetary"]],
    on="customer_id", how="left"
)
health = health.merge(
    churn[["customer_id", "churn_probability"]],
    on="customer_id", how="left"
)

# Fill missing churn probability with median
health["churn_probability"] = health["churn_probability"].fillna(
    health["churn_probability"].median()
)

# --- Normalise each component to 0–100 ---

# RFM component (higher = better, from scores 1-5)
health["R_score"] = pd.to_numeric(health["R_score"], errors="coerce").fillna(3)
health["F_score"] = pd.to_numeric(health["F_score"], errors="coerce").fillna(3)
health["M_score"] = pd.to_numeric(health["M_score"], errors="coerce").fillna(3)
health["rfm_norm"] = ((health["R_score"] + health["F_score"] + health["M_score"]) / 15 * 100).clip(0, 100)

# Churn risk component (invert: low churn probability = high health)
health["churn_risk_norm"] = ((1 - health["churn_probability"]) * 100).clip(0, 100)

# CLV component (normalise to 0-100)
clv_max = health["clv_forecast"].quantile(0.99)  # cap at 99th percentile to avoid outliers
health["clv_norm"] = ((health["clv_forecast"] / clv_max) * 100).clip(0, 100)

# --- Weighted composite health score ---
# RFM: 35%, Churn Risk: 40%, CLV: 25%
health["health_score"] = (
    0.35 * health["rfm_norm"] +
    0.40 * health["churn_risk_norm"] +
    0.25 * health["clv_norm"]
).round(1)

print(f"\n📊 CUSTOMER HEALTH SCORE SUMMARY:")
print(f"  Mean health score:   {health['health_score'].mean():.1f} / 100")
print(f"  Median health score: {health['health_score'].median():.1f} / 100")
print(f"  Std dev:             {health['health_score'].std():.1f}")
print(f"\n  Score distribution:")
bins = [0, 20, 40, 60, 80, 100]
labels = ["Critical (0-20)", "Poor (20-40)", "Fair (40-60)", "Good (60-80)", "Excellent (80-100)"]
health["health_tier"] = pd.cut(health["health_score"], bins=bins, labels=labels, include_lowest=True)
tier_dist = health["health_tier"].value_counts().sort_index()
for tier, count in tier_dist.items():
    print(f"    {tier}: {count:,} ({count/len(health)*100:.1f}%)")

# %% [markdown]
# ## 6. Visualizations

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Health score distribution
ax = axes[0, 0]
colors_map = {"Critical (0-20)": "#ef4444", "Poor (20-40)": "#f97316",
              "Fair (40-60)": "#eab308", "Good (60-80)": "#22c55e", "Excellent (80-100)": "#10b981"}
ax.hist(health["health_score"], bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
ax.axvline(health["health_score"].median(), color="red", linestyle="--",
           label=f"Median: {health['health_score'].median():.1f}")
ax.set_xlabel("Health Score (0-100)")
ax.set_ylabel("Count")
ax.set_title("Customer Health Score Distribution", fontweight="bold")
ax.legend()

# Health score by segment
ax = axes[0, 1]
seg_health = health.groupby("Segment")["health_score"].mean().sort_values()
seg_colors = {
    "Champions": "#10b981", "Loyal Customers": "#059669",
    "New Customers": "#3b82f6", "Need Attention": "#f59e0b",
    "At-Risk": "#ef4444", "Can't Lose Them": "#dc2626",
    "Hibernating": "#6b7280", "Others": "#9ca3af"
}
bar_colors = [seg_colors.get(s, "#999") for s in seg_health.index]
ax.barh(seg_health.index, seg_health.values, color=bar_colors)
ax.set_xlabel("Average Health Score")
ax.set_title("Health Score by Customer Segment", fontweight="bold")
for i, v in enumerate(seg_health.values):
    ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

# CLV distribution
ax = axes[1, 0]
clv_capped = health["clv_forecast"].clip(upper=health["clv_forecast"].quantile(0.95))
ax.hist(clv_capped, bins=50, color="#9B59B6", edgecolor="white", alpha=0.8)
ax.axvline(health["clv_forecast"].median(), color="red", linestyle="--",
           label=f"Median: R${health['clv_forecast'].median():,.0f}")
ax.set_xlabel("Predicted CLV (R$, 12-month)")
ax.set_ylabel("Count")
ax.set_title("Customer Lifetime Value Forecast", fontweight="bold")
ax.legend()

# Health score vs CLV scatter
ax = axes[1, 1]
sample = health.sample(min(3000, len(health)), random_state=42)
scatter = ax.scatter(
    sample["health_score"], sample["clv_forecast"].clip(upper=2000),
    c=sample["churn_probability"], cmap="RdYlGn_r", alpha=0.4,
    s=15, edgecolors="none"
)
ax.set_xlabel("Health Score")
ax.set_ylabel("CLV Forecast (R$)")
ax.set_title("Health Score vs CLV (color = churn risk)", fontweight="bold")
plt.colorbar(scatter, ax=ax, label="Churn Probability", shrink=0.8)

plt.tight_layout()
plt.savefig(FIG_DIR / "21_health_score_clv.png")
plt.close()
print("✓ Saved health score & CLV charts")

# %% [markdown]
# ## 7. Save Results

# %%
# Save full customer health data
output_cols = [
    "customer_id", "Segment", "Recency", "Monetary",
    "order_count", "total_spend", "clv_forecast",
    "churn_probability", "rfm_norm", "churn_risk_norm", "clv_norm",
    "health_score", "health_tier"
]
# Add BG/NBD columns if available
if "prob_alive" in health.columns:
    output_cols.insert(7, "prob_alive")
if "predicted_purchases_90d" in health.columns:
    output_cols.insert(8, "predicted_purchases_90d")

health[output_cols].to_csv(OUTPUT_DIR / "customer_health.csv", index=False)
print(f"✓ Saved customer health data — {len(health):,} customers")

# Print top and bottom customers
print("\n🏆 TOP 10 HEALTHIEST CUSTOMERS:")
top10 = health.nlargest(10, "health_score")[
    ["customer_id", "Segment", "health_score", "clv_forecast", "order_count", "churn_probability"]
]
print(top10.to_string(index=False))

print("\n⚠️  BOTTOM 10 — NEED INTERVENTION:")
bottom10 = health.nsmallest(10, "health_score")[
    ["customer_id", "Segment", "health_score", "clv_forecast", "order_count", "churn_probability"]
]
print(bottom10.to_string(index=False))

print(f"\n✅ CLV Forecast & Health Score complete")
