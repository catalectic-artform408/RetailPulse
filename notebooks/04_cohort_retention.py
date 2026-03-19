# %% [markdown]
# # 04 — Cohort Retention Analysis
# Assign customers to cohorts based on first purchase month,
# compute retention matrix, and visualize as an annotated heatmap.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150})
sns.set_theme(style="whitegrid")

# %%
conn = sqlite3.connect(DB_PATH)
query = """
SELECT o.order_id, o.order_purchase_timestamp,
       c.customer_unique_id AS customer_id
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
"""
df = pd.read_sql(query, conn, parse_dates=["order_purchase_timestamp"])
conn.close()

print(f"Working with {df['customer_id'].nunique():,} unique customers")

# %% [markdown]
# ## 1. Assign Cohorts

# %%
# Cohort month = month of first purchase for each customer
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
df["cohort_month"] = df.groupby("customer_id")["order_purchase_timestamp"]\
    .transform("min").dt.to_period("M")
df["period_number"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

print(f"Cohorts span from {df['cohort_month'].min()} to {df['cohort_month'].max()}")
print(f"Max retention period: {df['period_number'].max()} months")

# %% [markdown]
# ## 2. Build Retention Matrix

# %%
cohort_data = df.groupby(["cohort_month", "period_number"])\
    ["customer_id"].nunique().reset_index()
cohort_data.columns = ["cohort_month", "period_number", "customers"]

cohort_pivot = cohort_data.pivot_table(
    index="cohort_month", columns="period_number", values="customers"
)

# Retention %
cohort_sizes = cohort_pivot[0]
retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100

# Filter to cohorts with at least 50 customers and up to 12 periods
significant_cohorts = cohort_sizes[cohort_sizes >= 50].index
retention_filtered = retention.loc[significant_cohorts].iloc[:, :13]

print(f"\nRetention Matrix ({len(retention_filtered)} cohorts × {retention_filtered.shape[1]} periods):")
print(f"Cohort sizes range: {cohort_sizes[significant_cohorts].min():.0f} – {cohort_sizes[significant_cohorts].max():.0f}")

# %% [markdown]
# ## 3. Retention Heatmap

# %%
fig, ax = plt.subplots(figsize=(16, 10))

# Format index for display
retention_display = retention_filtered.copy()
retention_display.index = retention_display.index.astype(str)

sns.heatmap(
    retention_display,
    annot=True, fmt=".0f",
    cmap="Blues", vmin=0, vmax=100,
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Retention Rate (%)", "shrink": 0.8},
    ax=ax
)

ax.set_title("Cohort Retention Heatmap (% of original cohort)", fontsize=16, fontweight="bold")
ax.set_xlabel("Months Since First Purchase", fontsize=12)
ax.set_ylabel("Cohort Month", fontsize=12)
ax.tick_params(axis="both", labelsize=10)

plt.savefig(FIG_DIR / "11_cohort_retention_heatmap.png")
plt.close()
print("✓ Saved cohort retention heatmap")

# %% [markdown]
# ## 4. Average Retention Curve

# %%
avg_retention = retention_filtered.mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(avg_retention.index, avg_retention.values, "o-", color="#2E86AB",
        linewidth=2, markersize=8)
ax.fill_between(avg_retention.index, avg_retention.values, alpha=0.15, color="#2E86AB")
ax.set_xlabel("Months Since First Purchase", fontsize=12)
ax.set_ylabel("Average Retention Rate (%)", fontsize=12)
ax.set_title("Average Customer Retention Curve Across Cohorts", fontsize=14, fontweight="bold")
ax.set_xticks(avg_retention.index)
ax.set_ylim(0, max(avg_retention.values) * 1.1)

# Annotate key points
for i, (x, y) in enumerate(zip(avg_retention.index, avg_retention.values)):
    if i <= 6 or i == len(avg_retention) - 1:
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

plt.savefig(FIG_DIR / "12_avg_retention_curve.png")
plt.close()
print("✓ Saved average retention curve")

# %% [markdown]
# ## 5. Key Insights

# %%
month1_retention = avg_retention.iloc[1] if len(avg_retention) > 1 else 0
month3_retention = avg_retention.iloc[3] if len(avg_retention) > 3 else 0

# Find best performing cohort at period 1
if 1 in retention_filtered.columns:
    best_cohort_m1 = retention_filtered[1].idxmax()
    best_cohort_m1_val = retention_filtered[1].max()
else:
    best_cohort_m1 = "N/A"
    best_cohort_m1_val = 0

print("\n📊 COHORT RETENTION INSIGHTS:")
print(f"  • Average Month-1 retention: {month1_retention:.1f}%")
print(f"  • Average Month-3 retention: {month3_retention:.1f}%")
print(f"  • Best performing cohort (Month-1): {best_cohort_m1} at {best_cohort_m1_val:.1f}%")
print(f"  • Average initial cohort size: {cohort_sizes[significant_cohorts].mean():.0f} customers")
print(f"\n  💡 Insight: Low retention rates suggest most customers are one-time buyers.")
print(f"  → This makes churn prediction and re-engagement campaigns critical.")

# Save retention data for dashboard
retention_filtered.to_csv(PROJECT_ROOT / "outputs" / "cohort_retention.csv")
print("\n✓ Cohort retention data saved")
