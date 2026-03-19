# %% [markdown]
# # 03 — RFM Customer Segmentation
# Recency, Frequency, Monetary scoring and customer segment classification.

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

# Build merged dataset: orders + payments + customers
query = """
SELECT o.order_id, o.order_purchase_timestamp,
       c.customer_unique_id AS customer_id,
       p.payment_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
"""
df = pd.read_sql(query, conn, parse_dates=["order_purchase_timestamp"])
conn.close()

# Aggregate payments per order (handle multi-payment orders)
order_pay = df.groupby(["order_id", "customer_id", "order_purchase_timestamp"])\
    ["payment_value"].sum().reset_index()

print(f"Working with {order_pay['customer_id'].nunique():,} unique customers, "
      f"{len(order_pay):,} delivered orders")

# %% [markdown]
# ## 1. Compute RFM Values

# %%
snapshot_date = order_pay["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date}")

rfm = order_pay.groupby("customer_id").agg(
    Recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
    Frequency=("order_id", "nunique"),
    Monetary=("payment_value", "sum")
).reset_index()

print(f"\nRFM Summary Statistics:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))

# %% [markdown]
# ## 2. RFM Scoring (Quintile-based)

# %%
# Recency: lower is better → reversed labels
rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

# Frequency: handle few unique values with rank method
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Monetary: higher is better
rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_score"] = (rfm["R_score"].astype(str) +
                     rfm["F_score"].astype(str) +
                     rfm["M_score"].astype(str))

# %% [markdown]
# ## 3. Customer Segmentation

# %%
def assign_segment(row):
    r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r <= 2 and f >= 3:
        return "At-Risk"
    elif r <= 2 and f <= 2 and m >= 3:
        return "Can't Lose Them"
    elif r <= 2:
        return "Hibernating"
    elif r == 3:
        return "Need Attention"
    else:
        return "Others"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

seg_summary = rfm.groupby("Segment").agg(
    customers=("customer_id", "count"),
    avg_recency=("Recency", "mean"),
    avg_frequency=("Frequency", "mean"),
    avg_monetary=("Monetary", "mean"),
    total_revenue=("Monetary", "sum")
).round(2)

seg_summary["pct_customers"] = (seg_summary["customers"] / len(rfm) * 100).round(1)
seg_summary["pct_revenue"] = (seg_summary["total_revenue"] / rfm["Monetary"].sum() * 100).round(1)
seg_summary = seg_summary.sort_values("total_revenue", ascending=False)

print("\n📊 SEGMENT SUMMARY:")
print(seg_summary.to_string())

# %% [markdown]
# ## 4. Visualizations

# %%
# --- Segment Distribution (Donut Chart) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

colors_map = {
    "Champions": "#2ECC71", "Loyal Customers": "#27AE60",
    "New Customers": "#3498DB", "Need Attention": "#F39C12",
    "At-Risk": "#E74C3C", "Can't Lose Them": "#C0392B",
    "Hibernating": "#95A5A6", "Others": "#BDC3C7"
}
seg_sorted = seg_summary.sort_values("customers", ascending=False)
seg_colors = [colors_map.get(s, "#999") for s in seg_sorted.index]

wedges, texts, autotexts = ax1.pie(
    seg_sorted["customers"], labels=seg_sorted.index,
    autopct="%1.1f%%", colors=seg_colors, startangle=90,
    pctdistance=0.8, textprops={"fontsize": 9}
)
centre_circle = plt.Circle((0, 0), 0.55, fc="white")
ax1.add_artist(centre_circle)
ax1.set_title("Customer Distribution by Segment", fontsize=14, fontweight="bold")

# --- Revenue by Segment ---
ax2.barh(seg_sorted.index, seg_sorted["total_revenue"], color=seg_colors)
ax2.set_xlabel("Total Revenue (R$)")
ax2.set_title("Revenue by Customer Segment", fontsize=14, fontweight="bold")
for i, (v, pct) in enumerate(zip(seg_sorted["total_revenue"], seg_sorted["pct_revenue"])):
    ax2.text(v + 1000, i, f"R${v:,.0f} ({pct:.1f}%)", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "09_rfm_segments.png")
plt.close()
print("✓ Saved RFM segment chart")

# --- RFM Score Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title in zip(axes, ["Recency", "Frequency", "Monetary"],
                            ["Recency (days)", "Frequency (orders)", "Monetary (R$)"]):
    ax.hist(rfm[col], bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(rfm[col].median(), color="red", linestyle="--",
               label=f"Median: {rfm[col].median():.0f}")
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} Distribution")
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "10_rfm_distributions.png")
plt.close()
print("✓ Saved RFM distributions")

# %% [markdown]
# ## 5. Save RFM Data for Dashboard

# %%
rfm.to_csv(PROJECT_ROOT / "outputs" / "rfm_segments.csv", index=False)
print(f"✓ RFM data saved — {len(rfm):,} customers segmented")

# Business actions per segment
print("\n📋 RECOMMENDED BUSINESS ACTIONS:")
actions = {
    "Champions": "Offer exclusive rewards. Cross-sell premium products. Ask for referrals.",
    "Loyal Customers": "Upsell higher-value items. Engage with loyalty programs.",
    "New Customers": "Welcome email series. Offer onboarding discounts on 2nd purchase.",
    "Need Attention": "Personalized re-engagement campaigns. Satisfaction surveys.",
    "At-Risk": "Send win-back offers with urgency. 10% discount voucher campaign.",
    "Can't Lose Them": "High-priority outreach. Personal account manager follow-up.",
    "Hibernating": "Low-cost reactivation attempts. Remove from paid campaigns.",
    "Others": "Monitor behavior. Generic promotional campaigns."
}
for seg, action in actions.items():
    if seg in seg_summary.index:
        n = seg_summary.loc[seg, "customers"]
        print(f"  {seg} ({n:,} customers): {action}")
