# %% [markdown]
# # 02 — Exploratory Data Analysis
# Distributions, outliers, revenue trends, order status breakdown,
# and payment type analysis. All figures saved to outputs/figures/.

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

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})
sns.set_theme(style="whitegrid", palette="muted")

# %%
conn = sqlite3.connect(DB_PATH)

orders = pd.read_sql("SELECT * FROM orders", conn, parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"
])
payments = pd.read_sql("SELECT * FROM order_payments", conn)
items = pd.read_sql("SELECT * FROM order_items", conn)
products = pd.read_sql("SELECT * FROM products", conn)
categories = pd.read_sql("SELECT * FROM category_translation", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)
reviews = pd.read_sql("SELECT * FROM order_reviews", conn)

conn.close()
print(f"Orders: {len(orders):,} | Items: {len(items):,} | Payments: {len(payments):,}")

# %% [markdown]
# ## 1. Order Status Distribution

# %%
status_counts = orders["order_status"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("viridis", len(status_counts))
bars = ax.barh(status_counts.index, status_counts.values, color=colors)
ax.set_xlabel("Number of Orders")
ax.set_title("Order Status Distribution")
for bar, val in zip(bars, status_counts.values):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=10)
plt.savefig(FIG_DIR / "01_order_status_distribution.png")
plt.close()
print("✓ Saved order status distribution")

# %% [markdown]
# ## 2. Monthly Revenue Trend

# %%
delivered = orders[orders["order_status"] == "delivered"].copy()
order_revenue = payments.groupby("order_id")["payment_value"].sum().reset_index()
delivered = delivered.merge(order_revenue, on="order_id", how="left")
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M")

monthly = delivered.groupby("month").agg(
    revenue=("payment_value", "sum"),
    orders=("order_id", "nunique")
).reset_index()
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(14, 6))

color1 = "#2E86AB"
color2 = "#E8451E"

ax1.fill_between(monthly["month_dt"], monthly["revenue"], alpha=0.3, color=color1)
ax1.plot(monthly["month_dt"], monthly["revenue"], color=color1, linewidth=2, marker="o", markersize=4)
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue (R$)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
ax2.plot(monthly["month_dt"], monthly["orders"], color=color2, linewidth=2, linestyle="--", marker="s", markersize=4)
ax2.set_ylabel("Number of Orders", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

ax1.set_title("Monthly Revenue Trend & Order Volume (Delivered Orders)")
fig.tight_layout()
plt.savefig(FIG_DIR / "02_monthly_revenue_trend.png")
plt.close()
print("✓ Saved monthly revenue trend")

# %% [markdown]
# ## 3. Payment Type Distribution

# %%
pay_dist = payments.groupby("payment_type")["payment_value"].agg(["sum", "count"]).reset_index()
pay_dist.columns = ["payment_type", "total_value", "count"]
pay_dist = pay_dist.sort_values("total_value", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.barh(pay_dist["payment_type"], pay_dist["total_value"], color=sns.color_palette("coolwarm", len(pay_dist)))
ax1.set_xlabel("Total Value (R$)")
ax1.set_title("Payment Value by Type")

ax2.barh(pay_dist["payment_type"], pay_dist["count"], color=sns.color_palette("coolwarm", len(pay_dist)))
ax2.set_xlabel("Number of Transactions")
ax2.set_title("Transaction Count by Type")

plt.tight_layout()
plt.savefig(FIG_DIR / "03_payment_type_distribution.png")
plt.close()
print("✓ Saved payment type distribution")

# %% [markdown]
# ## 4. Order Value Distribution

# %%
order_values = payments.groupby("order_id")["payment_value"].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(order_values, bins=100, color="#5C946E", edgecolor="white", alpha=0.8)
ax1.set_xlabel("Order Value (R$)")
ax1.set_ylabel("Frequency")
ax1.set_title("Order Value Distribution (All)")
ax1.axvline(order_values.median(), color="red", linestyle="--", label=f"Median: R${order_values.median():.0f}")
ax1.axvline(order_values.mean(), color="blue", linestyle="--", label=f"Mean: R${order_values.mean():.0f}")
ax1.legend()

# Zoomed view (< 500)
under_500 = order_values[order_values < 500]
ax2.hist(under_500, bins=80, color="#5C946E", edgecolor="white", alpha=0.8)
ax2.set_xlabel("Order Value (R$)")
ax2.set_ylabel("Frequency")
ax2.set_title(f"Order Value Distribution (< R$500, {len(under_500)/len(order_values)*100:.0f}% of orders)")

plt.tight_layout()
plt.savefig(FIG_DIR / "04_order_value_distribution.png")
plt.close()
print(f"✓ Saved order value distribution — Median: R${order_values.median():.0f}, Mean: R${order_values.mean():.0f}")

# %% [markdown]
# ## 5. Review Score Distribution

# %%
fig, ax = plt.subplots(figsize=(8, 5))
score_counts = reviews["review_score"].value_counts().sort_index()
colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#27AE60"]
ax.bar(score_counts.index, score_counts.values, color=colors, edgecolor="white", width=0.6)
ax.set_xlabel("Review Score")
ax.set_ylabel("Count")
ax.set_title("Review Score Distribution")
ax.set_xticks([1, 2, 3, 4, 5])
for i, v in enumerate(score_counts.values):
    ax.text(score_counts.index[i], v + 500, f"{v:,}", ha="center", fontsize=10)
plt.savefig(FIG_DIR / "05_review_score_distribution.png")
plt.close()
print("✓ Saved review score distribution")

# %% [markdown]
# ## 6. Delivery Time Analysis

# %%
del_orders = orders.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"]).copy()
del_orders["delivery_days"] = (
    del_orders["order_delivered_customer_date"] - del_orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(del_orders["delivery_days"], bins=80, color="#3498DB", edgecolor="white", alpha=0.8, range=(0, 60))
ax1.axvline(del_orders["delivery_days"].median(), color="red", linestyle="--",
            label=f"Median: {del_orders['delivery_days'].median():.0f} days")
ax1.set_xlabel("Days to Deliver")
ax1.set_ylabel("Frequency")
ax1.set_title("Delivery Time Distribution")
ax1.legend()

# Late vs on-time
del_orders_est = del_orders.dropna(subset=["order_estimated_delivery_date"])
del_orders_est["late"] = (
    del_orders_est["order_delivered_customer_date"] > del_orders_est["order_estimated_delivery_date"]
)
late_pct = del_orders_est["late"].mean() * 100
late_counts = del_orders_est["late"].value_counts()
ax2.pie(late_counts, labels=["On-time", "Late"], autopct="%1.1f%%",
        colors=["#2ECC71", "#E74C3C"], startangle=90, textprops={"fontsize": 12})
ax2.set_title(f"On-time vs Late Delivery ({late_pct:.1f}% late)")

plt.tight_layout()
plt.savefig(FIG_DIR / "06_delivery_time_analysis.png")
plt.close()
print(f"✓ Saved delivery time analysis — {late_pct:.1f}% late delivery rate")

# %% [markdown]
# ## 7. Top 10 Customer States

# %%
state_orders = delivered.merge(customers, on="customer_id", how="left")
state_agg = state_orders.groupby("customer_state").agg(
    orders=("order_id", "nunique"),
    revenue=("payment_value", "sum")
).sort_values("revenue", ascending=True).tail(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.barh(state_agg.index, state_agg["revenue"], color=sns.color_palette("Blues_r", 10))
ax1.set_xlabel("Revenue (R$)")
ax1.set_title("Top 10 States by Revenue")

ax2.barh(state_agg.index, state_agg["orders"], color=sns.color_palette("Greens_r", 10))
ax2.set_xlabel("Number of Orders")
ax2.set_title("Top 10 States by Order Count")

plt.tight_layout()
plt.savefig(FIG_DIR / "07_top_states.png")
plt.close()
print("✓ Saved top 10 states analysis")

# %% [markdown]
# ## 8. Day-of-Week & Hour-of-Day Patterns

# %%
delivered["dow"] = delivered["order_purchase_timestamp"].dt.day_name()
delivered["hour"] = delivered["order_purchase_timestamp"].dt.hour

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

dow_counts = delivered["dow"].value_counts().reindex(dow_order)
ax1.bar(range(7), dow_counts.values, color=sns.color_palette("viridis", 7))
ax1.set_xticks(range(7))
ax1.set_xticklabels(dow_order, rotation=45, ha="right")
ax1.set_ylabel("Orders")
ax1.set_title("Orders by Day of Week")

hour_counts = delivered["hour"].value_counts().sort_index()
ax2.fill_between(hour_counts.index, hour_counts.values, alpha=0.3, color="#9B59B6")
ax2.plot(hour_counts.index, hour_counts.values, color="#9B59B6", linewidth=2)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Orders")
ax2.set_title("Orders by Hour of Day")
ax2.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(FIG_DIR / "08_temporal_patterns.png")
plt.close()
print("✓ Saved temporal patterns")

# %%
print("\n✅ EDA complete — all figures saved to outputs/figures/")
