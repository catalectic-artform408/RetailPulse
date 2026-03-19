# %% [markdown]
# # 06 — Product & Category Deep-Dive
# Revenue by category, review scores, delivery time impact, seasonality analysis.

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
SELECT i.order_id, i.product_id, i.seller_id, i.price, i.freight_value,
       o.order_purchase_timestamp, o.order_delivered_customer_date,
       o.order_estimated_delivery_date, o.order_status,
       COALESCE(t.product_category_name_english, p.product_category_name) AS category,
       r.review_score
FROM order_items i
JOIN orders o ON i.order_id = o.order_id
JOIN products p ON i.product_id = p.product_id
LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
"""
df = pd.read_sql(query, conn, parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date",
    "order_estimated_delivery_date"
])
conn.close()

print(f"Loaded {len(df):,} order items across {df['category'].nunique()} categories")

# %% [markdown]
# ## 1. Top 15 Categories by Revenue

# %%
cat_revenue = df.groupby("category").agg(
    revenue=("price", "sum"),
    orders=("order_id", "nunique"),
    avg_price=("price", "mean"),
    avg_review=("review_score", "mean")
).sort_values("revenue", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("viridis", 15)
bars = ax.barh(cat_revenue.index, cat_revenue["revenue"], color=colors)
ax.set_xlabel("Total Revenue (R$)", fontsize=12)
ax.set_title("Top 15 Product Categories by Revenue", fontsize=14, fontweight="bold")

for bar, val in zip(bars, cat_revenue["revenue"]):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f"R${val:,.0f}", va="center", fontsize=9)

plt.savefig(FIG_DIR / "13_top_categories_revenue.png")
plt.close()
print("✓ Saved top categories by revenue")

# %% [markdown]
# ## 2. Delivery Days vs Review Score

# %%
df_delivered = df.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"]).copy()
df_delivered["delivery_days"] = (
    df_delivered["order_delivered_customer_date"] - df_delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Aggregate by category
cat_delivery = df_delivered.groupby("category").agg(
    avg_delivery_days=("delivery_days", "mean"),
    avg_review_score=("review_score", "mean"),
    order_count=("order_id", "nunique")
).reset_index()

# Filter for categories with at least 100 orders
cat_delivery = cat_delivery[cat_delivery["order_count"] >= 100]

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    cat_delivery["avg_delivery_days"],
    cat_delivery["avg_review_score"],
    s=cat_delivery["order_count"] / 3,
    c=cat_delivery["avg_review_score"],
    cmap="RdYlGn", alpha=0.7, edgecolors="gray", linewidth=0.5
)

# Label top/bottom categories
for _, row in cat_delivery.nlargest(3, "order_count").iterrows():
    ax.annotate(row["category"][:20], (row["avg_delivery_days"], row["avg_review_score"]),
                fontsize=8, ha="center", va="bottom")
for _, row in cat_delivery.nsmallest(3, "avg_review_score").iterrows():
    ax.annotate(row["category"][:20], (row["avg_delivery_days"], row["avg_review_score"]),
                fontsize=8, ha="center", va="top", color="red")

ax.set_xlabel("Average Delivery Days", fontsize=12)
ax.set_ylabel("Average Review Score", fontsize=12)
ax.set_title("Delivery Time vs Review Score by Category\n(bubble size = order count)", fontsize=14, fontweight="bold")
plt.colorbar(scatter, ax=ax, label="Avg Review Score", shrink=0.8)

plt.savefig(FIG_DIR / "14_delivery_vs_review.png")
plt.close()
print("✓ Saved delivery vs review scatter")

# %% [markdown]
# ## 3. Category × Month Seasonality Heatmap

# %%
df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
top_cats = df.groupby("category")["price"].sum().nlargest(12).index

df_top = df[df["category"].isin(top_cats)]
cat_monthly = df_top.pivot_table(
    index="category", columns="month", values="price", aggfunc="sum", fill_value=0
)
cat_monthly.columns = cat_monthly.columns.astype(str)

fig, ax = plt.subplots(figsize=(18, 8))
sns.heatmap(cat_monthly, cmap="YlOrRd", annot=False,
            linewidths=0.3, linecolor="white", ax=ax)
ax.set_title("Monthly Revenue by Top 12 Categories (Seasonality View)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=8)

plt.savefig(FIG_DIR / "15_category_seasonality.png")
plt.close()
print("✓ Saved category seasonality heatmap")

# %% [markdown]
# ## 4. High Volume / Low Revenue Products (Low AOV)

# %%
cat_aov = df.groupby("category").agg(
    orders=("order_id", "nunique"),
    total_revenue=("price", "sum"),
    avg_price=("price", "mean")
).reset_index()
cat_aov = cat_aov[cat_aov["orders"] >= 50]
cat_aov["aov_rank"] = cat_aov["avg_price"].rank()
cat_aov["volume_rank"] = cat_aov["orders"].rank(ascending=False)

# High volume, low AOV = potential upsell opportunities
high_vol_low_aov = cat_aov[
    (cat_aov["volume_rank"] <= cat_aov["volume_rank"].quantile(0.25)) &
    (cat_aov["aov_rank"] <= cat_aov["aov_rank"].quantile(0.25))
].sort_values("orders", ascending=False)

print("\n📊 HIGH VOLUME, LOW AOV CATEGORIES (Upsell Opportunities):")
print(high_vol_low_aov[["category", "orders", "avg_price", "total_revenue"]].to_string(index=False))

# Category review quality
print("\n📊 CATEGORIES WITH WORST REVIEW SCORES:")
worst_reviewed = cat_delivery.nsmallest(10, "avg_review_score")[
    ["category", "avg_review_score", "avg_delivery_days", "order_count"]
]
print(worst_reviewed.to_string(index=False))

print("\n✅ Product & Category deep-dive complete")
