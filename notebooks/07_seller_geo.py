# %% [markdown]
# # 07 — Seller & Geographic Analysis
# Seller scorecard, underperformer flagging, and geographic revenue choropleth.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import sqlite3
import json
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150})
sns.set_theme(style="whitegrid")

# %%
conn = sqlite3.connect(DB_PATH)

# Seller performance data
query = """
SELECT i.seller_id, i.order_id, i.price, i.freight_value,
       s.seller_city, s.seller_state,
       o.order_delivered_customer_date, o.order_estimated_delivery_date,
       o.order_purchase_timestamp, o.order_status,
       r.review_score
FROM order_items i
JOIN sellers s ON i.seller_id = s.seller_id
JOIN orders o ON i.order_id = o.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
"""
df = pd.read_sql(query, conn, parse_dates=[
    "order_delivered_customer_date", "order_estimated_delivery_date",
    "order_purchase_timestamp"
])

# Customer data for geo analysis
customers = pd.read_sql("SELECT * FROM customers", conn)
geo = pd.read_sql("SELECT * FROM geolocation", conn)

# Order-level data for state analysis
order_query = """
SELECT o.order_id, o.order_status,
       c.customer_state,
       p.payment_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
"""
state_df = pd.read_sql(order_query, conn)

conn.close()
print(f"Loaded {len(df):,} order items from {df['seller_id'].nunique():,} sellers")

# %% [markdown]
# ## 1. Seller Scorecard

# %%
# Compute delivery metrics
df_del = df.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"]).copy()
df_del["late"] = df_del["order_delivered_customer_date"] > df_del["order_estimated_delivery_date"]

seller_scorecard = df.groupby("seller_id").agg(
    total_revenue=("price", "sum"),
    order_count=("order_id", "nunique"),
    avg_review_score=("review_score", "mean"),
    avg_freight=("freight_value", "mean"),
    seller_state=("seller_state", "first"),
    seller_city=("seller_city", "first"),
).reset_index()

# On-time delivery %
late_by_seller = df_del.groupby("seller_id")["late"].mean().reset_index()
late_by_seller.columns = ["seller_id", "late_pct"]
late_by_seller["ontime_pct"] = (1 - late_by_seller["late_pct"]) * 100

seller_scorecard = seller_scorecard.merge(late_by_seller, on="seller_id", how="left")
seller_scorecard["ontime_pct"] = seller_scorecard["ontime_pct"].fillna(0)

# Rank sellers
seller_scorecard["revenue_rank"] = seller_scorecard["total_revenue"].rank(ascending=False).astype(int)
seller_scorecard = seller_scorecard.sort_values("revenue_rank")

print("📊 TOP 20 SELLERS:")
top20 = seller_scorecard.head(20)[[
    "revenue_rank", "seller_id", "seller_city", "seller_state",
    "total_revenue", "order_count", "avg_review_score", "ontime_pct"
]].copy()
top20["total_revenue"] = top20["total_revenue"].map("R${:,.0f}".format)
top20["avg_review_score"] = top20["avg_review_score"].round(2)
top20["ontime_pct"] = top20["ontime_pct"].round(1)
print(top20.to_string(index=False))

# %% [markdown]
# ## 2. Flag Underperformers

# %%
# Underperformers: high revenue but low rating or high late %
threshold_orders = 10  # at least 10 orders to be evaluated
active_sellers = seller_scorecard[seller_scorecard["order_count"] >= threshold_orders]

underperformers = active_sellers[
    (active_sellers["avg_review_score"] < 3.5) |
    (active_sellers["ontime_pct"] < 85)
].sort_values("total_revenue", ascending=False)

print(f"\n⚠️  UNDERPERFORMING SELLERS ({len(underperformers)} of {len(active_sellers)} active sellers):")
print(f"  Criteria: avg review < 3.5 OR on-time delivery < 85%")
if len(underperformers) > 0:
    print(underperformers.head(15)[[
        "revenue_rank", "seller_city", "seller_state",
        "total_revenue", "order_count", "avg_review_score", "ontime_pct"
    ]].to_string(index=False))

# %% [markdown]
# ## 3. Geographic Revenue Analysis

# %%
# Revenue by customer state
state_revenue = state_df.groupby("customer_state").agg(
    revenue=("payment_value", "sum"),
    orders=("order_id", "nunique")
).reset_index()
state_revenue["avg_cart_value"] = state_revenue["revenue"] / state_revenue["orders"]
state_revenue = state_revenue.sort_values("revenue", ascending=False)

print("\n📊 REVENUE BY STATE:")
state_display = state_revenue.copy()
state_display["revenue"] = state_display["revenue"].map("R${:,.0f}".format)
state_display["avg_cart_value"] = state_display["avg_cart_value"].map("R${:,.0f}".format)
print(state_display.head(10).to_string(index=False))

# %% [markdown]
# ## 4. Choropleth Map

# %%
# Brazilian states GeoJSON
try:
    with urlopen("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson") as response:
        brazil_geojson = json.loads(response.read())
    print("✓ Loaded Brazil GeoJSON")
except Exception as e:
    print(f"⚠️  Could not load GeoJSON: {e}")
    brazil_geojson = None

if brazil_geojson:
    # Map state abbreviations to GeoJSON feature names
    state_mapping = {
        "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
        "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
        "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
        "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
        "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba",
        "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná",
        "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
        "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
        "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins"
    }

    state_revenue["state_name"] = state_revenue["customer_state"].map(state_mapping)

    fig = px.choropleth(
        state_revenue,
        geojson=brazil_geojson,
        locations="state_name",
        featureidkey="properties.name",
        color="revenue",
        color_continuous_scale="Purples",
        title="Revenue by Brazilian State",
        labels={"revenue": "Revenue (R$)", "state_name": "State"},
        hover_data={"orders": True, "avg_cart_value": ":.0f"}
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        width=800, height=600
    )

    # Save as static image
    fig.write_image(str(FIG_DIR / "16_revenue_choropleth.png"))
    # Also save interactive HTML
    fig.write_html(str(FIG_DIR / "16_revenue_choropleth.html"))
    print("✓ Saved revenue choropleth map (PNG + HTML)")

# %% [markdown]
# ## 5. Seller Visualizations

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Revenue vs Review Score scatter
ax = axes[0]
active = seller_scorecard[seller_scorecard["order_count"] >= 10]
scatter = ax.scatter(
    active["total_revenue"], active["avg_review_score"],
    s=active["order_count"], alpha=0.5,
    c=active["ontime_pct"], cmap="RdYlGn", edgecolors="gray", linewidth=0.3
)
ax.set_xlabel("Total Revenue (R$)")
ax.set_ylabel("Average Review Score")
ax.set_title("Seller Revenue vs Review Score\n(size=orders, color=on-time %)")
plt.colorbar(scatter, ax=ax, label="On-time %", shrink=0.8)

# Seller state distribution
ax = axes[1]
seller_states = seller_scorecard.groupby("seller_state")["seller_id"].nunique()\
    .sort_values(ascending=True).tail(10)
ax.barh(seller_states.index, seller_states.values, color=sns.color_palette("coolwarm", 10))
ax.set_xlabel("Number of Sellers")
ax.set_title("Top 10 States by Seller Count")

plt.tight_layout()
plt.savefig(FIG_DIR / "17_seller_analysis.png")
plt.close()
print("✓ Saved seller analysis charts")

# Save scorecard for dashboard
seller_scorecard.to_csv(PROJECT_ROOT / "outputs" / "seller_scorecard.csv", index=False)
print("\n✅ Seller & Geographic analysis complete")
