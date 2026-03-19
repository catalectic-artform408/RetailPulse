"""
Olist E-Commerce Customer Intelligence Dashboard
3-page interactive dashboard built with Plotly Dash
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from urllib.request import urlopen

# ─── Data Loading ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

conn = sqlite3.connect(DB_PATH)

# Load core data
orders = pd.read_sql("""
    SELECT o.*, c.customer_unique_id, c.customer_state, c.customer_city
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
""", conn, parse_dates=["order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"])

payments = pd.read_sql("SELECT * FROM order_payments", conn)
items = pd.read_sql("""
    SELECT i.*, COALESCE(t.product_category_name_english, p.product_category_name) AS category
    FROM order_items i
    JOIN products p ON i.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
""", conn)
reviews = pd.read_sql("SELECT * FROM order_reviews", conn)
sellers = pd.read_sql("SELECT * FROM sellers", conn)

conn.close()

# Load pre-computed data
rfm = pd.read_csv(OUTPUT_DIR / "rfm_segments.csv") if (OUTPUT_DIR / "rfm_segments.csv").exists() else None
seller_scorecard = pd.read_csv(OUTPUT_DIR / "seller_scorecard.csv") if (OUTPUT_DIR / "seller_scorecard.csv").exists() else None

# ─── Precompute Metrics ───────────────────────────────────────────────────────

delivered = orders[orders["order_status"] == "delivered"].copy()
order_revenue = payments.groupby("order_id")["payment_value"].sum().reset_index()
delivered = delivered.merge(order_revenue, on="order_id", how="left")
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)

# KPIs
total_revenue = delivered["payment_value"].sum()
total_orders = delivered["order_id"].nunique()
avg_order_value = total_revenue / total_orders
avg_review = reviews["review_score"].mean()

# Monthly trend
monthly = delivered.groupby("month").agg(
    revenue=("payment_value", "sum"),
    orders=("order_id", "nunique")
).reset_index()

# Top categories
cat_revenue = items.merge(delivered[["order_id"]], on="order_id").groupby("category").agg(
    revenue=("price", "sum"),
    orders=("order_id", "nunique")
).reset_index().sort_values("revenue", ascending=False).head(10)

# Delivery analysis
del_orders = delivered.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"]).copy()
del_orders["late"] = del_orders["order_delivered_customer_date"] > del_orders["order_estimated_delivery_date"]
del_orders["delay_days"] = (
    del_orders["order_delivered_customer_date"] - del_orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400
late_pct = del_orders["late"].mean() * 100

# Delivery vs reviews
del_reviews = del_orders.merge(reviews, on="order_id", how="inner")
del_reviews["delivery_status"] = np.where(del_reviews["late"], "Late", "On-time")
del_rev_agg = del_reviews.groupby("delivery_status")["review_score"].mean().reset_index()

# State revenue
state_rev = delivered.groupby("customer_state").agg(
    revenue=("payment_value", "sum"),
    orders=("order_id", "nunique")
).reset_index().sort_values("revenue", ascending=False)

# Load Brazil GeoJSON
try:
    with urlopen("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson") as response:
        brazil_geojson = json.loads(response.read())
except:
    brazil_geojson = None

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
state_rev["state_name"] = state_rev["customer_state"].map(state_mapping)

# ─── App Setup ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Olist Customer Intelligence Dashboard"
)

# ─── Color & Style Constants ──────────────────────────────────────────────────

CARD_STYLE = {
    "backgroundColor": "#1a1a2e",
    "borderRadius": "12px",
    "border": "1px solid rgba(255,255,255,0.08)",
    "padding": "20px",
    "boxShadow": "0 4px 15px rgba(0,0,0,0.3)",
}

KPI_CARD_STYLE = {
    **CARD_STYLE,
    "textAlign": "center",
    "background": "linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)",
}

ACCENT_COLORS = {
    "primary": "#00d4ff",
    "secondary": "#7c3aed",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}

# ─── Layout Components ────────────────────────────────────────────────────────

def make_kpi_card(title, value, icon, color):
    return dbc.Col(html.Div([
        html.Div([
            html.I(className=f"fas {icon}", style={"fontSize": "28px", "color": color}),
        ], style={"marginBottom": "8px"}),
        html.H2(value, style={"color": "#fff", "fontWeight": "700", "margin": "0", "fontSize": "1.8rem"}),
        html.P(title, style={"color": "rgba(255,255,255,0.6)", "margin": "4px 0 0", "fontSize": "0.85rem"}),
    ], style=KPI_CARD_STYLE), md=3, sm=6, className="mb-3")


# ─── Page 1: Executive Overview ───────────────────────────────────────────────

def page_overview():
    # Revenue trend chart
    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["revenue"],
        fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
        line=dict(color="#00d4ff", width=2),
        name="Revenue", hovertemplate="Month: %{x}<br>Revenue: R$%{y:,.0f}<extra></extra>"
    ))
    fig_revenue.add_trace(go.Bar(
        x=monthly["month"], y=monthly["orders"],
        name="Orders", yaxis="y2",
        marker_color="rgba(124,58,237,0.4)",
        hovertemplate="Orders: %{y:,}<extra></extra>"
    ))
    fig_revenue.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Monthly Revenue & Order Volume", font=dict(size=16)),
        yaxis=dict(title="Revenue (R$)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=-45),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=50, r=50, t=60, b=80),
        height=400,
    )

    # Top categories
    fig_cats = go.Figure(go.Bar(
        y=cat_revenue["category"].str.replace("_", " ").str.title(),
        x=cat_revenue["revenue"],
        orientation="h",
        marker=dict(color=cat_revenue["revenue"], colorscale="Viridis"),
        hovertemplate="Category: %{y}<br>Revenue: R$%{x:,.0f}<extra></extra>",
    ))
    fig_cats.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Top 10 Product Categories by Revenue", font=dict(size=16)),
        xaxis=dict(title="Revenue (R$)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=180, r=20, t=60, b=40),
        height=400,
    )

    # Order status donut
    status_counts = orders["order_status"].value_counts()
    fig_status = go.Figure(go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.55,
        marker=dict(colors=px.colors.qualitative.Set3),
        textinfo="percent+label",
        hovertemplate="%{label}<br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
    ))
    fig_status.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Order Status Distribution", font=dict(size=16)),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )

    return html.Div([
        dbc.Row([
            make_kpi_card("Total Revenue", f"R${total_revenue:,.0f}", "fa-dollar-sign", ACCENT_COLORS["primary"]),
            make_kpi_card("Total Orders", f"{total_orders:,}", "fa-shopping-cart", ACCENT_COLORS["secondary"]),
            make_kpi_card("Avg Order Value", f"R${avg_order_value:,.0f}", "fa-receipt", ACCENT_COLORS["success"]),
            make_kpi_card("Avg Review Score", f"{avg_review:.2f} ★", "fa-star", ACCENT_COLORS["warning"]),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_revenue), style=CARD_STYLE), md=8),
            dbc.Col(html.Div(dcc.Graph(figure=fig_status), style=CARD_STYLE), md=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_cats), style=CARD_STYLE), md=12),
        ]),
    ])


# ─── Page 2: Customer Intelligence ────────────────────────────────────────────

def page_customers():
    charts = []

    if rfm is not None:
        # RFM segment donut
        seg_counts = rfm["Segment"].value_counts()
        seg_colors = {
            "Champions": "#10b981", "Loyal Customers": "#059669",
            "New Customers": "#3b82f6", "Need Attention": "#f59e0b",
            "At-Risk": "#ef4444", "Can't Lose Them": "#dc2626",
            "Hibernating": "#6b7280", "Others": "#9ca3af"
        }

        fig_rfm = go.Figure(go.Pie(
            labels=seg_counts.index,
            values=seg_counts.values,
            hole=0.55,
            marker=dict(colors=[seg_colors.get(s, "#999") for s in seg_counts.index]),
            textinfo="percent+label",
            textfont=dict(size=10),
        ))
        fig_rfm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Customer Segments (RFM Analysis)", font=dict(size=16)),
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
            height=420,
        )

        # Revenue by segment
        seg_rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=True)
        fig_seg_rev = go.Figure(go.Bar(
            y=seg_rev.index,
            x=seg_rev.values,
            orientation="h",
            marker=dict(color=[seg_colors.get(s, "#999") for s in seg_rev.index]),
            hovertemplate="%{y}<br>Revenue: R$%{x:,.0f}<extra></extra>",
        ))
        fig_seg_rev.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Revenue by Customer Segment", font=dict(size=16)),
            xaxis=dict(title="Total Revenue (R$)", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=150, r=20, t=60, b=40),
            height=420,
        )

        # Segment metrics table
        seg_metrics = rfm.groupby("Segment").agg(
            Customers=("customer_id", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum")
        ).round(1).reset_index().sort_values("Total_Revenue", ascending=False)

        seg_table = dash_table.DataTable(
            data=seg_metrics.to_dict("records"),
            columns=[
                {"name": "Segment", "id": "Segment"},
                {"name": "Customers", "id": "Customers", "type": "numeric", "format": dash_table.Format.Format(group=",")},
                {"name": "Avg Recency", "id": "Avg_Recency"},
                {"name": "Avg Frequency", "id": "Avg_Frequency"},
                {"name": "Avg Monetary (R$)", "id": "Avg_Monetary", "type": "numeric", "format": dash_table.Format.Format(precision=0, group=",")},
                {"name": "Total Revenue (R$)", "id": "Total_Revenue", "type": "numeric", "format": dash_table.Format.Format(precision=0, group=",")},
            ],
            style_header={"backgroundColor": "#16213e", "color": "#00d4ff", "fontWeight": "bold", "border": "none"},
            style_cell={"backgroundColor": "#1a1a2e", "color": "#e0e0e0", "border": "1px solid rgba(255,255,255,0.05)", "textAlign": "center", "padding": "10px"},
            style_data_conditional=[
                {"if": {"filter_query": "{Segment} = 'Champions'"}, "backgroundColor": "rgba(16,185,129,0.15)"},
                {"if": {"filter_query": "{Segment} = 'At-Risk'"}, "backgroundColor": "rgba(239,68,68,0.15)"},
                {"if": {"filter_query": "{Segment} = \"Can't Lose Them\""}, "backgroundColor": "rgba(220,38,38,0.15)"},
            ],
        )

        charts.extend([
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(figure=fig_rfm), style=CARD_STYLE), md=5),
                dbc.Col(html.Div(dcc.Graph(figure=fig_seg_rev), style=CARD_STYLE), md=7),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("Segment Details", style={"color": "#00d4ff", "marginBottom": "15px"}),
                    seg_table,
                ], style=CARD_STYLE), md=12),
            ], className="mb-4"),
        ])

    # Top 20 customers by CLV
    clv = rfm.nlargest(20, "Monetary")[["customer_id", "Segment", "Recency", "Frequency", "Monetary"]] if rfm is not None else pd.DataFrame()
    if not clv.empty:
        clv_table = dash_table.DataTable(
            data=clv.to_dict("records"),
            columns=[
                {"name": "Customer ID", "id": "customer_id"},
                {"name": "Segment", "id": "Segment"},
                {"name": "Recency (days)", "id": "Recency"},
                {"name": "Frequency", "id": "Frequency"},
                {"name": "Lifetime Value (R$)", "id": "Monetary", "type": "numeric", "format": dash_table.Format.Format(precision=2, group=",")},
            ],
            style_header={"backgroundColor": "#16213e", "color": "#00d4ff", "fontWeight": "bold", "border": "none"},
            style_cell={"backgroundColor": "#1a1a2e", "color": "#e0e0e0", "border": "1px solid rgba(255,255,255,0.05)", "padding": "10px", "textAlign": "center"},
            page_size=10,
        )
        charts.append(dbc.Row([
            dbc.Col(html.Div([
                html.H5("Top 20 High-Value Customers (CLV)", style={"color": "#00d4ff", "marginBottom": "15px"}),
                clv_table,
            ], style=CARD_STYLE), md=12),
        ]))

    return html.Div(charts)


# ─── Page 3: Operations & Sellers ──────────────────────────────────────────────

def page_operations():
    # Delivery delay vs review
    fig_delay = go.Figure()
    for status, color in [("On-time", "#10b981"), ("Late", "#ef4444")]:
        subset = del_reviews[del_reviews["delivery_status"] == status]
        score_dist = subset["review_score"].value_counts().sort_index()
        fig_delay.add_trace(go.Bar(
            x=score_dist.index, y=score_dist.values,
            name=status, marker_color=color, opacity=0.8
        ))
    fig_delay.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Review Score: On-time vs Late Delivery", font=dict(size=16)),
        xaxis=dict(title="Review Score", tickvals=[1,2,3,4,5], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
        barmode="group",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=50, r=20, t=60, b=40),
        height=400,
    )

    # Late delivery gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=late_pct,
        number=dict(suffix="%", font=dict(size=36)),
        title=dict(text="Late Delivery Rate", font=dict(size=14, color="#e0e0e0")),
        gauge=dict(
            axis=dict(range=[0, 30], tickcolor="#e0e0e0"),
            bar=dict(color="#ef4444"),
            bgcolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0, 5], color="rgba(16,185,129,0.3)"),
                dict(range=[5, 15], color="rgba(245,158,11,0.3)"),
                dict(range=[15, 30], color="rgba(239,68,68,0.3)"),
            ],
            threshold=dict(line=dict(color="#00d4ff", width=3), thickness=0.8, value=10),
        ),
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )

    # Choropleth map
    fig_map = None
    if brazil_geojson:
        fig_map = px.choropleth(
            state_rev, geojson=brazil_geojson, locations="state_name",
            featureidkey="properties.name", color="revenue",
            color_continuous_scale="Purples",
            title="Customer Revenue by State",
            labels={"revenue": "Revenue (R$)"},
            hover_data={"orders": True, "customer_state": True},
        )
        fig_map.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
        )

    # Seller leaderboard
    seller_table_data = None
    if seller_scorecard is not None:
        top_sellers = seller_scorecard.nlargest(15, "total_revenue")[[
            "revenue_rank", "seller_city", "seller_state",
            "total_revenue", "order_count", "avg_review_score", "ontime_pct"
        ]].copy()
        top_sellers["avg_review_score"] = top_sellers["avg_review_score"].round(2)
        top_sellers["ontime_pct"] = top_sellers["ontime_pct"].round(1)

        seller_table_data = dash_table.DataTable(
            data=top_sellers.to_dict("records"),
            columns=[
                {"name": "Rank", "id": "revenue_rank"},
                {"name": "City", "id": "seller_city"},
                {"name": "State", "id": "seller_state"},
                {"name": "Revenue (R$)", "id": "total_revenue", "type": "numeric", "format": dash_table.Format.Format(precision=0, group=",")},
                {"name": "Orders", "id": "order_count"},
                {"name": "Avg Rating", "id": "avg_review_score"},
                {"name": "On-time %", "id": "ontime_pct"},
            ],
            style_header={"backgroundColor": "#16213e", "color": "#00d4ff", "fontWeight": "bold", "border": "none"},
            style_cell={"backgroundColor": "#1a1a2e", "color": "#e0e0e0", "border": "1px solid rgba(255,255,255,0.05)", "padding": "10px", "textAlign": "center"},
            style_data_conditional=[
                {"if": {"filter_query": "{avg_review_score} < 3.5", "column_id": "avg_review_score"},
                 "backgroundColor": "rgba(239,68,68,0.2)", "color": "#ef4444", "fontWeight": "bold"},
                {"if": {"filter_query": "{avg_review_score} >= 4.0", "column_id": "avg_review_score"},
                 "backgroundColor": "rgba(16,185,129,0.2)", "color": "#10b981", "fontWeight": "bold"},
                {"if": {"filter_query": "{ontime_pct} < 85", "column_id": "ontime_pct"},
                 "backgroundColor": "rgba(239,68,68,0.2)", "color": "#ef4444", "fontWeight": "bold"},
                {"if": {"filter_query": "{ontime_pct} >= 95", "column_id": "ontime_pct"},
                 "backgroundColor": "rgba(16,185,129,0.2)", "color": "#10b981", "fontWeight": "bold"},
            ],
            page_size=15,
        )

    layout_items = [
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_delay), style=CARD_STYLE), md=8),
            dbc.Col(html.Div(dcc.Graph(figure=fig_gauge), style=CARD_STYLE), md=4),
        ], className="mb-4"),
    ]

    if fig_map:
        layout_items.append(dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_map), style=CARD_STYLE), md=12),
        ], className="mb-4"))

    if seller_table_data:
        layout_items.append(dbc.Row([
            dbc.Col(html.Div([
                html.H5("Top 15 Seller Leaderboard", style={"color": "#00d4ff", "marginBottom": "15px"}),
                seller_table_data,
            ], style=CARD_STYLE), md=12),
        ]))

    return html.Div(layout_items)


# ─── Main Layout ──────────────────────────────────────────────────────────────

NAVBAR = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.I(className="fas fa-chart-line", style={"fontSize": "24px", "color": "#00d4ff", "marginRight": "10px"}),
                html.Span("Olist Customer Intelligence", style={"fontWeight": "700", "fontSize": "1.2rem", "color": "#fff"}),
            ], style={"display": "flex", "alignItems": "center"})),
        ], align="center"),
        dbc.Row([
            dbc.Col(dbc.Nav([
                dbc.NavItem(dbc.NavLink("Executive Overview", id="nav-overview", href="/", active=True,
                    style={"color": "#00d4ff", "fontWeight": "600"})),
                dbc.NavItem(dbc.NavLink("Customer Intelligence", id="nav-customers", href="/customers",
                    style={"color": "rgba(255,255,255,0.6)"})),
                dbc.NavItem(dbc.NavLink("Operations & Sellers", id="nav-ops", href="/operations",
                    style={"color": "rgba(255,255,255,0.6)"})),
            ], navbar=True)),
        ]),
    ], fluid=True),
    color="#0f0f23",
    dark=True,
    style={"borderBottom": "1px solid rgba(0,212,255,0.2)", "padding": "10px 0"},
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    NAVBAR,
    dbc.Container(id="page-content", fluid=True, style={"padding": "30px", "minHeight": "90vh"}),
], style={"backgroundColor": "#0f0f23", "minHeight": "100vh"})

@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/customers":
        return page_customers()
    elif pathname == "/operations":
        return page_operations()
    return page_overview()


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Dashboard running at: http://localhost:8050")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=False, port=8050)
