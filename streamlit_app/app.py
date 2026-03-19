"""
Olist Customer Intelligence — Streamlit App
4-tab interactive app with What-If simulator, Customer Lookup,
AI-powered insights, and Natural Language query interface.

Deploy: streamlit run streamlit_app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from urllib.request import urlopen

# ─── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Olist Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #16161e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .metric-card h3 {
        margin: 0; padding: 0;
        color: rgba(255,255,255,0.55);
        font-size: 0.8rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .metric-card h1 {
        margin: 6px 0 2px; padding: 0;
        color: #ffffff; font-size: 1.9rem; font-weight: 700;
    }
    .metric-card p {
        margin: 0; padding: 0;
        font-size: 0.78rem; font-weight: 500;
    }

    .health-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #16161e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    .customer-profile {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 14px;
        padding: 24px;
        margin: 10px 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #16161e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 15px 20px;
    }

    /* Fix: make metric label + value visible on dark cards */
    div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.65) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    .ai-insight {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 15px 0;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "olist.db"
    output_dir = project_root / "outputs"

    conn = sqlite3.connect(db_path)
    orders = pd.read_sql("""
        SELECT o.*, c.customer_unique_id, c.customer_state
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    """, conn, parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                             "order_estimated_delivery_date"])
    payments = pd.read_sql("SELECT * FROM order_payments", conn)
    items = pd.read_sql("""
        SELECT i.*, COALESCE(t.product_category_name_english, p.product_category_name) AS category
        FROM order_items i
        JOIN products p ON i.product_id = p.product_id
        LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
    """, conn)
    reviews = pd.read_sql("SELECT * FROM order_reviews", conn)
    conn.close()

    health = pd.read_csv(output_dir / "customer_health.csv") if (output_dir / "customer_health.csv").exists() else None
    rfm = pd.read_csv(output_dir / "rfm_segments.csv") if (output_dir / "rfm_segments.csv").exists() else None
    churn = pd.read_csv(output_dir / "churn_predictions.csv") if (output_dir / "churn_predictions.csv").exists() else None
    seller_sc = pd.read_csv(output_dir / "seller_scorecard.csv") if (output_dir / "seller_scorecard.csv").exists() else None

    return orders, payments, items, reviews, health, rfm, churn, seller_sc

orders, payments, items, reviews, health, rfm, churn, seller_sc = load_data()

# Precompute
delivered = orders[orders["order_status"] == "delivered"].copy()
order_rev = payments.groupby("order_id")["payment_value"].sum().reset_index()
delivered = delivered.merge(order_rev, on="order_id", how="left")

total_revenue = delivered["payment_value"].sum()
total_orders = delivered["order_id"].nunique()
avg_order_value = total_revenue / total_orders
avg_review = reviews["review_score"].mean()
total_customers = delivered["customer_unique_id"].nunique()

# Late delivery stats
del_orders = delivered.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
late_pct = (del_orders["order_delivered_customer_date"] > del_orders["order_estimated_delivery_date"]).mean() * 100

# At-risk metrics (from health data)
if health is not None:
    at_risk_customers = health[health["churn_probability"] > 0.5]
    at_risk_revenue = at_risk_customers["total_spend"].sum()
    at_risk_count = len(at_risk_customers)
    avg_health = health["health_score"].mean()
else:
    at_risk_revenue = 0
    at_risk_count = 0
    avg_health = 0

# Monthly revenue
delivered["month"] = delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)
monthly = delivered.groupby("month").agg(
    revenue=("payment_value", "sum"),
    orders=("order_id", "nunique")
).reset_index()

# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛒 Olist Intelligence")
    st.markdown("---")

    # Simulated auto-refresh
    last_refresh = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"⏱️ **Last refreshed:** {last_refresh}")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Segment filter
    if rfm is not None:
        segments = st.multiselect(
            "Filter by Segment",
            options=sorted(rfm["Segment"].unique()),
            default=sorted(rfm["Segment"].unique()),
        )
    else:
        segments = []

    st.markdown("---")
    st.markdown(f"**📊 Dataset Stats**")
    st.markdown(f"- Orders: {total_orders:,}")
    st.markdown(f"- Customers: {total_customers:,}")
    st.markdown(f"- Revenue: R${total_revenue:,.0f}")

# ─── Tab Layout ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview",
    "👥 Customer Intelligence",
    "🔮 Churn & What-If",
    "🔎 Customer Lookup",
    "🤖 AI Insights"
])

# ─── Tab 1: Executive Overview ─────────────────────────────────────────────────

with tab1:
    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue", f"R${total_revenue/1_000_000:.1f}M", delta=None)
    with c2:
        st.metric("Total Orders", f"{total_orders:,}", delta=None)
    with c3:
        st.metric("Avg Order Value", f"R${avg_order_value:,.0f}", delta=None)
    with c4:
        st.metric("Avg Review Score", f"{avg_review:.2f} ★", delta=None)

    st.markdown("")

    # Revenue trend
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["revenue"],
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
            line=dict(color="#00d4ff", width=2),
            name="Revenue", hovertemplate="Revenue: R$%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["orders"],
            name="Orders", yaxis="y2",
            marker_color="rgba(124,58,237,0.35)",
            hovertemplate="Orders: %{y:,}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Monthly Revenue & Order Volume",
            yaxis=dict(title="Revenue (R$)", gridcolor="rgba(255,255,255,0.04)"),
            yaxis2=dict(title="Orders", overlaying="y", side="right"),
            xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.12), height=420, margin=dict(t=60, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        status_counts = orders["order_status"].value_counts()
        fig = go.Figure(go.Pie(
            labels=status_counts.index, values=status_counts.values,
            hole=0.55, marker=dict(colors=px.colors.qualitative.Set3),
            textinfo="percent", textposition="inside",
            textfont=dict(size=11, color="#1a1a2e"),
            insidetextorientation="radial",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Order Status",
            showlegend=True,
            legend=dict(font=dict(size=10), orientation="h", y=-0.15, x=0.5, xanchor="center"),
            height=420, margin=dict(t=60, b=60, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top categories
    cat_rev = items.merge(delivered[["order_id"]], on="order_id").groupby("category")\
        ["price"].sum().sort_values(ascending=False).head(10)
    fig = go.Figure(go.Bar(
        y=cat_rev.index.str.replace("_", " ").str.title(),
        x=cat_rev.values, orientation="h",
        marker=dict(color=cat_rev.values, colorscale="Viridis"),
        hovertemplate="%{y}: R$%{x:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Top 10 Categories by Revenue",
        xaxis=dict(title="Revenue (R$)", gridcolor="rgba(255,255,255,0.04)"),
        height=380, margin=dict(l=180, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Tab 2: Customer Intelligence ──────────────────────────────────────────────

with tab2:
    if health is not None and rfm is not None:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Customers", f"{len(health):,}")
        with c2:
            st.metric("Avg Health Score", f"{avg_health:.1f}/100")
        with c3:
            st.metric("At-Risk Customers", f"{at_risk_count:,}")
        with c4:
            st.metric("At-Risk Revenue", f"R${at_risk_revenue/1_000_000:.1f}M")

        st.markdown("")

        # RFM segments
        col1, col2 = st.columns(2)
        with col1:
            filtered_rfm = rfm[rfm["Segment"].isin(segments)] if segments else rfm
            seg_counts = filtered_rfm["Segment"].value_counts()
            seg_colors = {
                "Champions": "#10b981", "Loyal Customers": "#059669",
                "New Customers": "#3b82f6", "Need Attention": "#f59e0b",
                "At-Risk": "#ef4444", "Can't Lose Them": "#dc2626",
                "Hibernating": "#6b7280", "Others": "#9ca3af"
            }
            fig = go.Figure(go.Pie(
                labels=seg_counts.index, values=seg_counts.values, hole=0.55,
                marker=dict(colors=[seg_colors.get(s, "#999") for s in seg_counts.index]),
                textinfo="percent+label", textfont=dict(size=10)
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Customer Segments (RFM)", showlegend=False, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Health score distribution
            fig = go.Figure(go.Histogram(
                x=health["health_score"], nbinsx=50,
                marker_color="#00d4ff", opacity=0.8
            ))
            fig.add_vline(x=health["health_score"].median(), line_dash="dash",
                         line_color="red", annotation_text=f"Median: {health['health_score'].median():.1f}")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title="Customer Health Score Distribution",
                xaxis=dict(title="Health Score (0-100)"),
                yaxis=dict(title="Count"), height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Segment details table
        seg_metrics = health.groupby("Segment").agg(
            Customers=("customer_id", "count"),
            Avg_Health=("health_score", "mean"),
            Avg_CLV=("clv_forecast", "mean"),
            Total_Revenue=("total_spend", "sum"),
            Avg_Churn_Risk=("churn_probability", "mean"),
        ).round(1).sort_values("Total_Revenue", ascending=False).reset_index()

        st.markdown("### 📋 Segment Performance")
        st.dataframe(
            seg_metrics.style.format({
                "Avg_CLV": "R${:.0f}", "Total_Revenue": "R${:,.0f}",
                "Avg_Churn_Risk": "{:.1%}", "Avg_Health": "{:.1f}"
            }).background_gradient(subset=["Avg_Health"], cmap="RdYlGn")\
              .background_gradient(subset=["Avg_Churn_Risk"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True
        )

        # Top CLV customers
        st.markdown("### 🏆 Top 20 Customers by Lifetime Value")
        top_clv = health.nlargest(20, "clv_forecast")[
            ["customer_id", "Segment", "health_score", "clv_forecast",
             "order_count", "churn_probability", "health_tier"]
        ].reset_index(drop=True)
        top_clv.index += 1
        st.dataframe(
            top_clv.style.format({
                "clv_forecast": "R${:,.2f}", "health_score": "{:.1f}",
                "churn_probability": "{:.2%}"
            }),
            use_container_width=True
        )
    else:
        st.warning("Run `notebooks/08_clv_health_score.py` first to generate health data.")


# ─── Tab 3: Churn & What-If ────────────────────────────────────────────────────

with tab3:
    if health is not None:
        st.markdown("### 🎛️ What-If Scenario Simulator")
        st.markdown("_Simulate the revenue impact of reducing customer churn._")

        col1, col2 = st.columns([1, 2])
        with col1:
            churn_reduction = st.slider(
                "Churn Reduction (%)",
                min_value=0, max_value=30, value=10, step=1,
                help="Simulate: if we reduce churn by X%, how much revenue do we recover?"
            )

            avg_clv = health["clv_forecast"].mean()
            recovered_customers = int(at_risk_count * churn_reduction / 100)
            recovered_revenue = recovered_customers * avg_clv

            st.markdown("---")
            st.metric("At-Risk Customers", f"{at_risk_count:,}")
            st.metric("Customers Recovered", f"{recovered_customers:,}",
                      delta=f"+{churn_reduction}% retention")
            
            # Format delta dynamically based on size
            rec_delta_str = f"+R${recovered_revenue/1_000_000:.1f}M" if recovered_revenue >= 1_000_000 else f"+R${recovered_revenue:,.0f}"
            st.metric("💰 Revenue Recovered", f"R${recovered_revenue/1_000_000:.2f}M",
                      delta=rec_delta_str)

        with col2:
            # What-if visualization
            reductions = list(range(0, 31))
            revenues = [int(at_risk_count * r / 100) * avg_clv for r in reductions]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=reductions, y=revenues,
                fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
                line=dict(color="#10b981", width=3),
                hovertemplate="Churn reduction: %{x}%<br>Revenue recovered: R$%{y:,.0f}<extra></extra>"
            ))
            # Current selection marker
            fig.add_trace(go.Scatter(
                x=[churn_reduction], y=[recovered_revenue],
                mode="markers", marker=dict(size=16, color="#ef4444", symbol="diamond"),
                name=f"Selected: {churn_reduction}%", showlegend=True,
                hovertemplate=f"▶ {churn_reduction}% reduction<br>R${recovered_revenue:,.0f} recovered<extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title="Revenue Recovery Projection",
                xaxis=dict(title="Churn Reduction (%)", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Recovered Revenue (R$)", gridcolor="rgba(255,255,255,0.04)"),
                height=400, legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Feature importance
        st.markdown("### 📊 Churn Drivers — Feature Importance")
        fig_dir = Path(__file__).resolve().parent.parent / "outputs" / "figures"
        feat_img = fig_dir / "19_feature_importance.png"
        shap_img = fig_dir / "20_shap_summary.png"

        col1, col2 = st.columns(2)
        with col1:
            if feat_img.exists():
                st.image(str(feat_img), caption="Random Forest Feature Importance (Gini)")
        with col2:
            if shap_img.exists():
                st.image(str(shap_img), caption="SHAP Feature Impact on Churn")

        # ROC + Confusion matrix
        roc_img = fig_dir / "18_churn_model_evaluation.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Model Evaluation — ROC Curve & Confusion Matrix")
    else:
        st.warning("Run `notebooks/05_churn_prediction.py` and `08_clv_health_score.py` first.")


# ─── Tab 4: Customer Lookup ────────────────────────────────────────────────────

with tab4:
    st.markdown("### 🔎 Customer Profile Lookup")
    st.markdown("_Search for any customer to see their full profile, segment, health score, and churn risk._")

    if health is not None:
        # Search
        customer_ids = health["customer_id"].tolist()

        col1, col2 = st.columns([3, 1])
        with col1:
            search_id = st.text_input(
                "Enter Customer ID (or pick from sample below)",
                placeholder="e.g. 8a1e8b556fd5feb59a63729b5341e71a"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 Random Customer"):
                search_id = np.random.choice(customer_ids)
                st.rerun()

        # Quick samples
        st.markdown("**Sample customers to try:**")
        sample_cols = st.columns(5)
        sample_segments = ["Champions", "At-Risk", "Hibernating", "Loyal Customers", "New Customers"]
        for i, seg in enumerate(sample_segments):
            seg_data = health[health["Segment"] == seg]
            if len(seg_data) > 0:
                sample_id = seg_data.iloc[0]["customer_id"]
                with sample_cols[i]:
                    if st.button(f"📋 {seg}", key=f"sample_{seg}"):
                        search_id = sample_id

        if search_id and search_id in customer_ids:
            cust = health[health["customer_id"] == search_id].iloc[0]

            st.markdown("---")

            # Profile header
            health_color = "#10b981" if cust["health_score"] >= 60 else "#f59e0b" if cust["health_score"] >= 30 else "#ef4444"
            churn_color = "#10b981" if cust["churn_probability"] < 0.3 else "#f59e0b" if cust["churn_probability"] < 0.7 else "#ef4444"

            st.markdown(f"""
            <div class="customer-profile">
                <h2 style="color: #00d4ff; margin-bottom: 4px;">Customer Profile</h2>
                <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">ID: {cust['customer_id']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Segment", cust["Segment"])
            with m2:
                st.metric("Health Score", f"{cust['health_score']:.1f}/100")
            with m3:
                st.metric("CLV Forecast", f"R${cust['clv_forecast']:,.0f}")
            with m4:
                st.metric("Churn Risk", f"{cust['churn_probability']:.0%}")
            with m5:
                st.metric("Orders", f"{cust.get('order_count', 'N/A')}")

            # Detailed breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Score Breakdown")
                components = pd.DataFrame({
                    "Component": ["RFM Score (35%)", "Churn Risk (40%)", "CLV Score (25%)"],
                    "Score": [cust.get("rfm_norm", 0), cust.get("churn_risk_norm", 0), cust.get("clv_norm", 0)],
                    "Weight": [0.35, 0.40, 0.25]
                })
                components["Contribution"] = components["Score"] * components["Weight"]

                fig = go.Figure(go.Bar(
                    x=components["Component"],
                    y=components["Score"],
                    marker_color=["#3b82f6", "#ef4444", "#10b981"],
                    text=components["Score"].round(1),
                    textposition="outside"
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 110], title="Score (0-100)"),
                    height=300, margin=dict(t=30, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📋 Customer Details")
                details = {
                    "Total Spend": f"R${cust.get('total_spend', 0):,.2f}",
                    "CLV Forecast (12m)": f"R${cust['clv_forecast']:,.2f}",
                    "Health Tier": cust.get("health_tier", "N/A"),
                    "Recency (days)": cust.get("Recency", "N/A"),
                    "Monetary (R$)": f"R${cust.get('Monetary', 0):,.2f}" if pd.notna(cust.get("Monetary")) else "N/A",
                    "Churn Probability": f"{cust['churn_probability']:.2%}",
                }
                for k, v in details.items():
                    st.markdown(f"**{k}:** {v}")

                # Action recommendation
                st.markdown("---")
                st.markdown("#### 💡 Recommended Action")
                actions = {
                    "Champions": "🏆 Offer exclusive rewards. Cross-sell premium products. Ask for referrals.",
                    "Loyal Customers": "💎 Upsell higher-value items. Engage with loyalty program.",
                    "New Customers": "🌟 Send welcome email series. Offer 2nd purchase discount.",
                    "Need Attention": "📧 Personalized re-engagement. Send satisfaction survey.",
                    "At-Risk": "⚠️ Urgent win-back offer with 10% discount voucher.",
                    "Can't Lose Them": "🚨 High-priority: personal account manager outreach.",
                    "Hibernating": "💤 Low-cost reactivation. Remove from paid campaigns.",
                }
                st.info(actions.get(cust["Segment"], "Monitor behavior. Generic campaigns."))

        elif search_id:
            st.warning(f"Customer ID `{search_id}` not found. Try one of the sample buttons above.")
    else:
        st.warning("Run `notebooks/08_clv_health_score.py` first.")


# ─── Tab 5: AI Insights ────────────────────────────────────────────────────────

with tab5:
    st.markdown("### 🤖 AI-Powered Analytics")

    # --- Section 1: Auto-generated Insights ---
    st.markdown("#### 📝 Executive Summary — Auto-Generated")

    # Build metrics context
    metrics_context = {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "avg_review": avg_review,
        "late_delivery_pct": late_pct,
        "at_risk_customers": at_risk_count,
        "at_risk_revenue": at_risk_revenue,
        "avg_health_score": avg_health,
        "total_customers": total_customers,
    }

    def generate_ai_insight(metrics, question=None):
        """Try to generate insights using available LLM API, fallback to pre-generated."""
        try:
            # Try Anthropic first
            import anthropic
            client = anthropic.Anthropic()

            if question:
                prompt = f"""You are a senior business analyst. Given this e-commerce dataset context:
- Total Revenue: R${metrics['total_revenue']:,.0f}
- Total Orders: {metrics['total_orders']:,}
- Avg Order Value: R${metrics['avg_order_value']:,.0f}
- Avg Review Score: {metrics['avg_review']:.2f}
- Late Delivery Rate: {metrics['late_delivery_pct']:.1f}%
- At-Risk Customers: {metrics['at_risk_customers']:,}
- Revenue at Risk: R${metrics['at_risk_revenue']:,.0f}
- Avg Customer Health Score: {metrics['avg_health_score']:.1f}/100

Answer this question concisely: {question}"""
            else:
                prompt = f"""You are a senior business analyst. Given these KPIs from an e-commerce marketplace:
- Total Revenue: R${metrics['total_revenue']:,.0f}
- Total Orders: {metrics['total_orders']:,}
- Avg Order Value: R${metrics['avg_order_value']:,.0f}
- Avg Review Score: {metrics['avg_review']:.2f}/5.0
- Late Delivery Rate: {metrics['late_delivery_pct']:.1f}%
- At-Risk Customers: {metrics['at_risk_customers']:,} (R${metrics['at_risk_revenue']:,.0f} revenue at risk)
- Avg Customer Health Score: {metrics['avg_health_score']:.1f}/100
- Total Customers: {metrics['total_customers']:,}

Write a 4-sentence executive summary highlighting the most critical insight and one specific, actionable recommendation. Be direct and use numbers."""

            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text, "anthropic"

        except Exception:
            pass

        try:
            # Try OpenAI
            import openai
            client = openai.OpenAI()

            prompt_text = question if question else f"Write a 4-sentence executive summary for an e-commerce marketplace with R${metrics['total_revenue']:,.0f} revenue, {metrics['at_risk_customers']:,} at-risk customers (R${metrics['at_risk_revenue']:,.0f} at risk), {metrics['late_delivery_pct']:.1f}% late deliveries, and avg health score {metrics['avg_health_score']:.1f}/100."

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=300
            )
            return response.choices[0].message.content, "openai"

        except Exception:
            pass

        # Fallback: pre-generated insight
        fallback = (
            f"The marketplace generated **R${metrics['total_revenue']:,.0f}** across "
            f"**{metrics['total_orders']:,}** delivered orders with an average order value of "
            f"**R${metrics['avg_order_value']:,.0f}**. "
            f"Customer health is a concern: the average health score sits at "
            f"**{metrics['avg_health_score']:.1f}/100**, with **{metrics['at_risk_customers']:,}** "
            f"customers flagged as at-risk representing **R${metrics['at_risk_revenue']:,.0f}** "
            f"in potential revenue loss. "
            f"Late deliveries ({metrics['late_delivery_pct']:.1f}% of orders) are directly depressing "
            f"review scores — on-time orders average 4.1★ vs 2.5★ for late orders. "
            f"**Priority action:** Launch a targeted reactivation campaign for the at-risk segment "
            f"with a 10% voucher — even 8% conversion recovers R${metrics['at_risk_revenue'] * 0.08:,.0f}."
        )
        return fallback, "fallback"

    # Generate insight
    if st.button("🔄 Generate Fresh Insight", key="gen_insight"):
        with st.spinner("Generating executive summary..."):
            insight, source = generate_ai_insight(metrics_context)
        st.session_state["last_insight"] = insight
        st.session_state["insight_source"] = source

    insight = st.session_state.get("last_insight", None)
    source = st.session_state.get("insight_source", None)

    if insight is None:
        with st.spinner("Generating executive summary..."):
            insight, source = generate_ai_insight(metrics_context)
        st.session_state["last_insight"] = insight
        st.session_state["insight_source"] = source

    import re
    def format_ai_text(text):
        # Convert **bold** to HTML strong tags
        t = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #fff">\1</strong>', text)
        return t.replace('\n', '<br>')

    source_label = {"anthropic": "Claude", "openai": "GPT-4", "fallback": "Pre-generated"}.get(source, "Unknown")
    st.markdown(f'<div class="ai-insight">{format_ai_text(insight)}</div>', unsafe_allow_html=True)
    st.caption(f"Source: {source_label} | Generated at {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("---")

    # --- Section 2: Ask Your Data ---
    st.markdown("#### 💬 Ask Your Data — Natural Language Query")
    st.markdown("_Type a question about the data and get an answer powered by AI._")

    question = st.text_input(
        "Ask anything...",
        placeholder="e.g. Which state has the highest churn rate? What's the avg CLV for Champions?",
        key="nl_query"
    )

    if question:
        with st.spinner("Analyzing..."):
            # First try to answer with data directly
            answer = None

            # Simple pattern matching for common queries (no LLM needed)
            q_lower = question.lower()
            if health is not None:
                if "highest churn" in q_lower and "state" in q_lower:
                    merged = health.merge(
                        orders[["customer_unique_id", "customer_state"]].drop_duplicates(),
                        left_on="customer_id", right_on="customer_unique_id", how="left"
                    )
                    state_churn = merged.groupby("customer_state")["churn_probability"].mean()\
                        .sort_values(ascending=False).head(5)
                    answer = "**Top 5 states by churn rate:**\n\n"
                    for state, rate in state_churn.items():
                        answer += f"- **{state}**: {rate:.1%} avg churn probability\n"

                elif "clv" in q_lower and "champion" in q_lower:
                    champ = health[health["Segment"] == "Champions"]
                    answer = (f"**Champions segment CLV:** Average CLV forecast is "
                             f"**R${champ['clv_forecast'].mean():,.2f}**, with a total of "
                             f"**{len(champ):,}** customers generating "
                             f"**R${champ['total_spend'].sum():,.0f}** in historical revenue.")

                elif "how many" in q_lower and "at-risk" in q_lower:
                    answer = (f"There are **{at_risk_count:,}** at-risk customers with a total "
                             f"of **R${at_risk_revenue:,.0f}** in revenue at risk.")

                elif "health" in q_lower and "segment" in q_lower:
                    seg_health = health.groupby("Segment")["health_score"].mean()\
                        .sort_values(ascending=False)
                    answer = "**Average health score by segment:**\n\n"
                    for seg, score in seg_health.items():
                        answer += f"- **{seg}**: {score:.1f}/100\n"

            # If no pattern matched, try LLM
            if answer is None:
                answer, source = generate_ai_insight(metrics_context, question=question)
                st.markdown(f'<div class="ai-insight">{format_ai_text(answer)}</div>', unsafe_allow_html=True)
                st.caption(f"Answered by: {source_label}")
            else:
                st.markdown(f'<div class="ai-insight">{format_ai_text(answer)}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Try these questions:**")
    example_qs = [
        "Which state has the highest churn rate?",
        "What's the average CLV for Champions?",
        "How many at-risk customers are there?",
        "What's the health score by segment?",
    ]
    for eq in example_qs:
        st.code(eq, language=None)

# ─── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem;'>"
    "Olist Customer Intelligence Dashboard • Built with Streamlit & Plotly • "
    f"Data refreshed: {last_refresh}</p>",
    unsafe_allow_html=True
)
