"""
Streamlit Analytics Dashboard
==============================
PM-facing dashboard to monitor AI personalization performance:
  - Customer segment health
  - A/B test live results
  - Churn risk heatmap
  - Sentiment distribution
  - Chatbot interaction logs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Nurturing — PM Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #0f3460, #1a5276);
        border-radius: 12px; padding: 20px;
        color: white; text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
    .metric-delta { font-size: 1rem; color: #2ecc71; font-weight: 600; }
    .stMetric label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Simulated Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    n   = 500
    dates = pd.date_range(end=datetime.today(), periods=30)

    customers = pd.DataFrame({
        "customer_id":  [f"CUST-{i:04d}" for i in range(n)],
        "segment":      rng.choice(["High-LTV", "At-Risk", "New", "Dormant"], n, p=[0.25, 0.20, 0.30, 0.25]),
        "churn_risk":   rng.uniform(0, 1, n),
        "sentiment":    rng.choice(["Positive", "Neutral", "Negative"], n, p=[0.52, 0.31, 0.17]),
        "ltv":          rng.exponential(scale=500_000, size=n),
        "last_active":  [datetime.today() - timedelta(days=int(d)) for d in rng.integers(0, 60, n)],
        "nba_clicked":  rng.binomial(1, 0.38, n),
    })

    # A/B test results over 30 days
    ab_daily = pd.DataFrame({
        "date":             dates,
        "control_rate":     0.18  + rng.normal(0, 0.01, 30),
        "variant_rate":     0.205 + rng.normal(0, 0.01, 30),
        "chatbot_csat_ctrl": 0.55 + rng.normal(0, 0.02, 30),
        "chatbot_csat_var":  0.72 + rng.normal(0, 0.02, 30),
    })

    return customers, ab_daily


customers, ab_daily = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/robot.png", width=60)
st.sidebar.title("🤖 AI Nurturing PM")
st.sidebar.markdown("---")

segment_filter = st.sidebar.multiselect(
    "Filter Segment",
    options=["High-LTV", "At-Risk", "New", "Dormant"],
    default=["High-LTV", "At-Risk", "New", "Dormant"],
)
risk_threshold = st.sidebar.slider("Churn Risk Threshold", 0.0, 1.0, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Last updated:** " + datetime.now().strftime("%d %b %Y, %H:%M"))

# ── Filter Data ────────────────────────────────────────────────────────────
df = customers[customers["segment"].isin(segment_filter)]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🤖 AI Customer Nurturing & Personalization")
st.markdown("*AI Product Manager Dashboard — Personal Portfolio Project 2026*")
st.markdown("---")

# ── KPI Metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Customers", f"{len(df):,}", "+12% MoM")
with col2:
    high_risk = (df["churn_risk"] > risk_threshold).sum()
    st.metric("At-Risk Customers", f"{high_risk:,}", f"{high_risk/len(df):.0%} of base", delta_color="inverse")
with col3:
    retention_uplift = 25.3
    st.metric("Retention Uplift", f"+{retention_uplift}%", "vs baseline")
with col4:
    avg_sentiment_pos = (df["sentiment"] == "Positive").mean()
    st.metric("Positive Sentiment", f"{avg_sentiment_pos:.0%}", "+8pp vs last month")
with col5:
    nba_ctr = df["nba_clicked"].mean()
    st.metric("NBA Click-Through", f"{nba_ctr:.0%}", "+22% vs rule-based")

st.markdown("---")

# ── Row 1: Segment + Churn ─────────────────────────────────────────────────
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("📊 Customer Segment Distribution")
    seg_counts = df["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig_seg = px.pie(
        seg_counts, values="Count", names="Segment",
        color_discrete_map={
            "High-LTV": "#0f3460", "At-Risk": "#e94560",
            "New": "#c9963a", "Dormant": "#7f8c8d",
        },
        hole=0.5,
    )
    fig_seg.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
    st.plotly_chart(fig_seg, use_container_width=True)

with col_b:
    st.subheader("🔥 Churn Risk Heatmap by Segment")
    churn_pivot = df.groupby("segment")["churn_risk"].mean().reset_index()
    churn_pivot.columns = ["Segment", "Avg Churn Risk"]
    churn_pivot = churn_pivot.sort_values("Avg Churn Risk", ascending=True)
    fig_churn = px.bar(
        churn_pivot, x="Avg Churn Risk", y="Segment",
        orientation="h",
        color="Avg Churn Risk",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e94560"],
        range_color=[0, 1],
    )
    fig_churn.update_layout(margin=dict(t=0, b=0), height=280, coloraxis_showscale=False)
    st.plotly_chart(fig_churn, use_container_width=True)

# ── Row 2: A/B Test Results ────────────────────────────────────────────────
st.subheader("🧪 A/B Test Live Results")

tab1, tab2, tab3 = st.tabs([
    "Feature 1: Notification Timing",
    "Feature 2: NBA Recommendation",
    "Feature 3: GenAI Chatbot",
])

with tab1:
    fig_ab1 = go.Figure()
    fig_ab1.add_trace(go.Scatter(
        x=ab_daily["date"], y=ab_daily["control_rate"],
        name="Control (Fixed 9AM)", line=dict(color="#7f8c8d", dash="dash")
    ))
    fig_ab1.add_trace(go.Scatter(
        x=ab_daily["date"], y=ab_daily["variant_rate"],
        name="Variant (Sentiment-Triggered)", line=dict(color="#0f3460")
    ))
    fig_ab1.update_layout(
        yaxis_tickformat=".0%", height=260,
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig_ab1, use_container_width=True)
    st.success("✅ **Result:** +14% open rate | p=0.032 | **SHIP VARIANT**")

with tab2:
    col_x, col_y = st.columns(2)
    with col_x:
        st.metric("Control (Rule-Based) CTR", "9.0%")
        st.metric("Variant (ML NBA) CTR", "11.0%", "+22%")
    with col_y:
        st.metric("p-value", "0.018", "Significant ✅")
        st.metric("Hit Rate @5", "74%", "NBA model")
    st.success("✅ **Result:** +22% click-through | p=0.018 | **SHIP VARIANT**")

with tab3:
    fig_ab3 = go.Figure()
    fig_ab3.add_trace(go.Scatter(
        x=ab_daily["date"], y=ab_daily["chatbot_csat_ctrl"],
        name="Control (Scripted)", line=dict(color="#7f8c8d", dash="dash")
    ))
    fig_ab3.add_trace(go.Scatter(
        x=ab_daily["date"], y=ab_daily["chatbot_csat_var"],
        name="Variant (GenAI RAG)", line=dict(color="#e94560")
    ))
    fig_ab3.update_layout(
        yaxis_tickformat=".0%", height=260,
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig_ab3, use_container_width=True)
    st.success("✅ **Result:** +31% CSAT | p=0.041 | **SHIP VARIANT**")

# ── Row 3: Sentiment + LTV ─────────────────────────────────────────────────
col_c, col_d = st.columns([1, 1])

with col_c:
    st.subheader("💬 Sentiment Distribution")
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    fig_sent = px.bar(
        sent_counts, x="Sentiment", y="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e94560"},
    )
    fig_sent.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10))
    st.plotly_chart(fig_sent, use_container_width=True)

with col_d:
    st.subheader("💰 LTV Distribution by Segment")
    fig_ltv = px.box(
        df, x="segment", y="ltv",
        color="segment",
        color_discrete_map={
            "High-LTV": "#0f3460", "At-Risk": "#e94560",
            "New": "#c9963a", "Dormant": "#7f8c8d",
        },
    )
    fig_ltv.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10))
    fig_ltv.update_yaxes(tickprefix="₫", tickformat=",.0f")
    st.plotly_chart(fig_ltv, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built by **Tuyet Nguyen** · AI Product Manager · "
    "[LinkedIn](https://linkedin.com/in/tuyetnguyen1368/) · "
    "[GitHub](https://github.com/tuyetngth2558)"
)
