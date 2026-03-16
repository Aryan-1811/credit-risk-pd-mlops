"""
Credit Risk PD Scoring — Streamlit Web App.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import glob
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Risk · PD Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #05080f; }
.block-container { padding: 2.5rem 2.5rem 2rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #080d18 !important;
    border-right: 1px solid #0f2236;
}
section[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem; }

.big-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    color: #f0f8ff;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.sub-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #1e6b9e;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.kpi {
    background: #080d18;
    border: 1px solid #0f2236;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #1e4d6b;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    line-height: 1;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #1e4d6b;
    margin-top: 0.3rem;
}

.formula-box {
    background: #080d18;
    border: 1px solid #0f2236;
    border-left: 3px solid #1e6b9e;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #4a7a9b;
    line-height: 2;
    margin-top: 1rem;
}

.decision-box {
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #1e4d6b;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    margin-top: 1.4rem;
}

.stSlider label { font-family: 'Syne', sans-serif !important; color: #5a8aaa !important; font-size: 0.82rem !important; }
.stNumberInput label { font-family: 'Syne', sans-serif !important; color: #5a8aaa !important; font-size: 0.82rem !important; }

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a5276, #1e6b9e);
    color: #e8f4ff;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    padding: 0.65rem 1.2rem;
    width: 100%;
    margin-top: 1rem;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1e6b9e, #2980b9);
    transform: translateY(-1px);
}

hr { border-color: #0f2236 !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    pkls = glob.glob("mlruns/**/model.pkl", recursive=True)
    if not pkls:
        st.error("No trained model found. Run training_pipeline.py first.")
        st.stop()
    latest = max(pkls, key=os.path.getmtime)
    with open(latest, "rb") as f:
        return pickle.load(f)


FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfDependents",
]
LGD = 0.45


def score(model, features):
    df = pd.DataFrame([features])[FEATURE_COLS]
    return float(model.predict_proba(df)[0, 1])


def risk_info(pd_score):
    if pd_score < 0.05:   return "LOW",       "#00e5a0", "#001a0f"
    elif pd_score < 0.15: return "MEDIUM",     "#f5c518", "#1a1400"
    elif pd_score < 0.30: return "HIGH",       "#ff7b2c", "#1a0e00"
    else:                 return "VERY HIGH",  "#ff3b5c", "#1a0008"


def gauge(pd_score, colour):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pd_score * 100, 2),
        number={
            "suffix": "%",
            "font": {"size": 48, "color": colour, "family": "JetBrains Mono"},
            "valueformat": ".2f",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#0f2236",
                "tickfont": {"color": "#1e4d6b", "size": 9, "family": "JetBrains Mono"},
                "nticks": 6,
            },
            "bar": {"color": colour, "thickness": 0.2},
            "bgcolor": "#05080f",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5],    "color": "#001a0f"},
                {"range": [5, 15],   "color": "#1a1200"},
                {"range": [15, 30],  "color": "#1a0900"},
                {"range": [30, 100], "color": "#1a0005"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 4},
                "thickness": 0.85,
                "value": pd_score * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#05080f",
        plot_bgcolor="#05080f",
        margin=dict(t=30, b=10, l=40, r=40),
        height=240,
        font={"family": "JetBrains Mono"},
    )
    return fig


def stress_chart(base_pd):
    labels = ["Baseline", "Mild\nStress", "Adverse", "Severely\nAdverse"]
    multipliers = [1.0, 1.3, 1.7, 2.5]
    values = [min(base_pd * m, 1.0) * 100 for m in multipliers]
    colours = [risk_info(v / 100)[1] for v in values]

    fig = go.Figure()
    for i, (label, value, colour) in enumerate(zip(labels, values, colours)):
        fig.add_trace(go.Bar(
            x=[label], y=[value],
            marker_color=colour,
            marker_line_width=0,
            text=[f"{value:.1f}%"],
            textposition="outside",
            textfont={"color": colour, "size": 11, "family": "JetBrains Mono"},
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="#05080f",
        plot_bgcolor="#05080f",
        margin=dict(t=30, b=10, l=10, r=10),
        height=210,
        barmode="group",
        bargap=0.35,
        xaxis={
            "tickfont": {"color": "#2a5a7b", "size": 10, "family": "JetBrains Mono"},
            "gridcolor": "#05080f", "zeroline": False,
        },
        yaxis={
            "tickfont": {"color": "#2a5a7b", "size": 9, "family": "JetBrains Mono"},
            "gridcolor": "#0a1520", "zeroline": False,
            "ticksuffix": "%",
            "range": [0, max(values) * 1.35],
        },
    )
    return fig


# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.1rem;color:#f0f8ff;margin-bottom:0.2rem">Borrower Profile</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;color:#1e4d6b;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:1rem">Input financial features</p>', unsafe_allow_html=True)
    st.divider()

    utilisation = st.slider("Credit Utilisation Rate", 0.0, 1.0, 0.30, 0.01)
    age         = st.slider("Age", 18, 80, 40)
    late_30     = st.slider("Times 30–59 Days Late", 0, 10, 0)
    debt_ratio  = st.slider("Debt Ratio", 0.0, 2.0, 0.35, 0.01)
    income      = st.number_input("Monthly Income (£)", 0, 100000, 5000, 500)
    open_lines  = st.slider("Open Credit Lines", 0, 30, 4)
    dependents  = st.slider("Dependents", 0, 10, 0)

    st.divider()
    st.button("⚡  Recalculate")

# ── Score ─────────────────────────────────────────────────────────────────────
features = {
    "RevolvingUtilizationOfUnsecuredLines":    utilisation,
    "age":                                      age,
    "NumberOfTime30-59DaysPastDueNotWorse":    late_30,
    "DebtRatio":                               debt_ratio,
    "MonthlyIncome":                           income,
    "NumberOfOpenCreditLinesAndLoans":         open_lines,
    "NumberOfDependents":                      dependents,
}
pd_score              = score(model, features)
risk_band, colour, bg = risk_info(pd_score)
el                    = pd_score * LGD

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="sub-title">Credit Risk · Probability of Default · Basel II IRB</p>', unsafe_allow_html=True)
st.markdown('<p class="big-title">PD Scoring<br>Dashboard</p>', unsafe_allow_html=True)
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1], gap="large")

with left:
    # Gauge
    st.plotly_chart(gauge(pd_score, colour), use_container_width=True)

    # KPI row
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Risk Band</div>
            <div class="kpi-value" style="color:{colour}">{risk_band}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">Expected Loss</div>
            <div class="kpi-value" style="color:{colour}">£{el:.4f}</div>
            <div class="kpi-sub">per £1 lent</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">LGD Floor</div>
            <div class="kpi-value" style="color:#4a7a9b">45%</div>
            <div class="kpi-sub">Basel II IRB</div>
        </div>""", unsafe_allow_html=True)

    # Formula
    st.markdown(f"""<div class="formula-box">
        EL &nbsp;=&nbsp; PD &nbsp;×&nbsp; LGD &nbsp;×&nbsp; EAD<br>
        &nbsp;&nbsp;&nbsp;=&nbsp; <span style="color:#e8f4ff">{pd_score:.4f}</span>
        &nbsp;×&nbsp; <span style="color:#e8f4ff">0.45</span>
        &nbsp;×&nbsp; <span style="color:#e8f4ff">1.00</span>
        &nbsp;=&nbsp; <span style="color:{colour};font-weight:500">£{el:.4f}</span>
    </div>""", unsafe_allow_html=True)

with right:
    # Stress test
    st.markdown('<p class="section-label">Macro Stress Testing</p>', unsafe_allow_html=True)
    st.plotly_chart(stress_chart(pd_score), use_container_width=True)

    # Decision
    st.markdown('<p class="section-label">Credit Decision</p>', unsafe_allow_html=True)
    if pd_score < 0.05:
        st.markdown(f'<div class="decision-box" style="background:#001a0f;border:1px solid #00e5a0;color:#00e5a0">✅ &nbsp;APPROVE — Strong borrower profile. Standard rate applies.</div>', unsafe_allow_html=True)
    elif pd_score < 0.15:
        st.markdown(f'<div class="decision-box" style="background:#1a1400;border:1px solid #f5c518;color:#f5c518">⚠️ &nbsp;APPROVE WITH CONDITIONS — Acceptable risk. Consider rate adjustment.</div>', unsafe_allow_html=True)
    elif pd_score < 0.30:
        st.markdown(f'<div class="decision-box" style="background:#1a0e00;border:1px solid #ff7b2c;color:#ff7b2c">🔶 &nbsp;REFER TO UNDERWRITER — Elevated risk. Manual review required.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="decision-box" style="background:#1a0008;border:1px solid #ff3b5c;color:#ff3b5c">❌ &nbsp;DECLINE — PD exceeds risk appetite threshold.</div>', unsafe_allow_html=True)

    # Model info
    st.markdown('<p class="section-label">Model Stack</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="kpi" style="margin-top:0">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#2a5a7b;line-height:2.2">
            <span style="color:#1e6b9e">MODEL &nbsp;&nbsp;&nbsp;</span> LightGBM · calibrated (isotonic)<br>
            <span style="color:#1e6b9e">FEATURES </span> WoE-encoded · IV ≥ 0.02 · 7 selected<br>
            <span style="color:#1e6b9e">DATA &nbsp;&nbsp;&nbsp;&nbsp;</span> Give Me Some Credit · 150k rows<br>
            <span style="color:#1e6b9e">TRACKING </span> MLflow Model Registry · v1<br>
            <span style="color:#1e6b9e">PIPELINE </span> Prefect · GitHub Actions CI/CD
        </div>
    </div>""", unsafe_allow_html=True)
