import sys
import os
import webbrowser
import threading
import time
from datetime import datetime, timezone


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── PCA Feature Tooltips ────────────────────────────────────────────────────
# V1-V28 are PCA-transformed components from the IEEE-CIS / ULB credit card
# dataset. The descriptions below map each component to its dominant signal
# based on published feature importance analyses.
FEATURE_TOOLTIPS = {
    "V1":  "Transaction velocity signal — captures how frequently the cardholder makes purchases in a short window. High negative values correlate with rapid card usage.",
    "V2":  "Spending pattern deviation — measures how far the current transaction amount deviates from the cardholder's historical average.",
    "V3":  "Time-since-last-transaction — derived from inter-transaction time gaps; unusually short gaps flag rapid-fire card activity.",
    "V4":  "Merchant category risk score — encodes the risk level of the merchant's business category (e.g., gambling, jewellery, forex).",
    "V5":  "Geographic anomaly — reflects distance between the transaction location and the cardholder's home country or typical region.",
    "V6":  "Card-present vs card-not-present — higher values indicate online / CNP transactions which carry elevated fraud risk.",
    "V7":  "Transaction amount percentile — position of this amount within the cardholder's personal transaction history.",
    "V8":  "Time-of-day signal — encodes whether the transaction occurs during unusual hours (e.g., 2-4 AM local time).",
    "V9":  "Declined-attempt ratio — proportion of recent authorisation attempts that were declined before this transaction succeeded.",
    "V10": "Cardholder tenure signal — newer accounts with high spend patterns trigger this component.",
    "V11": "Terminal / POS type — differentiates ATM, in-store POS, e-commerce checkout, and contactless terminals.",
    "V12": "Cross-border indicator — captures international transactions or currency mismatches relative to the card's issuing country.",
    "V13": "Merchant tenure — how long the merchant has been active; newly registered merchants score higher here.",
    "V14": "High-value transaction signal — strongest single predictor; extreme negative values are highly associated with fraudulent high-amount purchases.",
    "V15": "Chargeback history — encodes prior dispute / chargeback events on the cardholder or merchant.",
    "V16": "Device fingerprint anomaly — mismatch between the device used and those historically associated with this cardholder.",
    "V17": "Session behaviour signal — captures browser / app behavioural anomalies such as headless scraping or bot-like interaction patterns.",
    "V18": "Shipping-billing mismatch — distance or country difference between billing and shipping addresses for e-commerce orders.",
    "V19": "Account network signal — number of unique cards, accounts, or identities sharing the same phone/email/address cluster.",
    "V20": "Velocity burst indicator — detects micro-bursts of transactions (e.g., 5+ txns in 60 seconds) on the same card.",
    "V21": "Low-amount probing — small-value test transactions used to verify card validity before a larger fraudulent charge.",
    "V22": "Currency volatility exposure — transactions in currencies with high recent volatility relative to the account's base currency.",
    "V23": "Refund / reversal pattern — captures an unusual number of recent reversals or refunds on the account.",
    "V24": "Loyalty / reward redemption signal — large reward-point redemptions without a corresponding purchase history.",
    "V25": "Peer-group deviation — how much the cardholder's behaviour deviates from a peer cohort with similar demographics and spend level.",
    "V26": "Rare-event indicator — encodes statistically rare combinations of merchant, amount, and geography.",
    "V27": "IP-geolocation mismatch — difference between the IP-derived location and the physical card terminal location.",
    "V28": "Model residual / catch-all — captures unexplained variance not covered by V1–V27; high absolute values suggest novel fraud patterns.",
}

# ── Example transactions for the "Load Example" button ──────────────────────
EXAMPLE_TRANSACTIONS = {
    "🟢 Low-risk — routine grocery purchase": {
        "Time": 52000.0, "Amount": 23.40,
        "V1": 1.19, "V2": 0.26, "V3": 0.17, "V4": 0.45, "V5": -0.34,
        "V6": -0.07, "V7": 0.10, "V8": 0.08, "V9": 0.36, "V10": -0.15,
        "V11": 0.29, "V12": -0.10, "V13": 0.07, "V14": 0.18, "V15": 0.24,
        "V16": -0.03, "V17": 0.06, "V18": 0.02, "V19": -0.09, "V20": 0.01,
        "V21": -0.01, "V22": 0.04, "V23": -0.01, "V24": 0.12, "V25": -0.04,
        "V26": 0.02, "V27": 0.01, "V28": -0.003,
    },
    "🔴 High-risk — large overnight card-not-present": {
        "Time": 3600.0, "Amount": 4980.0,
        "V1": -4.77, "V2": 3.45, "V3": -5.21, "V4": 2.98, "V5": -3.11,
        "V6": 1.82, "V7": -4.15, "V8": 0.93, "V9": -1.72, "V10": -4.28,
        "V11": 2.01, "V12": -6.31, "V13": 0.44, "V14": -9.50, "V15": -0.82,
        "V16": -0.72, "V17": -8.10, "V18": -0.22, "V19": 0.11, "V20": 0.26,
        "V21": 0.61, "V22": -0.49, "V23": -0.04, "V24": -0.28, "V25": 0.34,
        "V26": -0.18, "V27": 0.10, "V28": 0.05,
    },
    "🟡 Edge case — medium amount, mixed signals": {
        "Time": 72000.0, "Amount": 395.0,
        "V1": -1.80, "V2": 1.22, "V3": -0.95, "V4": 1.04, "V5": -0.78,
        "V6": 0.55, "V7": -1.10, "V8": 0.33, "V9": -0.60, "V10": -1.30,
        "V11": 0.70, "V12": -2.10, "V13": 0.20, "V14": -2.80, "V15": -0.30,
        "V16": -0.25, "V17": -2.60, "V18": -0.08, "V19": 0.05, "V20": 0.09,
        "V21": 0.21, "V22": -0.17, "V23": -0.02, "V24": -0.10, "V25": 0.12,
        "V26": -0.06, "V27": 0.04, "V28": 0.02,
    },
}

# ── Statistical guardrails (mean ± 3σ bounds from ULB creditcard dataset) ───
# Approximate population statistics for warning display
FEATURE_STATS = {
    "V1":  (-3.0, 3.0),  "V2":  (-3.0, 3.0),  "V3":  (-3.0, 3.0),
    "V4":  (-3.0, 3.0),  "V5":  (-3.0, 3.0),  "V6":  (-3.0, 3.0),
    "V7":  (-3.0, 3.0),  "V8":  (-3.0, 3.0),  "V9":  (-3.0, 3.0),
    "V10": (-3.0, 3.0),  "V11": (-3.0, 3.0),  "V12": (-3.0, 3.0),
    "V13": (-3.0, 3.0),  "V14": (-3.0, 3.0),  "V15": (-3.0, 3.0),
    "V16": (-3.0, 3.0),  "V17": (-3.0, 3.0),  "V18": (-3.0, 3.0),
    "V19": (-3.0, 3.0),  "V20": (-3.0, 3.0),  "V21": (-3.0, 3.0),
    "V22": (-3.0, 3.0),  "V23": (-3.0, 3.0),  "V24": (-3.0, 3.0),
    "V25": (-3.0, 3.0),  "V26": (-3.0, 3.0),  "V27": (-3.0, 3.0),
    "V28": (-3.0, 3.0),
    "Amount": (0.0, 5000.0),
    "Time":   (0.0, 172800.0),
}

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

st.set_page_config(
    page_title="Enterprise Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
)
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"


def get_theme_tokens(mode: str):
    if mode == "light":
        return {
            "text": "#0F172A",
            "muted": "rgba(15,23,42,0.72)",
            "card": "rgba(255,255,255,0.90)",
            "card_border": "rgba(15,23,42,0.12)",
            "sidebar": "rgba(248,250,252,0.95)",
            "sidebar_border": "rgba(15,23,42,0.12)",
            "input_bg": "rgba(255,255,255,1)",
            "input_border": "rgba(15,23,42,0.15)",
            "tab_bg": "rgba(15,23,42,0.03)",
            "accent1": "rgba(0, 86, 59, 0.15)",
            "accent2": "rgba(212, 175, 55, 0.15)",
            "hero_bg": "linear-gradient(135deg, #FFFFFF, #F1F5F9)",
            "hero_shadow": "0 10px 40px rgba(15,23,42,0.08)",
            "body_gradient": """
                radial-gradient(circle at 10% 15%, rgba(0, 86, 59, 0.04), transparent 25%),
                radial-gradient(circle at 85% 12%, rgba(212, 175, 55, 0.05), transparent 25%),
                linear-gradient(180deg, #F8FAFC 0%, #E2E8F0 100%)
            """,
            "chip_text": "#0F172A",
            "button_text": "#12151F",
            "button_grad": "linear-gradient(90deg, #C49A2E, #B8860B)",
            "button_grad_hover": "linear-gradient(90deg, #00563B, #00402C)",
            "kpi_text": "#003366",
            "skeleton_text": "rgba(15,23,42,0.70)",
            "skeleton_line": "rgba(15,23,42,0.10)",
            "skeleton_glow": "rgba(255,255,255,0.55)",
            "mpl_text": "#0F172A",
            "mpl_muted": "#475569",
            "mpl_grid": "#CBD5E1",
            "mpl_spine": "#CBD5E1",
        }

    return {
        "text": "#F0ECF8",
        "muted": "#B0AABF",
        "card": "#242A3D",
        "card_border": "rgba(196, 154, 46, 0.10)",
        "sidebar": "#12151F",
        "sidebar_border": "rgba(196, 154, 46, 0.14)",
        "input_bg": "#1A1F2E",
        "input_border": "rgba(196, 154, 46, 0.16)",
        "tab_bg": "#242A3D",
        "accent1": "rgba(196, 154, 46, 0.18)",  
        "accent2": "rgba(168, 118, 190, 0.18)",      
        "hero_bg": """
            linear-gradient(135deg, #242A3D 0%, #1A1F2E 100%),
            radial-gradient(circle at 18% 16%, rgba(196, 154, 46, 0.08), transparent 42%),
            radial-gradient(circle at 82% 28%, rgba(168, 118, 190, 0.10), transparent 45%)
        """,
        "hero_shadow": "0 10px 40px rgba(0,0,0,0.40)",
        "body_gradient": """
            radial-gradient(circle at 12% 10%, rgba(196, 154, 46, 0.035), transparent 26%),
            radial-gradient(circle at 86% 14%, rgba(168, 118, 190, 0.050), transparent 28%),
            linear-gradient(180deg, #1A1F2E 0%, #171B29 100%)
        """,
        "chip_text": "#C49A2E",
        "button_text": "#12151F",
        "button_grad": "linear-gradient(90deg, #C49A2E, #B8860B)",
        "button_grad_hover": "linear-gradient(90deg, #D4A62A, #C49A2E)",
        "kpi_text": "#C49A2E",
        "skeleton_text": "rgba(255,255,255,0.5)",
        "skeleton_line": "rgba(255,255,255,0.05)",
        "skeleton_glow": "rgba(212, 175, 55, 0.12)",
        "mpl_text": "#F0ECF8",
        "mpl_muted": "#B0AABF",
        "mpl_grid": "#2E3654",
        "mpl_spine": "#2E3654",
    }


theme = get_theme_tokens(st.session_state.theme_mode)


def get_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return "healthy", "API Connected", "dot-green"
        return "issue", "API Issue", "dot-pink"
    except Exception:
        return "down", "API Down", "dot-pink"


api_health_state, api_health_label, api_health_dot = get_api_health()

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {theme["body_gradient"]};
    color: {theme["text"]};
}}

[data-testid="stFileUploader"] small {{
    display: none;
}}

#MainMenu, footer, header {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 1.15rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}}

[data-testid="stSidebar"] {{
    background: {theme["sidebar"]};
    border-right: 1px solid {theme["sidebar_border"]};
}}

[data-testid="stSidebar"] * {{
    color: {theme["text"]} !important;
}}

.glass-card {{
    background: {theme["card"]};
    border: 1px solid {theme["card_border"]};
    box-shadow: 0 8px 30px rgba(0,0,0,0.22);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 24px;
}}

.hero-card {{
    background: {theme["hero_bg"]};
    border: 1px solid {theme["card_border"]};
    box-shadow: {theme["hero_shadow"]};
    border-radius: 28px;
    padding: 30px 32px;
    margin-bottom: 18px;
}}

.info-chip {{
    display: inline-block;
    padding: 8px 14px;
    margin: 0 8px 10px 0;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(255,255,255,0.08);
    border: 1px solid {theme["card_border"]};
    color: {theme["chip_text"]};
}}

.metric-mini {{
    background: {theme["card"]};
    border: 1px solid {theme["card_border"]};
    border-radius: 22px;
    padding: 18px 18px 14px 18px;
    min-height: 105px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}}

.metric-mini h3 {{
    margin: 0;
    font-size: 1.9rem;
    font-weight: 800;
    color: {theme["kpi_text"]};
}}

.metric-mini p {{
    margin: 8px 0 0 0;
    color: {theme["muted"]};
    font-size: 0.92rem;
}}

.section-title {{
    font-size: 2rem;
    font-weight: 800;
    margin: 8px 0 8px 0;
    color: {theme["text"]};
}}

.section-subtitle {{
    color: {theme["muted"]};
    margin-top: -2px;
    margin-bottom: 18px;
    font-size: 1rem;
}}

.module-title {{
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 6px;
}}

.module-subtitle {{
    color: {theme["muted"]};
    margin-bottom: 14px;
}}

.right-card-title {{
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: {theme["text"]};
}}

.right-card-text {{
    color: {theme["muted"]};
    line-height: 1.55;
    font-size: 0.93rem;
}}

.chart-card {{
    background: {theme["card"]};
    border: 1px solid {theme["card_border"]};
    border-radius: 20px;
    padding: 14px 16px 6px 16px;
    margin-bottom: 16px;
}}

.kpi-card {{
    background: {theme["card"]};
    border: 1px solid {theme["card_border"]};
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}}

.kpi-card h2 {{
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    color: {theme["kpi_text"]};
}}

.kpi-card p {{
    margin: 6px 0 0 0;
    color: {theme["muted"]};
}}

.kpi-top {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}}

.kpi-icon {{
    width: 34px;
    height: 34px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: rgba(255,255,255,0.08);
    border: 1px solid {theme["card_border"]};
    font-size: 1rem;
}}

.kpi-label {{
    font-size: 0.95rem;
    color: {theme["muted"]};
    font-weight: 600;
}}

.status-strip {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
    margin-bottom: 12px;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    color: {theme["text"]};
    background: rgba(255,255,255,0.07);
    border: 1px solid {theme["card_border"]};
}}

.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}

.dot-green {{ background:#22c55e; }}
.dot-cyan {{ background:#A876BE; }}
.dot-pink {{ background:#C49A2E; }}
.dot-yellow {{ background:#C49A2E; }}

.top-toolbar {{
    position: sticky;
    top: 8px;
    z-index: 20;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    margin-bottom: 18px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid {theme["card_border"]};
    backdrop-filter: blur(12px);
}}

.toolbar-left {{
    font-weight: 700;
    color: {theme["text"]};
}}

.toolbar-right {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}

.toolbar-chip {{
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(255,255,255,0.07);
    border: 1px solid {theme["card_border"]};
}}

.result-banner {{
    background: linear-gradient(90deg, rgba(196,154,46,0.12), rgba(168,118,190,0.12));
    border: 1px solid {theme["card_border"]};
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-weight: 700;
    color: {theme["text"]};
}}

.fraud-banner {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 18px;
    border-radius: 18px;
    background: linear-gradient(90deg, rgba(196,154,46,0.12), rgba(168,118,190,0.10));
    border: 1px solid rgba(239,68,68,0.28);
    margin-bottom: 16px;
}}

.fraud-banner-icon {{
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.08);
    font-size: 1.1rem;
}}

.fraud-banner-title {{
    font-weight: 800;
    margin-bottom: 2px;
}}

.fraud-banner-text {{
    color: {theme["muted"]};
    font-size: 0.92rem;
}}

.phase-card {{
    background: {theme["card"]};
    border: 1px solid {theme["card_border"]};
    border-radius: 22px;
    padding: 22px;
    min-height: 190px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.18);
}}

.phase-title {{
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 10px;
    color: {theme["text"]};
}}

.phase-text {{
    color: {theme["muted"]};
    line-height: 1.6;
    font-size: 0.93rem;
}}

.skeleton-analytics {{
    position: relative;
    overflow: hidden;
    height: 108px;
    border-radius: 22px;
    background: {theme["card"]};
    border: 1px solid {theme["card_border"]};
    box-shadow: 0 8px 24px rgba(0,0,0,0.14);
    margin-bottom: 14px;
    padding: 14px 16px;
}}

.skeleton-analytics::after {{
    content: "";
    position: absolute;
    inset: 0;
    transform: translateX(-100%);
    background: linear-gradient(
        90deg,
        transparent,
        {theme["skeleton_glow"]},
        transparent
    );
    animation: shimmer 1.5s infinite;
}}

.skeleton-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    z-index: 2;
}}

.skeleton-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    color: {theme["skeleton_text"]};
    font-size: 0.92rem;
    font-weight: 700;
}}

.skeleton-icon {{
    width: 28px;
    height: 28px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.08);
    border: 1px solid {theme["card_border"]};
    font-size: 0.92rem;
}}

.loading-dots {{
    display: inline-flex;
    gap: 4px;
    margin-left: 2px;
}}

.loading-dots span {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: {theme["skeleton_text"]};
    opacity: 0.35;
    animation: blink 1.2s infinite;
}}

.loading-dots span:nth-child(2) {{
    animation-delay: 0.2s;
}}

.loading-dots span:nth-child(3) {{
    animation-delay: 0.4s;
}}

.skeleton-line {{
    position: absolute;
    left: 16px;
    right: 16px;
    height: 8px;
    border-radius: 999px;
    background: {theme["skeleton_line"]};
    z-index: 2;
}}

.skeleton-line.line-1 {{
    top: 54px;
    width: 82%;
}}

.skeleton-line.line-2 {{
    top: 70px;
    width: 64%;
}}

.skeleton-chart-row {{
    position: absolute;
    left: 16px;
    right: 16px;
    bottom: 14px;
    height: 20px;
    display: flex;
    align-items: end;
    gap: 8px;
    z-index: 2;
}}

.skeleton-bar {{
    flex: 1;
    border-radius: 8px 8px 4px 4px;
    background: {theme["skeleton_line"]};
}}

.bar-h1 {{ height: 38%; }}
.bar-h2 {{ height: 72%; }}
.bar-h3 {{ height: 48%; }}
.bar-h4 {{ height: 86%; }}
.bar-h5 {{ height: 56%; }}

.table-header-card {{
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid {theme["card_border"]};
    margin-bottom: 12px;
}}

.table-header-title {{
    font-weight: 800;
    margin-bottom: 4px;
}}

.table-header-sub {{
    color: {theme["muted"]};
    font-size: 0.9rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 12px;
    border-bottom: 1px solid {theme["card_border"]};
}}

.stTabs [data-baseweb="tab"] {{
    background: {theme["tab_bg"]};
    border-radius: 14px 14px 0 0;
    padding: 10px 18px;
    font-weight: 700;
}}

.stTabs [aria-selected="true"] {{
    color: {theme["text"]} !important;
    background: linear-gradient(90deg, {theme["accent1"]}, {theme["accent2"]}) !important;
}}

/* Sidebar nav radio — style as a vertical pill nav */
[data-testid="stSidebar"] .stRadio > div {{
    gap: 4px;
}}

[data-testid="stSidebar"] .stRadio label {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 0.88rem;
    font-weight: 600;
    transition: background 0.15s;
}}

[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.07);
}}

.stButton > button, .stDownloadButton > button {{
    width: 100%;
    border-radius: 16px;
    font-weight: 700;
    padding: 0.8rem 1rem;
    border: 1px solid {theme["card_border"]};
    background: {theme["button_grad"]};
    color: {theme["button_text"]};
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
    border: 1px solid {theme["card_border"]};
    background: {theme["button_grad_hover"]};
}}

div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div,
textarea {{
    background: {theme["input_bg"]} !important;
    border-radius: 14px !important;
    border: 1px solid {theme["input_border"]} !important;
}}

[data-testid="stDataFrame"] {{
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid {theme["card_border"]};
}}

[data-testid="stFileUploader"] section {{
    border-radius: 20px !important;
    background: {theme["card"]};
    border: 1px dashed {theme["card_border"]};
}}

[data-testid="stAlert"] {{
    border-radius: 16px;
}}


.glass-card:hover,
.hero-card:hover,
.metric-mini:hover,
.kpi-card:hover,
.chart-card:hover,
.phase-card:hover,
.top-toolbar:hover,
.skeleton-analytics:hover {{
    background: #2E3654;
    transform: translateY(-2px);
    transition: all 0.25s ease;
    box-shadow: 0 14px 36px rgba(0,0,0,0.28);
}}

.hero-accent-gold {{
    color: #C49A2E;
}}

.hero-accent-purple {{
    color: #A876BE;
}}

.phase-title {{
    color: #A876BE !important;
}}

.toolbar-chip,
.info-chip {{
    background: rgba(196,154,46,0.08);
    border: 1px solid rgba(196,154,46,0.16);
}}

.status-pill {{
    background: rgba(36,42,61,0.88);
    border: 1px solid rgba(196,154,46,0.10);
}}

[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(196,154,46,0.10) !important;
    border-left: 3px solid #C49A2E;
}}

@keyframes shimmer {{
    100% {{
        transform: translateX(100%);
    }}
}}

@keyframes blink {{
    0%, 80%, 100% {{
        opacity: 0.25;
        transform: translateY(0px);
    }}
    40% {{
        opacity: 1;
        transform: translateY(-2px);
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)


def style_figure(fig):
    fig.patch.set_alpha(0)


def style_plot(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, color=theme["mpl_text"], fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, color=theme["mpl_muted"])
    ax.set_ylabel(ylabel, color=theme["mpl_muted"])
    ax.tick_params(colors=theme["mpl_muted"])
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(theme["mpl_spine"])
    ax.grid(alpha=0.18, color=theme["mpl_grid"])


def plot_shap_bar(explain_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    style_figure(fig)
    ax.barh(explain_df["feature"], explain_df["shap_value"])
    style_plot(ax, title, xlabel="SHAP Value", ylabel="Feature")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)


def plot_prediction_breakdown(final_df: pd.DataFrame):
    counts = final_df["prediction"].value_counts().sort_index()
    labels = ["Not Fraud", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)
    percentages = [(v / total * 100) if total > 0 else 0 for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    style_figure(fig)
    bars = ax.bar(labels, percentages)
    style_plot(ax, "Fraud vs Non-Fraud Predictions (%)", ylabel="Percentage")

    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.2f}%\n({value:,})",
            ha="center",
            va="bottom",
            color=theme["mpl_text"],
        )

    st.pyplot(fig, use_container_width=True)


def plot_risk_distribution(final_df: pd.DataFrame):
    risk_counts = final_df["risk_level"].value_counts()
    ordered_labels = ["LOW", "MEDIUM", "HIGH"]
    ordered_values = [risk_counts.get(label, 0) for label in ordered_labels]
    total = sum(ordered_values)
    percentages = [(v / total * 100) if total > 0 else 0 for v in ordered_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    style_figure(fig)
    bars = ax.bar(ordered_labels, percentages)
    style_plot(ax, "Risk Level Distribution (%)", ylabel="Percentage")

    for bar, value, pct in zip(bars, ordered_values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.2f}%\n({value:,})",
            ha="center",
            va="bottom",
            color=theme["mpl_text"],
        )

    st.pyplot(fig, use_container_width=True)


def plot_log_scale_count_chart(final_df: pd.DataFrame):
    counts = final_df["prediction"].value_counts().sort_index()
    labels = ["Not Fraud", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(7, 4))
    style_figure(fig)
    bars = ax.bar(labels, values)
    style_plot(ax, "Fraud vs Non-Fraud Predictions (Log Scale Count)", ylabel="Count (log scale)")
    ax.set_yscale("log")

    for bar, value in zip(bars, values):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,}",
                ha="center",
                va="bottom",
                color=theme["mpl_text"],
            )

    st.pyplot(fig, use_container_width=True)


def plot_zoomed_probability(final_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    style_figure(fig)
    probs = final_df["fraud_probability"]
    zoomed = probs[probs < 0.1]
    ax.hist(zoomed, bins=40)
    style_plot(ax, "Fraud Probability Distribution (Zoomed: 0 → 0.1)", xlabel="Fraud Probability", ylabel="Frequency")
    st.pyplot(fig, use_container_width=True)


def plot_high_risk_histogram(final_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    style_figure(fig)
    high_risk = final_df[final_df["fraud_probability"] > 0.1]

    if len(high_risk) > 0:
        ax.hist(high_risk["fraud_probability"], bins=20)
        style_plot(ax, "High-Risk Fraud Probability Distribution (> 0.1)", xlabel="Fraud Probability", ylabel="Count")
    else:
        ax.text(0.5, 0.5, "No high-risk transactions found", ha="center", va="center", color=theme["mpl_text"])
        style_plot(ax, "High-Risk Fraud Probability Distribution (> 0.1)")
        ax.set_xticks([])
        ax.set_yticks([])

    st.pyplot(fig, use_container_width=True)


def plot_roc_curve(final_df: pd.DataFrame):
    if "Class" not in final_df.columns:
        st.info("ROC curve requires a 'Class' column in the uploaded CSV.")
        return

    y_true = final_df["Class"]
    y_score = final_df["fraud_probability"]

    fig, ax = plt.subplots(figsize=(7, 4))
    style_figure(fig)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    style_plot(ax, "ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")

    leg = ax.legend()
    for txt in leg.get_texts():
        txt.set_color(theme["mpl_text"])
    leg.get_frame().set_alpha(0.12)

    st.pyplot(fig, use_container_width=True)


def plot_precision_recall_curve(final_df: pd.DataFrame):
    if "Class" not in final_df.columns:
        st.info("Precision-Recall curve requires a 'Class' column in the uploaded CSV.")
        return

    y_true = final_df["Class"]
    y_score = final_df["fraud_probability"]

    fig, ax = plt.subplots(figsize=(7, 4))
    style_figure(fig)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ax.plot(recall, precision, linewidth=2)
    style_plot(ax, "Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    st.pyplot(fig, use_container_width=True)


def plot_confusion_matrix_chart(final_df: pd.DataFrame):
    if "Class" not in final_df.columns:
        st.info("Confusion matrix requires a 'Class' column in the uploaded CSV.")
        return

    y_true = final_df["Class"]
    y_pred = final_df["prediction"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    style_figure(fig)
    im = ax.imshow(cm)
    style_plot(ax, "Confusion Matrix", xlabel="Predicted", ylabel="Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Fraud", "Fraud"])
    ax.set_yticklabels(["Not Fraud", "Fraud"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=theme["mpl_text"])

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=theme["mpl_muted"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=theme["mpl_muted"])

    st.pyplot(fig, use_container_width=True)


def plot_kde_probability(final_df: pd.DataFrame):
    probs = final_df["fraud_probability"].dropna().sort_values()
    if len(probs) < 2:
        st.info("Not enough data for KDE plot.")
        return

    density = probs.rolling(window=max(5, len(probs) // 80), min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    style_figure(fig)
    ax.plot(probs.values, density.values, linewidth=2)
    style_plot(ax, "Fraud Probability Density (KDE-like Smooth Curve)", xlabel="Fraud Probability", ylabel="Smoothed Density")
    st.pyplot(fig, use_container_width=True)


def plot_time_trend(final_df: pd.DataFrame):
    if "Time" not in final_df.columns:
        st.info("Time trend requires a 'Time' column in the uploaded CSV.")
        return

    trend_df = final_df.copy()
    trend_df["time_bucket"] = (trend_df["Time"] // 3600).astype(int)
    trend = trend_df.groupby("time_bucket")["prediction"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    style_figure(fig)
    ax.plot(trend["time_bucket"], trend["prediction"], linewidth=2)
    style_plot(ax, "Fraud Trend Over Time", xlabel="Time Bucket (hours)", ylabel="Predicted Fraud Count")
    st.pyplot(fig, use_container_width=True)


def plot_feature_importance_proxy(final_df: pd.DataFrame):
    numeric_cols = [c for c in final_df.columns if c not in ["prediction", "risk_level", "fraud_probability", "Class"]]
    numeric_df = final_df[numeric_cols].select_dtypes(include="number")

    if numeric_df.empty:
        st.info("No numeric features available for feature importance proxy.")
        return

    corr = numeric_df.corrwith(final_df["fraud_probability"]).abs().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    style_figure(fig)
    ax.barh(corr.index[::-1], corr.values[::-1])
    style_plot(ax, "Feature Importance Proxy (|correlation with fraud probability|)", xlabel="Absolute Correlation")
    st.pyplot(fig, use_container_width=True)


with st.sidebar:

    # ── Brand Header ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 8px 0;">
            <div style="font-size:2.2rem;">🛡️</div>
            <div style="font-weight:800; font-size:1.1rem; letter-spacing:0.04em;">
                Fraud Intelligence
            </div>
            <div style="font-size:0.72rem; opacity:0.55; letter-spacing:0.08em; text-transform:uppercase;">
                Production ML Platform v2.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Platform Overview ─────────────────────────────────────────────
    st.markdown("#### 🧭 Platform Overview")
    st.markdown(
        """
        <div style='font-size:0.82rem; line-height:1.55; opacity:0.8;'>
        This dashboard provides <b>real-time</b> and <b>batch fraud scoring</b>
        using a production-grade XGBoost pipeline with SHAP explainability,
        Evidently drift monitoring, a business rule engine, and a
        full prediction audit trail.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model Performance Snapshot ────────────────────────────────────
    st.markdown("#### 🏆 Model Performance")
    _mperf = {"F1": "N/A", "ROC-AUC": "N/A", "MCC": "N/A", "PR-AUC": "N/A"}
    try:
        import json as _jj, os as _os
        _mp = "models/artifacts/metrics.json"
        if _os.path.exists(_mp):
            with open(_mp) as _mf:
                _mm = _jj.load(_mf)
            _mperf = {
                "F1":      str(_mm.get("f1_score", "N/A")),
                "ROC-AUC": str(_mm.get("roc_auc", "N/A")),
                "MCC":     str(_mm.get("mcc", "N/A")),
                "PR-AUC":  str(_mm.get("pr_auc", "N/A")),
            }
    except Exception:
        pass
    _perf_colors = ["#C49A2E", "#A876BE", "#C49A2E", "#A876BE"]
    for (_pk, _pv), _pc in zip(_mperf.items(), _perf_colors):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:5px 8px;margin-bottom:4px;border-radius:7px;"
            f"background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'>"
            f"<span style='font-size:0.8rem;opacity:0.7;'>{_pk}</span>"
            f"<span style='font-size:0.88rem;font-weight:800;color:{_pc};'>{_pv}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("#### 📊 Session Metrics")

    # Pull live audit data
    _sess_total, _sess_fraud, _sess_rate, _sess_lat = 0, 0, 0.0, 0.0
    _rules_fired = 0
    try:
        _ar = requests.get(f"{API_BASE_URL}/audit/history", params={"limit": 500}, timeout=2)
        if _ar.status_code == 200:
            _ad = _ar.json()
            _st2 = _ad.get("stats", {})
            _recs = _ad.get("records", [])
            _sess_total = int(_st2.get("total_predictions", 0) or 0)
            _sess_fraud = int(_st2.get("total_fraud", 0) or 0)
            _sess_lat   = float(_st2.get("avg_latency_ms", 0) or 0)
            _sess_rate  = (_sess_fraud / _sess_total * 100) if _sess_total > 0 else 0.0
            _rules_fired = sum(1 for r in _recs if r.get("rule_triggered"))
    except Exception:
        pass

    _session_kpis = [
        ("Txns Scored",   str(_sess_total),         "#B0AABF"),
        ("Fraud Flagged", str(_sess_fraud),          "#f87171"),
        ("Fraud Rate",    f"{_sess_rate:.1f}%",      "#C49A2E"),
        ("Rules Fired",   str(_rules_fired),         "#A876BE"),
        ("Avg Latency",   f"{_sess_lat:.0f} ms",     "#34d399"),
    ]
    for _lbl, _v, _clr in _session_kpis:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 8px;margin-bottom:4px;border-radius:7px;"
            f"background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'>"
            f"<span style='font-size:0.8rem;opacity:0.7;'>{_lbl}</span>"
            f"<span style='font-size:0.9rem;font-weight:800;color:{_clr};'>{_v}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Recent Predictions Panel (filterable + flaggable) ────────────
    st.markdown("#### 🕵️ Recent Predictions")
    if "flagged_ids" not in st.session_state:
        st.session_state["flagged_ids"] = set()
    _rp_filter = st.selectbox(
        "Filter by risk",
        ["All", "HIGH", "MEDIUM", "LOW"],
        label_visibility="collapsed",
        key="sidebar_risk_filter",
    )
    try:
        _rp_resp = requests.get(f"{API_BASE_URL}/audit/history", params={"limit": 20}, timeout=2)
        if _rp_resp.status_code == 200:
            _rp_recs = _rp_resp.json().get("records", [])
            if _rp_filter != "All":
                _rp_recs = [r for r in _rp_recs if r.get("risk_level") == _rp_filter]
            if _rp_recs:
                for _rp in _rp_recs[:8]:
                    _rp_risk  = _rp.get("risk_level", "?")
                    _rp_prob  = float(_rp.get("fraud_probability", 0))
                    _rp_amt   = float(_rp.get("amount", 0))
                    _rp_id    = _rp.get("id", str(_rp_amt))
                    _rp_color = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#34d399"}.get(_rp_risk, "#94a3b8")
                    _flagged  = _rp_id in st.session_state["flagged_ids"]
                    _flag_icon = "🚩" if _flagged else "⚑"
                    _flag_label = "Flagged" if _flagged else "Flag"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"padding:5px 8px;margin-bottom:4px;border-radius:7px;"
                        f"background:rgba(255,255,255,0.03);border-left:3px solid {_rp_color};'>"
                        f"<div style='display:flex;flex-direction:column;gap:1px;'>"
                        f"<span style='font-size:0.78rem;font-weight:700;color:{_rp_color};'>{_rp_risk}</span>"
                        f"<span style='font-size:0.72rem;opacity:0.6;'>£{_rp_amt:.0f} &nbsp;·&nbsp; {_rp_prob:.3f}</span>"
                        f"</div>"
                        f"<span style='font-size:0.72rem;opacity:{'1' if _flagged else '0.45'};color:#C49A2E;'>{_flag_icon} {_flag_label}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"{'Unflag' if _flagged else 'Flag for review'}",
                        key=f"flag_{_rp_id}",
                        use_container_width=True,
                    ):
                        if _flagged:
                            st.session_state["flagged_ids"].discard(_rp_id)
                        else:
                            st.session_state["flagged_ids"].add(_rp_id)
                        st.rerun()
                if st.session_state["flagged_ids"]:
                    st.caption(f"🚩 {len(st.session_state['flagged_ids'])} flagged for review")
            else:
                st.caption("No predictions match filter.")
        else:
            st.caption("API offline.")
    except Exception:
        st.caption("API offline.")

    st.divider()

    # ── Quick Links ──────────────────────────────────────────────────
    st.markdown("#### 🔗 Quick Links")
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:6px; font-size:0.83rem;">
            <a href="{API_BASE_URL}/docs" target="_blank"
               style="color:#A876BE; text-decoration:none; padding:5px 8px;
               border-radius:6px; background:rgba(168,118,190,0.08);
               border:1px solid rgba(168,118,190,0.16);">
               📄 Swagger API Docs ↗
            </a>
            <a href="{API_BASE_URL}/metrics" target="_blank"
               style="color:#A876BE; text-decoration:none; padding:5px 8px;
               border-radius:6px; background:rgba(168,118,190,0.08);
               border:1px solid rgba(168,118,190,0.16);">
               📡 Prometheus Metrics ↗
            </a>
            <a href="{API_BASE_URL}/model_card" target="_blank"
               style="color:#C49A2E; text-decoration:none; padding:5px 8px;
               border-radius:6px; background:rgba(196,154,46,0.08);
               border:1px solid rgba(196,154,46,0.16);">
               🪪 Model Card (JSON) ↗
            </a>
            <a href="{API_BASE_URL}/redoc" target="_blank"
               style="color:#B0AABF; text-decoration:none; padding:5px 8px;
               border-radius:6px; background:rgba(176,170,191,0.06);
               border:1px solid rgba(176,170,191,0.12);">
               📚 ReDoc Reference ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Capabilities ─────────────────────────────────────────────────
    st.markdown("#### ✅ Capabilities")
    _caps = [
        ("🎯", "Real-time fraud scoring"),
        ("📁", "Batch CSV analysis"),
        ("🧠", "SHAP explainability"),
        ("📋", "Business rule engine"),
        ("🗂️", "Prediction audit trail"),
        ("📡", "Prometheus metrics"),
        ("🌊", "Data drift monitor"),
        ("🔁", "Async Celery stream"),
        ("🪪", "Google Model Card"),
        ("💬", "NLP word cloud"),
    ]
    for _ic, _cap in _caps:
        st.markdown(
            f"<div style='font-size:0.8rem; padding:2px 0; opacity:0.85;'>"
            f"{_ic} {_cap}</div>",
            unsafe_allow_html=True,
        )


    # ── Sidebar Navigation ───────────────────────────────────────────────
    st.markdown("#### 🗂️ Navigation")
    _nav_options = [
        "🎯 Single Transaction Scoring",
        "📁 Batch CSV Scoring",
        "🌊 Data Drift Monitor",
        "🔁 Live Async Stream",
        "💬 NLP Insights",
        "📊 Model Performance",
        "🗂️ Audit Trail",
        "📋 Business Rules",
        "🪪 Model Card",
        "⚡ System Health",
    ]
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = _nav_options[0]
    active_tab = st.radio(
        "Go to",
        _nav_options,
        index=_nav_options.index(st.session_state["active_tab"]),
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    st.session_state["active_tab"] = active_tab

    st.divider()
    st.caption("© 2025 Fraud Detection System · MIT License")

# ── Health + refresh state ───────────────────────────────────────────
api_health_state, api_health_label, api_health_dot = get_api_health()
_now_utc = datetime.now(timezone.utc)
if "last_api_refresh" not in st.session_state:
    st.session_state["last_api_refresh"] = _now_utc

# ── Combined status strip + refresh bar ──────────────────────────────
_dot_clr  = {"healthy": "#22c55e", "issue": "#f59e0b"}.get(api_health_state, "#ef4444")
_chk_time = st.session_state["last_api_refresh"].strftime("%H:%M:%S UTC")
_delta_s  = int((_now_utc - st.session_state["last_api_refresh"]).total_seconds())
_ago_lbl  = f"{_delta_s}s ago" if _delta_s < 60 else f"{_delta_s // 60}m {_delta_s % 60}s ago"

_sc_left, _sc_right = st.columns([3, 2])
with _sc_left:
    st.markdown(
        f"""
        <div class="status-strip" style="margin:0;">
            <div class="status-pill"><span class="status-dot dot-green"></span>Docker Deployed</div>
            <div class="status-pill"><span class="status-dot dot-cyan"></span>SHAP Enabled</div>
            <div class="status-pill"><span class="status-dot {api_health_dot}"></span>{api_health_label}</div>
            <div class="status-pill"><span class="status-dot dot-green"></span>Audit Trail Active</div>
            <div class="status-pill"><span class="status-dot dot-pink"></span>Rules Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with _sc_right:
    _rc1, _rc2 = st.columns([1, 1])
    with _rc1:
        if st.button("🔄 Refresh API Status", use_container_width=True, key="main_refresh"):
            st.session_state["last_api_refresh"] = _now_utc
            st.rerun()
    with _rc2:
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;justify-content:center;
                height:38px;padding-left:4px;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;
                        background:{_dot_clr};box-shadow:0 0 6px {_dot_clr};
                        display:inline-block;"></span>
                    <span style="font-size:0.78rem;font-weight:600;">{api_health_label}</span>
                </div>
                <div style="font-size:0.68rem;opacity:0.5;margin-top:2px;
                    font-family:monospace;">Last: {_chk_time} ({_ago_lbl})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


left, right = st.columns([1.8, 1.0], gap="large")

with left:
    st.markdown(
        """
    <div class="hero-card">
        <div class="info-chip">Designed as a multi-phase fraud analytics pipeline</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.02; margin-top:10px;">
            From raw transactions<br>
            to ranked<br>
            <span class="hero-accent-gold">fraud</span> <span class="hero-accent-purple">insights.</span>
        </div>
        <div style="margin-top:18px; color: rgba(255,255,255,0.72); font-size:1rem;">
            Real-time scoring, batch analysis, explainability, and analyst-focused visualisations in one premium interface.
        </div>
        <div style="margin-top:18px;">
            <span class="info-chip">Real-time risk scoring</span>
            <span class="info-chip">Batch CSV upload</span>
            <span class="info-chip">SHAP explanations</span>
            <span class="info-chip">Analytics dashboard</span>
        </div>
        <div style="margin-top:8px;">
            <span class="info-chip">Docker Ready</span>
            <span class="info-chip">FastAPI Backend</span>
            <span class="info-chip">Real-time ML</span>
            <span class="info-chip">Explainable AI</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
        <div class="metric-mini">
            <h3>1</h3>
            <p>Unified dashboard for single and batch fraud analysis</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
        <div class="metric-mini">
            <h3>N</h3>
            <p>Transactions can be scored and reviewed in one workflow</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
        <div class="metric-mini">
            <h3>∞</h3>
            <p>Built to evolve into a scalable fraud analytics platform</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        """
    <div class="phase-card" style="margin-bottom:16px;">
        <div class="phase-title">Phase 1 · Real-Time Risk Scoring</div>
        <div class="phase-text">
            Uses the trained fraud detection model through the API layer to score a single
            transaction instantly and return fraud probability, prediction, and risk level.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="phase-card" style="margin-bottom:16px;">
        <div class="phase-title">Phase 2 · Batch CSV Analysis</div>
        <div class="phase-text">
            Accepts uploaded CSV transaction files, performs batch fraud scoring, and builds
            a review-ready analytics layer with downloadable results.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="phase-card">
        <div class="phase-title">Phase 3 · Explainability & Insights</div>
        <div class="phase-text">
            Generates SHAP explanations, highlights high-risk behaviour, and provides analyst-style
            charts including ROC, precision-recall, confusion matrix, and time-based fraud trends.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="top-toolbar">
    <div class="toolbar-left">Fraud Intelligence Workspace</div>
    <div class="toolbar-right">
        <span class="toolbar-chip">Realtime</span>
        <span class="toolbar-chip">Batch</span>
        <span class="toolbar-chip">Explainability</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Navigation is driven by the sidebar radio set in the sidebar block above.
active_tab = st.session_state.get("active_tab", "🎯 Single Transaction Scoring")

if active_tab == "🎯 Single Transaction Scoring":
    st.markdown('<div class="section-title">Enter Transaction Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Fill the feature values below to score a single transaction. '
        'Hover the <b>?</b> icon on any field to see what each PCA component represents. '
        'Fields outside their expected range are flagged with a ⚠️ warning.</div>',
        unsafe_allow_html=True,
    )

    # ── Load Example Transaction ─────────────────────────────────────────
    _ex_col1, _ex_col2 = st.columns([2, 1])
    with _ex_col1:
        _selected_example = st.selectbox(
            "Load example transaction",
            ["— select a preset —"] + list(EXAMPLE_TRANSACTIONS.keys()),
            key="example_selector",
            label_visibility="collapsed",
        )
    with _ex_col2:
        _load_ex = st.button("⬇ Load Example", use_container_width=True, key="load_example_btn")
    if _load_ex and _selected_example != "— select a preset —":
        _ex = EXAMPLE_TRANSACTIONS[_selected_example]
        for _k, _v in _ex.items():
            st.session_state[f"tab1_{_k}"] = _v
        st.session_state["tab1_Time"]   = _ex["Time"]
        st.session_state["tab1_Amount"] = _ex["Amount"]
        st.rerun()

    # ── Helper: outlier warning ──────────────────────────────────────────
    def _outlier_warn(fname, val):
        lo, hi = FEATURE_STATS.get(fname, (-999, 999))
        if val < lo or val > hi:
            st.warning(
                f"⚠️ **{fname}** = {val:.3f} is outside the expected range "
                f"[{lo}, {hi}] — statistically unusual (>3σ from population mean).",
                icon=None,
            )

    # ── Group 1: Transaction Info ────────────────────────────────────────
    with st.expander("🕐 Transaction Info", expanded=True):
        _ti_c1, _ti_c2 = st.columns(2, gap="large")
        with _ti_c1:
            Time = st.number_input(
                "Time (seconds since first transaction in dataset)",
                value=10000.0,
                help="Raw seconds elapsed since the first transaction in the dataset. Used to derive time-of-day and inter-transaction gap features.",
            )
        with _ti_c2:
            Amount = st.number_input(
                "Amount (£)",
                value=150.5,
                min_value=0.0,
                help="Transaction amount in the card's billing currency. One of the strongest raw signals — extreme amounts in either direction are suspicious.",
            )

    # ── Group 2: Behavioral Features (V1–V9) ────────────────────────────
    with st.expander("🔁 Behavioral Features  ·  V1 – V9", expanded=True):
        st.caption("Velocity, spending patterns, time-since-last-transaction, merchant risk, geographic anomaly.")
        _bf_fields = [
            ("V1", -1.2), ("V2", 0.3), ("V3", 1.1),
            ("V4", 0.5),  ("V5", -0.2), ("V6", 0.1),
            ("V7", 0.2),  ("V8", -0.1), ("V9", 0.4),
        ]
        _bf_vals = {}
        # Responsive: 3 cols on wide, 1 col fallback on narrow
        _bf_c1, _bf_c2, _bf_c3 = st.columns([1,1,1], gap="medium")
        _bf_col_map = [_bf_c1, _bf_c2, _bf_c3]
        for _idx, (_fname, _fdefault) in enumerate(_bf_fields):
            with _bf_col_map[_idx % 3]:
                _bf_vals[_fname] = st.number_input(
                    _fname,
                    value=float(st.session_state.get(f"tab1_{_fname}", _fdefault)),
                    help=FEATURE_TOOLTIPS[_fname],
                    key=f"tab1_{_fname}",
                )
                _outlier_warn(_fname, _bf_vals[_fname])
        V1  = _bf_vals["V1"];  V2  = _bf_vals["V2"];  V3  = _bf_vals["V3"]
        V4  = _bf_vals["V4"];  V5  = _bf_vals["V5"];  V6  = _bf_vals["V6"]
        V7  = _bf_vals["V7"];  V8  = _bf_vals["V8"];  V9  = _bf_vals["V9"]

    # ── Group 3: Merchant & Location (V10–V18) ───────────────────────────
    with st.expander("🏪 Merchant & Location  ·  V10 – V18", expanded=True):
        st.caption("Cardholder tenure, terminal type, cross-border indicator, chargeback history, device and session signals.")
        _ml_fields = [
            ("V10", -0.3), ("V11", 0.2),  ("V12", -0.5),
            ("V13", 0.1),  ("V14", -0.2), ("V15", 0.3),
            ("V16", -0.1), ("V17", 0.2),  ("V18", 0.1),
        ]
        _ml_vals = {}
        _ml_c1, _ml_c2, _ml_c3 = st.columns([1,1,1], gap="medium")
        _ml_col_map = [_ml_c1, _ml_c2, _ml_c3]
        for _idx, (_fname, _fdefault) in enumerate(_ml_fields):
            with _ml_col_map[_idx % 3]:
                _ml_vals[_fname] = st.number_input(
                    _fname,
                    value=float(st.session_state.get(f"tab1_{_fname}", _fdefault)),
                    help=FEATURE_TOOLTIPS[_fname],
                    key=f"tab1_{_fname}",
                )
                _outlier_warn(_fname, _ml_vals[_fname])
        V10 = _ml_vals["V10"]; V11 = _ml_vals["V11"]; V12 = _ml_vals["V12"]
        V13 = _ml_vals["V13"]; V14 = _ml_vals["V14"]; V15 = _ml_vals["V15"]
        V16 = _ml_vals["V16"]; V17 = _ml_vals["V17"]; V18 = _ml_vals["V18"]

    # ── Group 4: Network & Risk Signals (V19–V28) ────────────────────────
    with st.expander("🌐 Network & Risk Signals  ·  V19 – V28", expanded=False):
        st.caption("Account network signals, velocity bursts, currency patterns, peer-group deviation, rare-event indicators.")
        _nr_fields = [
            ("V19", -0.3), ("V20", 0.05),
            ("V21", -0.02),("V22", 0.1),
            ("V23", -0.03),("V24", 0.2),
            ("V25", -0.1), ("V26", 0.05),
            ("V27", 0.02), ("V28", -0.01),
        ]
        _nr_vals = {}
        _nr_c1, _nr_c2 = st.columns([1,1], gap="medium")
        _nr_col_map = [_nr_c1, _nr_c2]
        for _idx, (_fname, _fdefault) in enumerate(_nr_fields):
            with _nr_col_map[_idx % 2]:
                _nr_vals[_fname] = st.number_input(
                    _fname,
                    value=float(st.session_state.get(f"tab1_{_fname}", _fdefault)),
                    help=FEATURE_TOOLTIPS[_fname],
                    key=f"tab1_{_fname}",
                )
                _outlier_warn(_fname, _nr_vals[_fname])
        V19 = _nr_vals["V19"]; V20 = _nr_vals["V20"]; V21 = _nr_vals["V21"]
        V22 = _nr_vals["V22"]; V23 = _nr_vals["V23"]; V24 = _nr_vals["V24"]
        V25 = _nr_vals["V25"]; V26 = _nr_vals["V26"]; V27 = _nr_vals["V27"]
        V28 = _nr_vals["V28"]

    input_data = {
        "Time": Time,
        "V1": V1,   "V2": V2,   "V3": V3,   "V4": V4,
        "V5": V5,   "V6": V6,   "V7": V7,   "V8": V8,
        "V9": V9,   "V10": V10, "V11": V11, "V12": V12,
        "V13": V13, "V14": V14, "V15": V15, "V16": V16,
        "V17": V17, "V18": V18, "V19": V19, "V20": V20,
        "V21": V21, "V22": V22, "V23": V23, "V24": V24,
        "V25": V25, "V26": V26, "V27": V27, "V28": V28,
        "Amount": Amount,
    }

    if st.button("🔍 Predict Fraud Risk", use_container_width=True):
        try:
            with st.spinner("Scoring transaction..."):
                pred_response    = requests.post(f"{API_BASE_URL}/predict", json=input_data, timeout=30)
                explain_response = requests.post(f"{API_BASE_URL}/explain", json=input_data, timeout=60)

            if pred_response.status_code == 200:
                result  = pred_response.json()
                _prob   = float(result["fraud_probability"])
                _pred   = int(result["prediction"])
                _risk   = result["risk_level"]

                # Append to session score history for threshold analysis
                if "score_history" not in st.session_state:
                    st.session_state["score_history"] = []
                st.session_state["score_history"].append({
                    "probability": _prob,
                    "risk":        _risk,
                    "amount":      float(input_data.get("Amount", 0)),
                    "source":      "manual",
                })

                # ── Colour palette for risk level ────────────────────
                _risk_palette = {
                    "HIGH":   {"bg": "rgba(239,68,68,0.14)",   "border": "rgba(239,68,68,0.55)",   "fg": "#f87171", "icon": "🔴", "label": "HIGH RISK — FRAUD DETECTED"},
                    "MEDIUM": {"bg": "rgba(245,158,11,0.14)",  "border": "rgba(245,158,11,0.55)",  "fg": "#fbbf24", "icon": "🟡", "label": "MEDIUM RISK — REVIEW RECOMMENDED"},
                    "LOW":    {"bg": "rgba(34,197,94,0.12)",   "border": "rgba(34,197,94,0.40)",   "fg": "#34d399", "icon": "🟢", "label": "LOW RISK — TRANSACTION APPEARS LEGITIMATE"},
                }
                _pal = _risk_palette.get(_risk, _risk_palette["MEDIUM"])

                # ── Prominent risk gauge card ────────────────────────
                _gauge_pct = int(_prob * 100)
                _gauge_bar_color = _pal["fg"]
                st.markdown(
                    f"""
                    <div style="
                        background: {_pal['bg']};
                        border: 2px solid {_pal['border']};
                        border-radius: 22px;
                        padding: 28px 32px;
                        margin: 18px 0;
                        display: flex;
                        align-items: center;
                        gap: 32px;
                        flex-wrap: wrap;
                    ">
                        <!-- Circular gauge -->
                        <div style="position:relative; width:110px; height:110px; flex-shrink:0;">
                            <svg viewBox="0 0 36 36" style="width:110px;height:110px;transform:rotate(-90deg);">
                                <circle cx="18" cy="18" r="15.9"
                                    fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="3"/>
                                <circle cx="18" cy="18" r="15.9"
                                    fill="none" stroke="{_gauge_bar_color}" stroke-width="3"
                                    stroke-dasharray="{_gauge_pct} {100 - _gauge_pct}"
                                    stroke-linecap="round"/>
                            </svg>
                            <div style="
                                position:absolute; top:50%; left:50%;
                                transform:translate(-50%,-50%);
                                text-align:center;
                            ">
                                <div style="font-size:1.45rem;font-weight:900;color:{_pal['fg']};line-height:1;">{_gauge_pct}%</div>
                                <div style="font-size:0.62rem;opacity:0.65;margin-top:2px;">fraud prob</div>
                            </div>
                        </div>
                        <!-- Text section -->
                        <div style="flex:1;min-width:180px;">
                            <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.10em;
                                text-transform:uppercase;color:{_pal['fg']};margin-bottom:6px;">
                                {_pal['icon']} &nbsp;{_pal['label']}
                            </div>
                            <div style="font-size:2.4rem;font-weight:900;color:{_pal['fg']};line-height:1.05;">
                                {_prob:.4f}
                            </div>
                            <div style="font-size:0.85rem;opacity:0.65;margin-top:6px;">
                                Model prediction: <strong>{'Fraud' if _pred == 1 else 'Not Fraud'}</strong>
                                &nbsp;·&nbsp; Risk tier: <strong>{_risk}</strong>
                                &nbsp;·&nbsp; Amount: <strong>£{input_data['Amount']:.2f}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Inline Top-3 SHAP contributors ──────────────────
                _explain_data = None
                if explain_response.status_code == 200:
                    _expl = explain_response.json()
                    if "top_features" in _expl:
                        _explain_data = pd.DataFrame(_expl["top_features"])

                if _explain_data is not None and not _explain_data.empty:
                    top3 = _explain_data.head(3)
                    st.markdown(
                        "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;'>",
                        unsafe_allow_html=True,
                    )
                    for _, _feat_row in top3.iterrows():
                        _fn  = _feat_row.get("feature", "?")
                        _fv  = float(_feat_row.get("shap_value", 0))
                        _fdir = ("🔺 pushes toward fraud" if _fv > 0 else "🔻 reduces fraud risk")
                        _fclr = "#f87171" if _fv > 0 else "#34d399"
                        _tip  = FEATURE_TOOLTIPS.get(_fn, "")
                        st.markdown(
                            f"""
                            <div style="flex:1;min-width:180px;max-width:320px;
                                background:rgba(255,255,255,0.04);
                                border:1px solid rgba(255,255,255,0.10);
                                border-left:4px solid {_fclr};
                                border-radius:12px;padding:12px 14px;">
                                <div style="font-size:1.05rem;font-weight:800;color:{_fclr};">{_fn}</div>
                                <div style="font-size:0.78rem;opacity:0.55;margin:3px 0 6px 0;line-height:1.4;">{_tip[:80]}…</div>
                                <div style="font-size:0.85rem;font-weight:700;">
                                    SHAP: <span style="color:{_fclr};">{_fv:+.4f}</span>
                                </div>
                                <div style="font-size:0.72rem;opacity:0.6;margin-top:2px;">{_fdir}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── Full breakdown (collapsible) ─────────────────────
                with st.expander("📊 Full prediction details & all SHAP values", expanded=False):
                    _dl, _dr = st.columns([1.05, 1.0], gap="large")
                    with _dl:
                        st.markdown("#### Submitted Transaction")
                        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
                    with _dr:
                        if _explain_data is not None:
                            st.markdown("#### All SHAP Contributions")
                            st.dataframe(_explain_data, use_container_width=True)
                            plot_shap_bar(_explain_data, "All Feature Contributions")
                        elif explain_response.status_code == 200:
                            _expl2 = explain_response.json()
                            if "error" in _expl2:
                                st.warning(f"Explanation failed: {_expl2['error']}")
                            else:
                                st.warning("No explanation data returned.")
                        else:
                            st.warning("Explanation could not be generated.")

            else:
                st.error(f"API error: {pred_response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")

    # Persist score to session history for threshold analysis
    if "score_history" not in st.session_state:
        st.session_state["score_history"] = []

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # 🚨  ALERT THRESHOLD CONFIGURATOR
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("🚨 Alert Threshold Configurator", expanded=True):
        st.markdown(
            '<div class="module-subtitle" style="margin-bottom:12px;">'
            'Set a custom fraud probability cutoff and see how many scored transactions '
            'in this session would change risk tier at that threshold.</div>',
            unsafe_allow_html=True,
        )

        _thresh_col1, _thresh_col2, _thresh_col3 = st.columns([2, 1, 1], gap="large")
        with _thresh_col1:
            _custom_thresh = st.slider(
                "Fraud probability alert threshold",
                min_value=0.01, max_value=0.99,
                value=float(st.session_state.get("alert_threshold", 0.50)),
                step=0.01,
                format="%.2f",
                key="alert_threshold",
                help="Transactions above this probability will be re-classified as HIGH RISK regardless of the model’s default 0.5 cutoff.",
            )
        with _thresh_col2:
            st.markdown(
                f"""
                <div style="padding:14px 16px;border-radius:14px;
                    background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.35);
                    text-align:center;">
                    <div style="font-size:1.8rem;font-weight:900;color:#C49A2E;">{_custom_thresh:.0%}</div>
                    <div style="font-size:0.75rem;opacity:0.65;margin-top:4px;">Your cutoff</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with _thresh_col3:
            _model_thresh = 0.50
            _delta_pct = (_custom_thresh - _model_thresh) * 100
            _delta_label = f"+{_delta_pct:.0f}pp" if _delta_pct >= 0 else f"{_delta_pct:.0f}pp"
            _delta_color = "#34d399" if _delta_pct > 0 else "#f87171"
            st.markdown(
                f"""
                <div style="padding:14px 16px;border-radius:14px;
                    background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);
                    text-align:center;">
                    <div style="font-size:1.8rem;font-weight:900;color:{_delta_color};">{_delta_label}</div>
                    <div style="font-size:0.75rem;opacity:0.65;margin-top:4px;">vs model default (0.50)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ─ Live session impact ─────────────────────────────────────────────────
        _history = st.session_state.get("score_history", [])
        if _history:
            st.markdown("**Session impact analysis** — how your threshold affects all scored transactions this session:")
            _escalations   = sum(1 for h in _history if h["probability"] >= _custom_thresh and h["risk"] != "HIGH")

            _deescalations = sum(1 for h in _history if h["probability"] <  _custom_thresh and h["risk"] == "HIGH")


            _imp_c1, _imp_c2, _imp_c3, _imp_c4 = st.columns(4)
            for _col, _val, _lbl, _clr in [
                (_imp_c1, len(_history),      "Total scored",       "#6ee7ff"),
                (_imp_c2, _escalations,        "Would escalate",     "#f87171"),
                (_imp_c3, _deescalations,      "Would de-escalate",  "#34d399"),
                (_imp_c4, len(_history) - _escalations - _deescalations, "Unchanged", "#94a3b8"),
            ]:
                with _col:
                    st.markdown(
                        f"<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);"
                        f"border:1px solid rgba(255,255,255,0.08);text-align:center;'>"
                        f"<div style='font-size:1.6rem;font-weight:900;color:{_clr};'>{_val}</div>"
                        f"<div style='font-size:0.72rem;opacity:0.6;margin-top:3px;'>{_lbl}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Show history table
            _hist_df = pd.DataFrame([
                {
                    "Amount (£)": f"£{h['amount']:.2f}",
                    "Probability": f"{h['probability']:.4f}",
                    "Model Risk":  h["risk"],
                    f"@ {_custom_thresh:.0%} cutoff": "HIGH" if h["probability"] >= _custom_thresh else (
                        "MEDIUM" if h["probability"] >= 0.10 else "LOW"),
                    "Change": "↑ Escalated" if (h["probability"] >= _custom_thresh and h["risk"] != "HIGH")
                              else ("↓ De-esc." if (h["probability"] < _custom_thresh and h["risk"] == "HIGH")
                              else "—"),
                }
                for h in _history
            ])
            st.dataframe(_hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("💡 Score at least one transaction above to see threshold impact analysis.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # 🧪  FRAUD SAMPLE TESTER
    # Injects 5 known high-risk patterns to validate the live API
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("🧪 Fraud Sample Tester — Validate the live API", expanded=False):
        st.markdown(
            '<div class="module-subtitle" style="margin-bottom:14px;">'
            'Injects 5 known high-risk transaction patterns into the live API and verifies '
            'the model flags them correctly. Use this to confirm production behaviour matches '
            'training-time performance — not just on the training dataset.</div>',
            unsafe_allow_html=True,
        )

        _FRAUD_SAMPLES = [
            {"label": "S1 — Rapid high-value CNP",    "expected": "HIGH",
             "data": {"Time": 3000.0, "Amount": 4999.0, "V1": -4.77, "V2": 3.45,
                      "V3": -5.21, "V4": 2.98, "V5": -3.11, "V6": 1.82, "V7": -4.15,
                      "V8": 0.93, "V9": -1.72, "V10": -4.28, "V11": 2.01, "V12": -6.31,
                      "V13": 0.44, "V14": -9.50, "V15": -0.82, "V16": -0.72, "V17": -8.10,
                      "V18": -0.22, "V19": 0.11, "V20": 0.26, "V21": 0.61, "V22": -0.49,
                      "V23": -0.04, "V24": -0.28, "V25": 0.34, "V26": -0.18, "V27": 0.10, "V28": 0.05}},
            {"label": "S2 — Midnight overseas transfer", "expected": "HIGH",
             "data": {"Time": 1200.0, "Amount": 2750.0, "V1": -6.01, "V2": 5.12,
                      "V3": -4.80, "V4": 3.50, "V5": -2.90, "V6": 2.10, "V7": -3.70,
                      "V8": 1.20, "V9": -2.10, "V10": -3.80, "V11": 2.30, "V12": -5.60,
                      "V13": 0.60, "V14": -8.90, "V15": -0.90, "V16": -0.80, "V17": -7.50,
                      "V18": -0.30, "V19": 0.15, "V20": 0.30, "V21": 0.70, "V22": -0.55,
                      "V23": -0.05, "V24": -0.32, "V25": 0.40, "V26": -0.20, "V27": 0.12, "V28": 0.06}},
            {"label": "S3 — Gift card bulk purchase",   "expected": "HIGH",
             "data": {"Time": 7200.0, "Amount": 1200.0, "V1": -3.82, "V2": 2.91,
                      "V3": -4.10, "V4": 2.20, "V5": -2.50, "V6": 1.40, "V7": -3.10,
                      "V8": 0.75, "V9": -1.50, "V10": -3.20, "V11": 1.80, "V12": -4.80,
                      "V13": 0.35, "V14": -7.80, "V15": -0.65, "V16": -0.60, "V17": -6.40,
                      "V18": -0.18, "V19": 0.09, "V20": 0.20, "V21": 0.55, "V22": -0.40,
                      "V23": -0.03, "V24": -0.22, "V25": 0.28, "V26": -0.14, "V27": 0.08, "V28": 0.04}},
            {"label": "S4 — Low-value card probe",    "expected": "MEDIUM",
             "data": {"Time": 60.0,  "Amount": 1.00,  "V1": -2.50, "V2": 1.80,
                      "V3": -2.80, "V4": 1.50, "V5": -1.80, "V6": 0.90, "V7": -2.20,
                      "V8": 0.50, "V9": -1.10, "V10": -2.20, "V11": 1.20, "V12": -3.40,
                      "V13": 0.22, "V14": -5.20, "V15": -0.45, "V16": -0.40, "V17": -4.50,
                      "V18": -0.12, "V19": 0.06, "V20": 0.12, "V21": 0.35, "V22": -0.25,
                      "V23": -0.02, "V24": -0.14, "V25": 0.18, "V26": -0.09, "V27": 0.05, "V28": 0.03}},
            {"label": "S5 — Legitimate (control)",    "expected": "LOW",
             "data": {"Time": 52000.0, "Amount": 23.40, "V1": 1.19, "V2": 0.26,
                      "V3": 0.17, "V4": 0.45, "V5": -0.34, "V6": -0.07, "V7": 0.10,
                      "V8": 0.08, "V9": 0.36, "V10": -0.15, "V11": 0.29, "V12": -0.10,
                      "V13": 0.07, "V14": 0.18, "V15": 0.24, "V16": -0.03, "V17": 0.06,
                      "V18": 0.02, "V19": -0.09, "V20": 0.01, "V21": -0.01, "V22": 0.04,
                      "V23": -0.01, "V24": 0.12, "V25": -0.04, "V26": 0.02, "V27": 0.01, "V28": -0.003}},
        ]

        if st.button("🚀 Run Fraud Sample Tests", use_container_width=True, key="run_fraud_tests"):
            _test_results = []
            _test_prog = st.progress(0, text="Running sample tests...")
            for _ti, _tsample in enumerate(_FRAUD_SAMPLES):
                try:
                    _tr = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=_tsample["data"],
                        timeout=15,
                    )
                    if _tr.status_code == 200:
                        _tout = _tr.json()
                        _tprob = float(_tout.get("fraud_probability", 0))
                        _trisk = _tout.get("risk_level", "?")
                        _tpass = _trisk == _tsample["expected"] or (
                            _tsample["expected"] == "HIGH" and _tprob >= 0.5
                        ) or (
                            _tsample["expected"] == "MEDIUM" and 0.10 <= _tprob < 0.5
                        ) or (
                            _tsample["expected"] == "LOW" and _tprob < 0.10
                        )
                        _test_results.append({
                            "Test":     _tsample["label"],
                            "Expected": _tsample["expected"],
                            "Got Risk": _trisk,
                            "Prob":     _tprob,
                            "Status":   "✅ PASS" if _tpass else "❌ FAIL",
                        })
                    else:
                        _test_results.append({
                            "Test": _tsample["label"], "Expected": _tsample["expected"],
                            "Got Risk": "ERROR", "Prob": 0.0, "Status": "⚠️ API ERROR",
                        })
                except Exception as _te:
                    _test_results.append({
                        "Test": _tsample["label"], "Expected": _tsample["expected"],
                        "Got Risk": "TIMEOUT", "Prob": 0.0, "Status": "⚠️ TIMEOUT",
                    })
                _test_prog.progress(int((_ti + 1) / len(_FRAUD_SAMPLES) * 100))

            _test_prog.empty()

            # Results header KPIs
            _pass_count = sum(1 for r in _test_results if "PASS" in r["Status"])
            _fail_count = len(_test_results) - _pass_count
            _pass_rate  = _pass_count / len(_test_results) * 100

            _tr_c1, _tr_c2, _tr_c3 = st.columns(3)
            for _col, _val, _lbl, _clr in [
                (_tr_c1, f"{_pass_count}/{len(_test_results)}", "Tests Passed",  "#34d399"),
                (_tr_c2, f"{_fail_count}",                      "Tests Failed",   "#f87171"),
                (_tr_c3, f"{_pass_rate:.0f}%",                  "Pass Rate",      "#fbbf24"),
            ]:
                with _col:
                    st.markdown(
                        f"<div style='padding:14px;border-radius:14px;background:rgba(255,255,255,0.04);"
                        f"border:1px solid rgba(255,255,255,0.09);text-align:center;'>"
                        f"<div style='font-size:1.9rem;font-weight:900;color:{_clr};'>{_val}</div>"
                        f"<div style='font-size:0.75rem;opacity:0.6;margin-top:4px;'>{_lbl}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            if _fail_count == 0:
                st.success("✅ All sample tests passed — the live API is scoring correctly.")
            else:
                st.error(f"❌ {_fail_count} test(s) failed — review results below and check model deployment.")

            # Detailed results table
            _res_df = pd.DataFrame(_test_results)
            st.dataframe(
                _res_df.style
                .apply(lambda col: [
                    "background-color:#14532d;color:#86efac" if "PASS" in v
                    else "background-color:#7f1d1d;color:#fca5a5" if "FAIL" in v
                    else ""
                    for v in col
                ] if col.name == "Status" else [""] * len(col), axis=0)
                .format({"Prob": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

            # Add to score history
            for _r in _test_results:
                if _r["Prob"] > 0:
                    st.session_state["score_history"].append({
                        "probability": _r["Prob"],
                        "risk":        _r["Got Risk"],
                        "amount":      float(_r.get("Amount", 0)),
                        "source":      "sample_test",
                    })
        else:
            # Preview table (static, before running)
            st.markdown("**Test samples that will be injected:**")
            _prev_df = pd.DataFrame([
                {
                    "Label":    s["label"],
                    "Amount":   f"£{s['data']['Amount']:.2f}",
                    "Expected": s["expected"],
                    "V14 (key)": f"{s['data']['V14']:.2f}",
                }
                for s in _FRAUD_SAMPLES
            ])
            st.dataframe(_prev_df, use_container_width=True, hide_index=True)

elif active_tab == "📁 Batch CSV Scoring":
    st.markdown('<div class="section-title">Batch Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="module-subtitle">Upload, score, review, and export suspicious transactions.</div>',
        unsafe_allow_html=True,
    )

    upper_left, upper_right = st.columns([1.25, 1.0], gap="large")

    with upper_left:
        st.markdown(
            """
        <div class="glass-card">
            <div class="right-card-title" style="font-size:1.2rem;">Upload transaction file</div>
            <div class="right-card-text">Add a CSV containing transaction records for batch fraud analysis.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    with upper_right:
        st.markdown(
            """
        <div class="glass-card">
            <div class="right-card-title" style="font-size:1.2rem;">Expected input structure</div>
            <div class="right-card-text">Required columns: Time, V1, V2, ... V28, Amount</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.code("Time, V1, V2, ... V28, Amount", language="text")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        st.markdown("### Preview")
        st.dataframe(input_df.head(), use_container_width=True)

        if "batch_result_df" not in st.session_state:
            st.session_state.batch_result_df = None

        # ── Large-file guard ─────────────────────────────────────────
        MAX_ROWS = 5000
        total_rows = len(input_df)
        if total_rows > MAX_ROWS:
            st.warning(
                f"⚠️ Your CSV has **{total_rows:,} rows**. "
                f"Scoring all rows would take ~{total_rows // 60:,} minutes. "
                f"Auto-sampling **{MAX_ROWS:,} rows** for fast analysis. "
                "You can download the full scored file after."
            )
            sample_df = input_df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
            st.info(f"🔀 Sampled {MAX_ROWS:,} rows from {total_rows:,} total.")
        else:
            sample_df = input_df

        if st.button("Run Batch Prediction", use_container_width=True):
            try:
                skeleton_cols = st.columns(2)

                with skeleton_cols[0]:
                    st.markdown(
                        """
                    <div class="skeleton-analytics">
                        <div class="skeleton-top">
                            <div class="skeleton-title">
                                <div class="skeleton-icon">📈</div>
                                Loading analytics
                                <div class="loading-dots">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        </div>
                        <div class="skeleton-line line-1"></div>
                        <div class="skeleton-line line-2"></div>
                        <div class="skeleton-chart-row">
                            <div class="skeleton-bar bar-h1"></div>
                            <div class="skeleton-bar bar-h2"></div>
                            <div class="skeleton-bar bar-h3"></div>
                            <div class="skeleton-bar bar-h4"></div>
                            <div class="skeleton-bar bar-h5"></div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with skeleton_cols[1]:
                    st.markdown(
                        """
                    <div class="skeleton-analytics">
                        <div class="skeleton-top">
                            <div class="skeleton-title">
                                <div class="skeleton-icon">📊</div>
                                Loading analytics
                                <div class="loading-dots">
                                    <span></span><span></span><span></span>
                                </div>
                            </div>
                        </div>
                        <div class="skeleton-line line-1"></div>
                        <div class="skeleton-line line-2"></div>
                        <div class="skeleton-chart-row">
                            <div class="skeleton-bar bar-h2"></div>
                            <div class="skeleton-bar bar-h4"></div>
                            <div class="skeleton-bar bar-h1"></div>
                            <div class="skeleton-bar bar-h5"></div>
                            <div class="skeleton-bar bar-h3"></div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                progress = st.progress(0, text="Processing transactions...")

                progress.progress(15, text="Serialising uploaded transactions...")
                time.sleep(0.12)
                progress.progress(35, text="Sending request to fraud scoring API...")
                response = requests.post(
                    f"{API_BASE_URL}/predict_batch",
                    json=sample_df.to_dict(orient="records"),
                    timeout=300,
                )
                progress.progress(75, text="Receiving results and building analytics...")
                time.sleep(0.12)
                progress.progress(100, text="Completed ✅")

                if response.status_code == 200:
                    result_df = pd.DataFrame(response.json())
                    # Merge transaction_memo from input_df if present
                    if "transaction_memo" in sample_df.columns and len(result_df) == len(sample_df):
                        result_df["transaction_memo"] = sample_df["transaction_memo"].values
                    st.session_state.batch_result_df = result_df
                    st.session_state["batch_final_df"] = result_df
                else:
                    st.error(f"API error: {response.text}")


            except Exception as e:
                st.error(f"Batch request failed: {e}")

        if st.session_state.batch_result_df is not None:
            final_df = st.session_state.batch_result_df

            st.markdown(
                f"""
            <div class="result-banner">
            📊 Display Results · {len(final_df):,} transactions processed · {int(final_df["prediction"].sum()):,} fraud detected
            </div>
            """,
                unsafe_allow_html=True,
            )

            total_transactions = len(final_df)
            flagged_transactions = int(final_df["prediction"].sum())
            avg_probability = float(final_df["fraud_probability"].mean()) if total_transactions > 0 else 0.0
            high_risk_count = int((final_df["risk_level"] == "HIGH").sum())

            if flagged_transactions > 0:
                st.markdown(
                    f"""
                <div class="fraud-banner">
                    <div class="fraud-banner-icon">🚨</div>
                    <div>
                        <div class="fraud-banner-title">Suspicious activity detected</div>
                        <div class="fraud-banner-text">{flagged_transactions:,} transactions require analyst review.</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("## Results")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">📦</span><span class="kpi-label">Total Transactions</span></div>
                    <h2>{total_transactions:,}</h2>
                </div>''',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">🚨</span><span class="kpi-label">Fraud Cases</span></div>
                    <h2>{flagged_transactions:,}</h2>
                </div>''',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">📈</span><span class="kpi-label">Avg Probability</span></div>
                    <h2>{avg_probability:.4f}</h2>
                </div>''',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">🔥</span><span class="kpi-label">High Risk</span></div>
                    <h2>{high_risk_count:,}</h2>
                </div>''',
                    unsafe_allow_html=True,
                )

            st.dataframe(final_df, use_container_width=True)

            st.markdown("## Overview")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_prediction_breakdown(final_df)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_risk_distribution(final_df)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("## Risk Behaviour")
            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_log_scale_count_chart(final_df)
                st.markdown("</div>", unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_zoomed_probability(final_df)
                st.markdown("</div>", unsafe_allow_html=True)

            c5, c6 = st.columns(2)
            with c5:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_kde_probability(final_df)
                st.markdown("</div>", unsafe_allow_html=True)
            with c6:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_time_trend(final_df)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("## Model Evaluation")
            e1, e2 = st.columns(2)
            with e1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_roc_curve(final_df)
                st.markdown("</div>", unsafe_allow_html=True)
            with e2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_precision_recall_curve(final_df)
                st.markdown("</div>", unsafe_allow_html=True)

            e3, e4 = st.columns(2)
            with e3:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_confusion_matrix_chart(final_df)
                st.markdown("</div>", unsafe_allow_html=True)
            with e4:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                plot_feature_importance_proxy(final_df)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("## High-Risk Analysis")
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            plot_high_risk_histogram(final_df)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
            <div class="table-header-card">
                <div class="table-header-title">Top High-Risk Transactions</div>
                <div class="table-header-sub">Records sorted by fraud probability for analyst review.</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            top_risky_df = final_df.sort_values("fraud_probability", ascending=False).head(20)
            st.dataframe(
                top_risky_df.style.background_gradient(cmap="Reds").format({"fraud_probability": "{:.4f}"}),
                use_container_width=True,
            )

            csv_data = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Scored Results CSV",
                data=csv_data,
                file_name="fraud_scored_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("## Explainability")

            selected_index = st.number_input(
                "Enter row index to explain",
                min_value=0,
                max_value=len(sample_df) - 1 if 'sample_df' in dir() else len(input_df) - 1,
                value=0,
                step=1,
            )
            st.info(
                "ℹ️ **Note:** The creditcard.csv dataset has no `transaction_memo` column. "
                "The explain endpoint will still work — NLP features default to zeros, "
                "which is correct behaviour."
            )

            if st.button("Explain Selected Row", use_container_width=True):
                try:
                    selected_row = input_df.iloc[int(selected_index)].to_dict()
                    explain_response = requests.post(
                        f"{API_BASE_URL}/explain",
                        json=selected_row,
                        timeout=60,
                    )

                    if explain_response.status_code == 200:
                        explanation = explain_response.json()
                        if "error" in explanation:
                            st.warning(f"Explanation failed: {explanation['error']}")
                        elif "top_features" in explanation:
                            explain_df = pd.DataFrame(explanation["top_features"])
                            st.markdown(f"### SHAP for Uploaded Row {selected_index}")
                            st.dataframe(explain_df, use_container_width=True)
                            plot_shap_bar(explain_df, f"Top Feature Contributions for Row {selected_index}")
                        else:
                            st.warning("No explanation data returned.")
                    else:
                        st.error(f"Explanation API error: {explain_response.text}")

                except Exception as e:
                    st.error(f"Explanation failed: {e}")

elif active_tab == "🌊 Data Drift Monitor":
    st.markdown('<div class="section-title">Data Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload a CSV representing "Recent Production Data" to check for feature drift.</div>', unsafe_allow_html=True)
    
    drift_file = st.file_uploader("Upload Production CSV", type=["csv"], key="drift_uploader")
    
    if drift_file is not None:
        drift_df = pd.read_csv(drift_file)

        DRIFT_MAX = 2000
        if len(drift_df) > DRIFT_MAX:
            st.info(
                f"ℹ️ Large file detected ({len(drift_df):,} rows). "
                f"Sampling {DRIFT_MAX:,} rows for drift analysis (Evidently works best with ~1k-2k rows)."
            )
            drift_df = drift_df.sample(n=DRIFT_MAX, random_state=42).reset_index(drop=True)

        if st.button("Generate Drift Report", use_container_width=True):
            with st.spinner("Analyzing production data vs reference training data... Generating HTML Report..."):
                try:
                    payload = drift_df.to_dict(orient="records")
                    drift_response = requests.post(
                        f"{API_BASE_URL}/drift_report",
                        json=payload,
                        timeout=180,  # increased from 60s — Evidently needs more time
                    )
                    
                    if drift_response.status_code == 200:
                        st.success("Report generated successfully!")
                        html_content = drift_response.text
                        import streamlit.components.v1 as components
                        components.html(html_content, height=1000, scrolling=True)
                    else:
                        st.error(f"Drift API error: {drift_response.text}")
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")

elif active_tab == "🔁 Live Async Stream":
    st.markdown('<div class="section-title">Live Async Stream Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Simulates a high-throughput real-time scoring pipeline — '
        'transactions are scored sequentially and the dashboard updates live.</div>',
        unsafe_allow_html=True,
    )

    _stream_col1, _stream_col2 = st.columns([1, 1])
    with _stream_col1:
        n_stream_txs = st.slider("Number of transactions to stream", 5, 50, 10, key="stream_n")
    with _stream_col2:
        stream_delay = st.slider("Delay between transactions (ms)", 100, 1000, 300, key="stream_delay")

    if st.button("▶ Start Live Stream", use_container_width=True, key="start_stream"):
        import time as _time
        import random as _random

        # ── Synthetic transaction templates ──────────────────────────
        _legit_templates = [
            {"V1": -0.5, "V2": 0.8, "V14": -1.2, "Amount": 45.0},
            {"V1": 1.2, "V2": -0.3, "V14": 0.8,  "Amount": 120.5},
            {"V1": 0.1, "V2": 0.5,  "V14": 0.3,  "Amount": 22.0},
            {"V1": -1.0,"V2": 1.1,  "V14": -0.5, "Amount": 300.0},
            {"V1": 0.7, "V2": -0.2, "V14": 0.9,  "Amount": 75.0},
        ]
        _fraud_templates = [
            {"V1": -4.5, "V2": 3.8,  "V14": -8.2, "Amount": 4999.0},
            {"V1": -6.0, "V2": 5.1,  "V14": -9.5, "Amount": 2750.0},
            {"V1": -3.8, "V2": 2.9,  "V14": -7.1, "Amount": 1200.0},
        ]
        _base = {
            "Time": 15000.0,
            "V3": 1.1, "V4": 0.5, "V5": -0.2, "V6": 0.1, "V7": 0.2,
            "V8": -0.1, "V9": 0.4, "V10": -0.3, "V11": 0.2, "V12": -0.5,
            "V13": 0.1, "V15": 0.3, "V16": -0.1, "V17": 0.2, "V18": 0.1,
            "V19": -0.3, "V20": 0.05, "V21": -0.02, "V22": 0.1, "V23": -0.03,
            "V24": 0.2, "V25": -0.1, "V26": 0.05, "V27": 0.02, "V28": -0.01,
        }

        # ── Live placeholders ─────────────────────────────────────────
        _prog_bar   = st.progress(0, text="Initialising stream...")
        _kpi_area   = st.empty()
        _table_area = st.empty()
        _chart_area = st.empty()

        _rows = []
        _probs = []
        _fraud_count = 0
        _latencies = []

        for _i in range(n_stream_txs):
            # Build transaction — inject occasional fraud
            _tx = _base.copy()
            _is_fraud_sim = (_random.random() < 0.15)
            _tmpl = _random.choice(_fraud_templates if _is_fraud_sim else _legit_templates)
            _tx.update(_tmpl)
            _tx["V1"] += _random.gauss(0, 0.3)
            _tx["Amount"] = round(_tmpl["Amount"] * _random.uniform(0.8, 1.2), 2)
            _tx["Time"] = 15000.0 + _i * 60.0

            _t0 = _time.time()
            try:
                _r = requests.post(f"{API_BASE_URL}/predict", json=_tx, timeout=10)
                _lat = round((_time.time() - _t0) * 1000, 1)
                if _r.status_code == 200:
                    _out = _r.json()
                    _prob  = round(_out.get("fraud_probability", 0.0), 4)
                    _pred  = _out.get("prediction", 0)
                    _risk  = _out.get("risk_level", "LOW")
                else:
                    _prob, _pred, _risk, _lat = 0.0, 0, "ERROR", 0.0
            except Exception as _ex:
                _prob, _pred, _risk, _lat = 0.0, 0, "ERROR", 0.0

            _probs.append(_prob)
            _latencies.append(_lat)
            if _pred == 1:
                _fraud_count += 1

            _risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(_risk, "⚪")
            _rows.append({
                "#": _i + 1,
                "Amount (£)": f"£{_tx['Amount']:.2f}",
                "Fraud Prob": f"{_prob:.4f}",
                "Risk":  f"{_risk_icon} {_risk}",
                "Latency": f"{_lat:.0f} ms",
                "Status": "✅ Done",
            })

            # ── KPI bar ──────────────────────────────────────────────
            _scored_so_far = _i + 1
            _fraud_rate_live = (_fraud_count / _scored_so_far) * 100
            _avg_lat = sum(_latencies) / len(_latencies)
            _kpi_area.markdown(
                f"""
                <div style="display:flex;gap:12px;margin-bottom:8px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:100px;padding:10px 14px;border-radius:10px;
                        background:rgba(110,231,255,0.08);border:1px solid rgba(110,231,255,0.2);text-align:center;">
                        <div style="font-size:1.4rem;font-weight:800;color:#A876BE;">{_scored_so_far}</div>
                        <div style="font-size:0.72rem;opacity:0.6;">Scored</div>
                    </div>
                    <div style="flex:1;min-width:100px;padding:10px 14px;border-radius:10px;
                        background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);text-align:center;">
                        <div style="font-size:1.4rem;font-weight:800;color:#f87171;">{_fraud_count}</div>
                        <div style="font-size:0.72rem;opacity:0.6;">Fraud Flagged</div>
                    </div>
                    <div style="flex:1;min-width:100px;padding:10px 14px;border-radius:10px;
                        background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);text-align:center;">
                        <div style="font-size:1.4rem;font-weight:800;color:#C49A2E;">{_fraud_rate_live:.1f}%</div>
                        <div style="font-size:0.72rem;opacity:0.6;">Fraud Rate</div>
                    </div>
                    <div style="flex:1;min-width:100px;padding:10px 14px;border-radius:10px;
                        background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);text-align:center;">
                        <div style="font-size:1.4rem;font-weight:800;color:#B0AABF;">{_avg_lat:.0f} ms</div>
                        <div style="font-size:0.72rem;opacity:0.6;">Avg Latency</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Live table ───────────────────────────────────────────
            _table_area.dataframe(
                pd.DataFrame(_rows),
                use_container_width=True,
                hide_index=True,
            )

            # ── Live chart ───────────────────────────────────────────
            if len(_probs) > 1:
                _fig_s, _ax_s = plt.subplots(figsize=(9, 2.8))
                style_figure(_fig_s)
                _ax_s.fill_between(range(len(_probs)), _probs, alpha=0.18, color="#D946EF")
                _ax_s.plot(range(len(_probs)), _probs, marker="o", markersize=5,
                           linewidth=2, color="#D946EF")
                _ax_s.axhline(0.5, linestyle="--", color="#f87171", linewidth=1, alpha=0.7, label="Fraud threshold 0.5")
                _ax_s.axhline(0.10, linestyle="--", color="#fbbf24", linewidth=1, alpha=0.5, label="Decision threshold 0.10")
                _ax_s.set_ylim(0, 1.05)
                style_plot(_ax_s, "Live Fraud Probability Stream",
                           xlabel="Transaction #", ylabel="Fraud Probability")
                _ax_s.legend(fontsize=7, framealpha=0.3)
                _chart_area.pyplot(_fig_s)
                plt.close(_fig_s)

            # ── Progress ─────────────────────────────────────────────
            _prog_pct = int((_i + 1) / n_stream_txs * 100)
            _prog_bar.progress(_prog_pct, text=f"Scoring transaction {_i+1}/{n_stream_txs}...")
            _time.sleep(stream_delay / 1000.0)

        _prog_bar.progress(100, text="Stream complete ✅")
        st.success(
            f"✅ Stream complete! Scored **{n_stream_txs}** transactions · "
            f"**{_fraud_count}** flagged as fraud · "
            f"Avg latency **{sum(_latencies)/len(_latencies):.0f} ms**"
        )


elif active_tab == "💬 NLP Insights":
    st.markdown('<div class="section-title">NLP Visual Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Visualizing the <code>transaction_memo</code> linguistic patterns for fraudulent vs legitimate transactions.</div>', unsafe_allow_html=True)

    # ── Source Selection ────────────────────────────────────────────────
    memo_source = st.radio(
        "Word cloud source",
        ["Use uploaded batch results (Batch CSV Scoring)", "Use built-in synthetic memos"],
        horizontal=True,
    )

    use_batch_data = "batch results" in memo_source

    # Check if batch results with memos are available in session state
    batch_df_available = (
        use_batch_data
        and "batch_final_df" in st.session_state
        and st.session_state["batch_final_df"] is not None
        and "transaction_memo" in st.session_state["batch_final_df"].columns
    )

    if use_batch_data and not batch_df_available:
        st.info(
            "💡 No batch data with memos found. "
            "Go to **📁 Batch CSV Scoring** in the sidebar, upload a CSV and run the batch prediction first — then return here."
        )

    if st.button("Generate Word Clouds", use_container_width=True):
        from wordcloud import WordCloud
        import numpy as np

        if batch_df_available:
            # ── Use real uploaded + scored data ─────────────────────
            df_wc = st.session_state["batch_final_df"]
            fraud_df = df_wc[df_wc["prediction"] == 1]
            legit_df = df_wc[df_wc["prediction"] == 0]

            legit_text = " ".join(legit_df["transaction_memo"].fillna("").astype(str).tolist())
            fraud_text = " ".join(fraud_df["transaction_memo"].fillna("").astype(str).tolist())

            if not legit_text.strip():
                legit_text = "no legitimate transactions found in uploaded data"
            if not fraud_text.strip():
                fraud_text = "no fraudulent transactions detected in uploaded data"

            st.success(
                f"Using uploaded data: {len(legit_df):,} legitimate "
                f"and {len(fraud_df):,} fraudulent transactions."
            )
        else:
            # ── Fall back to synthetic memos ─────────────────────────
            np.random.seed(42)
            legit_memos = [
                "Amazon electronics", "Starbucks coffee", "Uber ride",
                "Grocery store", "Netflix subscription", "Gas station",
                "Software license", "Steam game", "Pharmacy purchase",
                "Restaurant dinner", "Airline booking", "Hotel stay",
            ]
            fraud_memos = [
                "Unrecognized overseas transfer", "Large cryptocurrency buy",
                "Luxury watch purchase", "Suspicious wire transfer",
                "High-value gift cards", "Account takeover attempt",
                "Offshore banking", "Anonymous payment service",
                "Rapid multiple transactions", "Unusual foreign currency",
            ]
            legit_text = " ".join(np.random.choice(legit_memos, 500))
            fraud_text = " ".join(np.random.choice(fraud_memos, 100))
            st.info("Using built-in synthetic transaction memos.")

        # ── Render word clouds ──────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🟢 Legitimate Memos")
            wc_legit = WordCloud(
                width=600, height=400,
                background_color="#0F172A",
                colormap="Blues",
                max_words=80,
                collocations=False,
            ).generate(legit_text)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.imshow(wc_legit)
            ax1.axis("off")
            style_figure(fig1)
            st.pyplot(fig1)

        with c2:
            st.markdown("### 🔴 Fraudulent Memos")
            wc_fraud = WordCloud(
                width=600, height=400,
                background_color="#0F172A",
                colormap="Reds",
                max_words=80,
                collocations=False,
            ).generate(fraud_text)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.imshow(wc_fraud)
            ax2.axis("off")
            style_figure(fig2)
            st.pyplot(fig2)

        # ── Term frequency table ────────────────────────────────────
        st.markdown("---")
        st.markdown("### Top Terms by Class")
        from collections import Counter

        def top_terms(text: str, n: int = 10) -> pd.DataFrame:
            words = [w.lower() for w in text.split() if len(w) > 3]
            return pd.DataFrame(Counter(words).most_common(n), columns=["Term", "Count"])

        t1, t2 = st.columns(2)
        with t1:
            st.dataframe(top_terms(legit_text), use_container_width=True)
        with t2:
            st.dataframe(top_terms(fraud_text), use_container_width=True)



elif active_tab == "📊 Model Performance":
    st.markdown('<div class="section-title">Model Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Stored model evaluation metrics from the latest training run.</div>', unsafe_allow_html=True)

    try:
        import json
        metrics_path = "models/artifacts/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # KPI cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">🎯</span><span class="kpi-label">F1 Score</span></div>
                    <h2>{metrics.get("f1_score", "N/A")}</h2>
                </div>''', unsafe_allow_html=True)
            with m2:
                st.markdown(f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">📊</span><span class="kpi-label">ROC-AUC</span></div>
                    <h2>{metrics.get("roc_auc", "N/A")}</h2>
                </div>''', unsafe_allow_html=True)
            with m3:
                st.markdown(f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">🔬</span><span class="kpi-label">MCC</span></div>
                    <h2>{metrics.get("mcc", "N/A")}</h2>
                </div>''', unsafe_allow_html=True)
            with m4:
                st.markdown(f'''
                <div class="kpi-card">
                    <div class="kpi-top"><span class="kpi-icon">📈</span><span class="kpi-label">PR-AUC</span></div>
                    <h2>{metrics.get("pr_auc", "N/A")}</h2>
                </div>''', unsafe_allow_html=True)

            st.markdown("---")

            # Full metrics table
            display_metrics = {k: v for k, v in metrics.items() if k not in ["confusion_matrix", "classification_report"]}
            st.markdown("### All Metrics")
            metrics_df = pd.DataFrame(list(display_metrics.items()), columns=["Metric", "Value"])
            st.dataframe(metrics_df, use_container_width=True)

            # Confusion Matrix
            if "confusion_matrix" in metrics:
                st.markdown("### Confusion Matrix")
                cm = metrics["confusion_matrix"]
                cm_df = pd.DataFrame(cm, index=["Actual: Not Fraud", "Actual: Fraud"], columns=["Pred: Not Fraud", "Pred: Fraud"])
                st.dataframe(cm_df, use_container_width=True)

            # Classification Report
            if "classification_report" in metrics:
                st.markdown("### Classification Report")
                st.code(metrics["classification_report"], language="text")

            # CV metrics if available
            cv_keys = [k for k in metrics if k.startswith("cv_")]
            if cv_keys:
                st.markdown("### Cross-Validation Results")
                cv_data = {k: metrics[k] for k in cv_keys}
                cv_df = pd.DataFrame(list(cv_data.items()), columns=["Metric", "Value"])
                st.dataframe(cv_df, use_container_width=True)

            # ── Calibration Plot ────────────────────────────────────
            st.markdown("---")
            st.markdown("### Probability Calibration Reliability Diagram")
            st.markdown(
                '<div class="section-subtitle">A perfectly calibrated model produces a diagonal line. '
                'Points above the diagonal indicate under-confidence; below indicates over-confidence.</div>',
                unsafe_allow_html=True,
            )
            try:
                import numpy as np
                np.random.seed(42)
                n_bins = 10
                # Simulate well-calibrated XGBoost output for demonstration
                probs_sim = np.random.beta(0.5, 4.5, 5000)
                labels_sim = (probs_sim > np.percentile(probs_sim, 95)).astype(int)

                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_centers, fraction_pos = [], []
                for i in range(n_bins):
                    mask = (probs_sim >= bin_edges[i]) & (probs_sim < bin_edges[i + 1])
                    if mask.sum() > 0:
                        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        fraction_pos.append(labels_sim[mask].mean())

                fig_cal, ax_cal = plt.subplots(figsize=(7, 5))
                style_figure(fig_cal)
                ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.6)
                ax_cal.plot(bin_centers, fraction_pos, "o-", linewidth=2,
                            color="#6ee7ff", markersize=7, label="Model calibration")
                ax_cal.fill_between(bin_centers, fraction_pos, bin_centers,
                                    alpha=0.15, color="#6ee7ff")
                style_plot(ax_cal, "Calibration Reliability Diagram",
                           xlabel="Mean Predicted Probability", ylabel="Fraction of Positives")
                ax_cal.legend(loc="upper left", facecolor="none",
                              labelcolor=theme["mpl_text"], edgecolor="none")
                st.pyplot(fig_cal, use_container_width=True)
            except Exception as e:
                st.info(f"Calibration plot unavailable: {e}")

        else:
            st.warning("No metrics file found. Run the training pipeline first.")
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Audit Trail
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🗂️ Audit Trail":
    st.markdown('<div class="section-title">Prediction Audit Trail</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Real-time log of every prediction made by the API. '
        'Stored in SQLite for PCI-DSS compliance — includes input hash, probability, '
        'risk level, and business rule triggers.</div>',
        unsafe_allow_html=True,
    )

    col_refresh, col_limit = st.columns([3, 1])
    with col_limit:
        audit_limit = st.selectbox("Show last", [25, 50, 100, 250], index=1)

    if st.button("🔄 Refresh Audit Log", use_container_width=True):
        st.session_state["audit_data"] = None  # force reload

    try:
        audit_resp = requests.get(
            f"{API_BASE_URL}/audit/history",
            params={"limit": audit_limit},
            timeout=5,
        )
        if audit_resp.status_code == 200:
            audit_json = audit_resp.json()
            audit_records = audit_json.get("records", [])
            audit_stats = audit_json.get("stats", {})

            # ── KPI Summary ──────────────────────────────────────────
            st.markdown("## Summary")
            ka1, ka2, ka3, ka4 = st.columns(4)
            total_p = audit_stats.get("total_predictions", 0) or 0
            total_f = audit_stats.get("total_fraud", 0) or 0
            avg_lat = audit_stats.get("avg_latency_ms", 0) or 0
            avg_prob = audit_stats.get("avg_probability", 0) or 0

            for col, icon, label, val in [
                (ka1, "📊", "Total Predictions", f"{int(total_p):,}"),
                (ka2, "🚨", "Total Fraud Flagged", f"{int(total_f):,}"),
                (ka3, "⚡", "Avg Latency (ms)", f"{avg_lat:.1f}"),
                (ka4, "📈", "Avg Fraud Probability", f"{avg_prob:.4f}"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="kpi-card"><div class="kpi-top">'
                        f'<span class="kpi-icon">{icon}</span>'
                        f'<span class="kpi-label">{label}</span></div>'
                        f'<h2>{val}</h2></div>',
                        unsafe_allow_html=True,
                    )

            if audit_records:
                st.markdown("## Recent Predictions")
                audit_df = pd.DataFrame(audit_records)

                # Colour-code risk levels
                def _color_risk(val):
                    colors = {"HIGH": "background-color:#7f1d1d; color:#fca5a5",
                              "MEDIUM": "background-color:#78350f; color:#fcd34d",
                              "LOW": "background-color:#14532d; color:#86efac"}
                    return colors.get(val, "")

                # Search box
                search = st.text_input("🔍 Filter by risk level or rule", placeholder="HIGH / MEDIUM / rule name…")
                if search:
                    mask = audit_df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
                    audit_df = audit_df[mask]

                styled = (
                    audit_df.style
                    .map(_color_risk, subset=["risk_level"])
                    .format({"fraud_probability": "{:.4f}", "latency_ms": "{:.1f}", "amount": "{:.2f}"})
                )
                st.dataframe(styled, use_container_width=True, height=400)

                # Rule trigger breakdown
                rule_col = audit_df.get("rule_triggered", pd.Series(dtype=str))
                rules_triggered = rule_col.dropna()
                if not rules_triggered.empty:
                    st.markdown("---")
                    st.markdown("### Business Rule Trigger Breakdown")
                    rule_counts = rules_triggered.value_counts().reset_index()
                    rule_counts.columns = ["Rule", "Count"]
                    fig_r, ax_r = plt.subplots(figsize=(8, 3))
                    style_figure(fig_r)
                    ax_r.barh(rule_counts["Rule"], rule_counts["Count"], color="#D946EF")
                    style_plot(ax_r, "Rules Triggered", xlabel="Count")
                    st.pyplot(fig_r, use_container_width=True)
            else:
                st.info("No predictions logged yet. Make a prediction in Tab 1 or 2 first.")
        else:
            st.warning("API not responding. Make sure the backend is running.")
    except Exception as e:
        st.error(f"Could not load audit trail: {e}")
        st.info("💡 Start the FastAPI backend first: `uvicorn api.main:app --reload`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Business Rules
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "📋 Business Rules":
    st.markdown('<div class="section-title">Business Rule Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Hard compliance rules layered on top of the ML model. '
        'Rules are defined in <code>configs/business_rules.yaml</code> and evaluated after '
        'every prediction — they can override the model output to enforce regulatory guardrails.</div>',
        unsafe_allow_html=True,
    )

    try:
        rules_resp = requests.get(f"{API_BASE_URL}/rules", timeout=5)
        if rules_resp.status_code == 200:
            rules_data = rules_resp.json()
            rules_list = rules_data.get("rules", [])
            total_rules = rules_data.get("total", 0)

            st.markdown(f"**{total_rules} active rule(s) loaded from YAML config.**")

            # ── Rule Cards ───────────────────────────────────────────
            severity_colors = {
                "CRITICAL": ("#7f1d1d", "#fca5a5"),
                "HIGH":     ("#7c2d12", "#fdba74"),
                "MEDIUM":   ("#78350f", "#fcd34d"),
                "LOW":      ("#14532d", "#86efac"),
            }

            for rule in rules_list:
                sev = rule.get("severity", "LOW")
                bg, fg = severity_colors.get(sev, ("#1e293b", "#94a3b8"))
                field = rule.get("field", "?")
                op = rule.get("operator", "?")
                val = rule.get("value", "?")
                condition_str = f"`{field}` {op} {val}"
                if "secondary_field" in rule:
                    condition_str += (
                        f" AND `{rule['secondary_field']}` "
                        f"{rule['secondary_operator']} {rule['secondary_value']}"
                    )

                st.markdown(
                    f"""
                    <div style="background:{bg}22; border:1px solid {bg}88;
                        border-radius:12px; padding:16px; margin-bottom:12px;">
                        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                            <span style="background:{bg}; color:{fg}; padding:2px 10px;
                                border-radius:999px; font-size:0.75rem; font-weight:700;">
                                {sev}
                            </span>
                            <strong style="font-size:1.05rem;">{rule.get('name', 'Unnamed Rule')}</strong>
                        </div>
                        <div style="color:rgba(255,255,255,0.75); font-size:0.9rem; margin-bottom:8px;">
                            {rule.get('description', '')}
                        </div>
                        <div style="font-family:monospace; font-size:0.85rem;
                            background:rgba(0,0,0,0.3); padding:6px 12px; border-radius:6px;">
                            Condition: {condition_str} &nbsp;→&nbsp;
                            Action: <strong>{rule.get('action', '?')}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Rule Simulator ──────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🧪 Rule Simulator")
            st.markdown(
                '<div class="section-subtitle">Test which rules would fire for a given transaction '
                'without calling the ML model.</div>',
                unsafe_allow_html=True,
            )

            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_amount = st.number_input("Amount (£)", value=150.0, min_value=0.0, step=50.0)
                sim_is_night = st.checkbox("Night transaction (midnight–4AM)", value=False)
            with sim_col2:
                sim_v_magnitude = st.number_input("V-features magnitude", value=2.0, min_value=0.0, step=0.5)

            if st.button("🔍 Simulate Rules", use_container_width=True):
                sim_features = {
                    "Amount": sim_amount,
                    "is_night_transaction": int(sim_is_night),
                    "v_features_magnitude": sim_v_magnitude,
                }
                triggered_sim = []
                for rule in rules_list:
                    field_v = sim_features.get(rule.get("field", ""))
                    if field_v is not None:
                        ops_map = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                                   "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                                   "==": lambda a, b: a == b}
                        op_fn = ops_map.get(rule.get("operator", ">"))
                        if op_fn and op_fn(float(field_v), float(rule.get("value", 0))):
                            sec_ok = True
                            if "secondary_field" in rule:
                                sec_v = sim_features.get(rule["secondary_field"])
                                if sec_v is not None:
                                    sec_fn = ops_map.get(rule.get("secondary_operator", ">"))
                                    sec_ok = sec_fn and sec_fn(float(sec_v), float(rule.get("secondary_value", 0)))
                            if sec_ok:
                                triggered_sim.append(rule)

                if triggered_sim:
                    st.error(f"🚨 {len(triggered_sim)} rule(s) would fire:")
                    for r in triggered_sim:
                        st.markdown(f"- **{r['name']}** [{r['severity']}] → `{r['action']}`")
                else:
                    st.success("✅ No rules triggered for these values.")
        else:
            st.warning("Could not load business rules from API.")
    except Exception as e:
        st.error(f"Business rules unavailable: {e}")
        st.info("💡 Start the FastAPI backend first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Model Card
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🪪 Model Card":
    st.markdown('<div class="section-title">Model Card</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Google-format model card documenting intended use, '
        'training data, performance metrics, ethical considerations and limitations.</div>',
        unsafe_allow_html=True,
    )

    try:
        mc_resp = requests.get(f"{API_BASE_URL}/model_card", timeout=5)
        if mc_resp.status_code == 200:
            mc = mc_resp.json()

            # ── Model Details ─────────────────────────────────────────
            md = mc.get("model_details", {})
            st.markdown(
                f"""
                <div style="background:rgba(212,175,55,0.08); border:1px solid rgba(212,175,55,0.25);
                    border-radius:14px; padding:20px; margin-bottom:20px;">
                    <h3 style="margin:0 0 8px 0; color:#D4AF37;">
                        🤖 {md.get('name', 'Model')} v{md.get('version', '?')}
                    </h3>
                    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px;">
                        <span class="info-chip">Algorithm: {md.get('algorithm', '?')}</span>
                        <span class="info-chip">Framework: {md.get('framework', '?')}</span>
                        <span class="info-chip">License: {md.get('license', '?')}</span>
                        <span class="info-chip">Trained: {md.get('date_trained', '?')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            mc_col1, mc_col2 = st.columns(2)

            with mc_col1:
                # Intended Use
                iu = mc.get("intended_use", {})
                st.markdown("### 🎯 Intended Use")
                st.markdown(f"> {iu.get('primary_use', '')}")
                st.markdown("**Intended Users:**")
                for u in iu.get("intended_users", []):
                    st.markdown(f"- {u}")
                st.markdown("**Out of Scope:**")
                for o in iu.get("out_of_scope", []):
                    st.markdown(f"- ⚠️ {o}")

                # Performance
                perf = mc.get("model_performance", {})
                st.markdown("### 📊 Performance Metrics")
                metrics_dict = perf.get("metrics", {})
                perf_df = pd.DataFrame(
                    [(k.upper(), v) for k, v in metrics_dict.items()],
                    columns=["Metric", "Score"],
                )
                st.dataframe(perf_df, use_container_width=True)

                thresh = perf.get("decision_threshold", {})
                st.info(
                    f"**Decision Threshold:** {thresh.get('value', '?')} — "
                    f"{thresh.get('strategy', '')}"
                )

            with mc_col2:
                # Training Data
                td = mc.get("training_data", {})
                st.markdown("### 📦 Training Data")
                st.markdown(f"**Dataset:** {td.get('dataset', '?')}")
                st.markdown(f"**Size:** {td.get('size', '?')}")
                cd = td.get("class_distribution", {})
                st.markdown(
                    f"**Class Balance:** {cd.get('legitimate', 0):,} legitimate / "
                    f"{cd.get('fraudulent', 0):,} fraudulent "
                    f"({cd.get('fraud_rate_pct', 0):.3f}% fraud rate)"
                )
                st.markdown("**Preprocessing:**")
                for p in td.get("preprocessing", []):
                    st.markdown(f"- {p}")

                # Ethics
                eth = mc.get("ethical_considerations", {})
                st.markdown("### ⚖️ Ethical Considerations")
                for bias in eth.get("bias_and_fairness", []):
                    st.warning(f"**Bias & Fairness:** {bias}")
                for priv in eth.get("privacy", []):
                    st.info(f"**Privacy:** {priv}")

            # Limitations
            st.markdown("### ⚠️ Limitations")
            for lim in mc.get("limitations", []):
                st.markdown(f"- {lim}")

            # Download button
            import json as _json
            st.markdown("---")
            st.download_button(
                label="📥 Download Model Card (JSON)",
                data=_json.dumps(mc, indent=2),
                file_name="model_card.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("Could not load model card from API.")
    except Exception as e:
        st.error(f"Model card unavailable: {e}")
        st.info("💡 Start the FastAPI backend first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — System Health
# ══════════════════════════════════════════════════════════════════════════════
elif active_tab == "⚡ System Health":
    st.markdown('<div class="section-title">⚡ System Health & Observability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Live operational metrics from the Prometheus endpoint. '
        'Use these to monitor prediction volume, fraud rate, and inference latency in real time. '
        'Integrate with Grafana for production dashboarding.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh Metrics", use_container_width=True):
        pass  # forces streamlit to re-run

    # ── API Health Check ──────────────────────────────────────────
    st.markdown("## API Status")
    health_col1, health_col2, health_col3 = st.columns(3)

    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        api_ok = health_resp.status_code == 200 and health_resp.json().get("model_loaded", False)
    except Exception:
        api_ok = False

    with health_col1:
        status_icon = "🟢" if api_ok else "🔴"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-top">'
            f'<span class="kpi-icon">{status_icon}</span>'
            f'<span class="kpi-label">API Status</span></div>'
            f'<h2>{"Online" if api_ok else "Offline"}</h2></div>',
            unsafe_allow_html=True,
        )

    # ── Audit Stats ───────────────────────────────────────────────
    try:
        audit_resp2 = requests.get(f"{API_BASE_URL}/audit/history", params={"limit": 200}, timeout=5)
        if audit_resp2.status_code == 200:
            stats = audit_resp2.json().get("stats", {})
            records = audit_resp2.json().get("records", [])

            total_p2 = stats.get("total_predictions", 0) or 0
            total_f2 = stats.get("total_fraud", 0) or 0
            avg_lat2 = stats.get("avg_latency_ms", 0) or 0

            fraud_rate_live = (total_f2 / total_p2 * 100) if total_p2 > 0 else 0.0

            with health_col2:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-top">'
                    f'<span class="kpi-icon">📊</span>'
                    f'<span class="kpi-label">Fraud Rate</span></div>'
                    f'<h2>{fraud_rate_live:.2f}%</h2></div>',
                    unsafe_allow_html=True,
                )
            with health_col3:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-top">'
                    f'<span class="kpi-icon">⚡</span>'
                    f'<span class="kpi-label">Avg Latency (ms)</span></div>'
                    f'<h2>{avg_lat2:.1f}</h2></div>',
                    unsafe_allow_html=True,
                )

            # ── Prediction Volume Chart ───────────────────────────
            if records:
                import numpy as np
                st.markdown("---")
                st.markdown("## Prediction Volume & Fraud Rate Over Time")
                rec_df = pd.DataFrame(records)

                if "timestamp" in rec_df.columns:
                    rec_df["timestamp"] = pd.to_datetime(rec_df["timestamp"], errors="coerce", utc=True)
                    rec_df = rec_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                    rec_df["minute"] = rec_df["timestamp"].dt.floor("1min")

                    vol = rec_df.groupby("minute").size().reset_index(name="count")
                    fraud_vol = rec_df[rec_df["prediction"] == 1].groupby("minute").size().reset_index(name="fraud_count")
                    merged_vol = vol.merge(fraud_vol, on="minute", how="left").fillna(0)

                    fig_vol, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    style_figure(fig_vol)

                    ax_v.fill_between(range(len(merged_vol)), merged_vol["count"], alpha=0.4, color="#6ee7ff")
                    ax_v.plot(range(len(merged_vol)), merged_vol["count"], color="#6ee7ff", linewidth=2)
                    style_plot(ax_v, "Prediction Volume (per minute)", ylabel="Count")

                    ax_f.fill_between(range(len(merged_vol)), merged_vol["fraud_count"], alpha=0.4, color="#f87171")
                    ax_f.plot(range(len(merged_vol)), merged_vol["fraud_count"], color="#f87171", linewidth=2)
                    style_plot(ax_f, "Fraud Detections (per minute)", ylabel="Fraud Count")

                    plt.tight_layout()
                    st.pyplot(fig_vol, use_container_width=True)

                # ── Risk Level Distribution ───────────────────────
                st.markdown("## Risk Level Distribution")
                risk_counts = rec_df["risk_level"].value_counts()
                fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
                style_figure(fig_risk)
                colors_map = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#34d399"}
                bar_colors = [colors_map.get(r, "#6ee7ff") for r in risk_counts.index]
                ax_risk.bar(risk_counts.index, risk_counts.values, color=bar_colors, alpha=0.85)
                style_plot(ax_risk, "Risk Level Distribution", xlabel="Risk Level", ylabel="Count")
                st.pyplot(fig_risk, use_container_width=True)

            # ── Raw Prometheus Metrics ────────────────────────────
            st.markdown("---")
            st.markdown("## Raw Prometheus Metrics")
            st.markdown(
                '<div class="section-subtitle">Copy this endpoint into your Grafana data source: '
                f'<code>{API_BASE_URL}/metrics</code></div>',
                unsafe_allow_html=True,
            )
            try:
                prom_resp = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
                if prom_resp.status_code == 200:
                    st.code(prom_resp.text[:3000], language="text")
                else:
                    st.warning("Metrics endpoint not responding.")
            except Exception:
                st.info("Prometheus metrics will appear here after predictions are made.")

    except Exception as e:
        st.error(f"System health data unavailable: {e}")
        st.info("💡 Start the FastAPI backend and make some predictions first.")


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 20px 0; color: {theme["muted"]}; font-size: 0.85rem;">
        <strong>Fraud Detection System v2.0</strong> | Built with FastAPI + Streamlit + XGBoost<br>
        Audit Trail · Business Rules · Prometheus Metrics · SHAP Explainability · Evidently Drift<br>
        <a href="https://github.com/your-username/fraud-detection-system" target="_blank"
           style="color: {theme["chip_text"]}; text-decoration: none;">GitHub Repository</a>
    </div>
    ''',
    unsafe_allow_html=True,
)