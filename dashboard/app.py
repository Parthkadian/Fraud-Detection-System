import sys
import os
import webbrowser
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
IS_DOCKER = os.getenv("DOCKER_ENV", "0") == "1"


def open_browser():
    webbrowser.open_new("http://localhost:8501")


if os.getenv("OPEN_BROWSER") == "1":
    threading.Timer(2, open_browser).start()

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"


def get_theme_tokens(mode: str):
    if mode == "light":
        return {
            "text": "#0F172A",
            "muted": "rgba(15,23,42,0.72)",
            "card": "rgba(255,255,255,0.82)",
            "card_border": "rgba(15,23,42,0.08)",
            "sidebar": "rgba(255,255,255,0.92)",
            "sidebar_border": "rgba(15,23,42,0.08)",
            "input_bg": "rgba(255,255,255,0.96)",
            "input_border": "rgba(15,23,42,0.10)",
            "tab_bg": "rgba(15,23,42,0.04)",
            "accent1": "rgba(0,198,255,0.24)",
            "accent2": "rgba(255,0,153,0.18)",
            "hero_bg": "linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.88))",
            "hero_shadow": "0 10px 40px rgba(15,23,42,0.10)",
            "body_gradient": """
                radial-gradient(circle at 10% 15%, rgba(0, 217, 245, 0.12), transparent 20%),
                radial-gradient(circle at 85% 12%, rgba(255, 90, 205, 0.10), transparent 22%),
                radial-gradient(circle at 50% 85%, rgba(0, 245, 160, 0.08), transparent 26%),
                linear-gradient(180deg, #F8FAFC 0%, #EEF4FF 100%)
            """,
            "chip_text": "#0F172A",
            "button_text": "#FFFFFF",
            "button_grad": "linear-gradient(90deg, #0EA5E9, #D946EF)",
            "button_grad_hover": "linear-gradient(90deg, #0284C7, #C026D3)",
            "kpi_text": "#0F172A",
            "skeleton_text": "rgba(15,23,42,0.70)",
            "skeleton_line": "rgba(15,23,42,0.10)",
            "skeleton_glow": "rgba(255,255,255,0.55)",
            "mpl_text": "#0F172A",
            "mpl_muted": "#475569",
            "mpl_grid": "#CBD5E1",
            "mpl_spine": "#CBD5E1",
        }

    return {
        "text": "#F5F7FB",
        "muted": "rgba(255,255,255,0.72)",
        "card": "rgba(255,255,255,0.06)",
        "card_border": "rgba(255,255,255,0.10)",
        "sidebar": "rgba(255,255,255,0.05)",
        "sidebar_border": "rgba(255,255,255,0.08)",
        "input_bg": "rgba(255,255,255,0.06)",
        "input_border": "rgba(255,255,255,0.08)",
        "tab_bg": "rgba(255,255,255,0.04)",
        "accent1": "rgba(0,198,255,0.22)",
        "accent2": "rgba(255,0,153,0.18)",
        "hero_bg": """
            linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04)),
            radial-gradient(circle at 20% 20%, rgba(0,217,245,0.12), transparent 35%),
            radial-gradient(circle at 80% 30%, rgba(255,90,205,0.10), transparent 35%)
        """,
        "hero_shadow": "0 10px 40px rgba(0,0,0,0.30)",
        "body_gradient": """
            radial-gradient(circle at 10% 15%, rgba(0, 217, 245, 0.18), transparent 20%),
            radial-gradient(circle at 85% 12%, rgba(255, 90, 205, 0.16), transparent 22%),
            radial-gradient(circle at 50% 85%, rgba(0, 245, 160, 0.10), transparent 26%),
            linear-gradient(180deg, #030712 0%, #07101f 100%)
        """,
        "chip_text": "#F3F7FB",
        "button_text": "#FFFFFF",
        "button_grad": "linear-gradient(90deg, rgba(0,198,255,0.25), rgba(255,0,153,0.20))",
        "button_grad_hover": "linear-gradient(90deg, rgba(0,198,255,0.30), rgba(255,0,153,0.25))",
        "kpi_text": "#FFFFFF",
        "skeleton_text": "rgba(255,255,255,0.72)",
        "skeleton_line": "rgba(255,255,255,0.09)",
        "skeleton_glow": "rgba(255,255,255,0.12)",
        "mpl_text": "#F8FAFC",
        "mpl_muted": "#CBD5E1",
        "mpl_grid": "#334155",
        "mpl_spine": "#334155",
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
.dot-cyan {{ background:#06b6d4; }}
.dot-pink {{ background:#ec4899; }}
.dot-yellow {{ background:#f59e0b; }}

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
    background: linear-gradient(90deg, rgba(6,182,212,0.20), rgba(217,70,239,0.20));
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
    background: linear-gradient(90deg, rgba(239,68,68,0.18), rgba(245,158,11,0.10));
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
    st.markdown("### Workspace")
    theme_choice = st.radio(
        "Appearance",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "dark" else 1,
        horizontal=True,
    )
    st.session_state.theme_mode = theme_choice.lower()

    st.markdown("## Platform Overview")
    st.markdown(
        """
        This dashboard provides **real-time** and **batch fraud scoring**
        using a production-style machine learning workflow.
        """
    )

    st.markdown("### Capabilities")
    st.markdown(
        """
- Real-time fraud scoring  
- Batch CSV fraud analysis  
- Fraud probability estimation  
- Risk-level classification  
- SHAP explainability  
- Analytics dashboard  
- Downloadable scored results  
"""
    )

    st.markdown("### API Status")
    if api_health_state == "healthy":
        st.success("API is healthy")
    elif api_health_state == "issue":
        st.error("API is not responding correctly")
    else:
        st.error("API is not running")

status_html = f"""
<div class="status-strip">
    <div class="status-pill"><span class="status-dot dot-green"></span>Docker Deployed</div>
    <div class="status-pill"><span class="status-dot dot-cyan"></span>SHAP Enabled</div>
    <div class="status-pill"><span class="status-dot {api_health_dot}"></span>{api_health_label}</div>
    <div class="status-pill"><span class="status-dot dot-pink"></span>Displays Results</div>
</div>
"""
st.markdown(status_html, unsafe_allow_html=True)

left, right = st.columns([1.8, 1.0], gap="large")

with left:
    st.markdown(
        """
    <div class="hero-card">
        <div class="info-chip">Designed as a multi-phase fraud analytics pipeline</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.02; margin-top:10px;">
            From raw transactions<br>
            to ranked<br>
            <span style="background: linear-gradient(90deg,#6ee7ff,#ff8bd7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                fraud insights.
            </span>
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

tab1, tab2 = st.tabs(["Single Transaction Scoring", "Batch CSV Scoring"])

with tab1:
    st.markdown('<div class="section-title">Enter Transaction Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Fill the feature values below to score a single transaction.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        Time = st.number_input("Time", value=10000.0)
        V1 = st.number_input("V1", value=-1.2)
        V2 = st.number_input("V2", value=0.3)
        V3 = st.number_input("V3", value=1.1)
        V4 = st.number_input("V4", value=0.5)
        V5 = st.number_input("V5", value=-0.2)
        V6 = st.number_input("V6", value=0.1)
        V7 = st.number_input("V7", value=0.2)
        V8 = st.number_input("V8", value=-0.1)
        V9 = st.number_input("V9", value=0.4)
        V10 = st.number_input("V10", value=-0.3)

    with col2:
        V11 = st.number_input("V11", value=0.2)
        V12 = st.number_input("V12", value=-0.5)
        V13 = st.number_input("V13", value=0.1)
        V14 = st.number_input("V14", value=-0.2)
        V15 = st.number_input("V15", value=0.3)
        V16 = st.number_input("V16", value=-0.1)
        V17 = st.number_input("V17", value=0.2)
        V18 = st.number_input("V18", value=0.1)
        V19 = st.number_input("V19", value=-0.3)
        V20 = st.number_input("V20", value=0.05)
        V21 = st.number_input("V21", value=-0.02)

    with col3:
        V22 = st.number_input("V22", value=0.1)
        V23 = st.number_input("V23", value=-0.03)
        V24 = st.number_input("V24", value=0.2)
        V25 = st.number_input("V25", value=-0.1)
        V26 = st.number_input("V26", value=0.05)
        V27 = st.number_input("V27", value=0.02)
        V28 = st.number_input("V28", value=-0.01)
        Amount = st.number_input("Amount", value=150.5, min_value=0.0)

    input_data = {
        "Time": Time,
        "V1": V1,
        "V2": V2,
        "V3": V3,
        "V4": V4,
        "V5": V5,
        "V6": V6,
        "V7": V7,
        "V8": V8,
        "V9": V9,
        "V10": V10,
        "V11": V11,
        "V12": V12,
        "V13": V13,
        "V14": V14,
        "V15": V15,
        "V16": V16,
        "V17": V17,
        "V18": V18,
        "V19": V19,
        "V20": V20,
        "V21": V21,
        "V22": V22,
        "V23": V23,
        "V24": V24,
        "V25": V25,
        "V26": V26,
        "V27": V27,
        "V28": V28,
        "Amount": Amount,
    }

    if st.button("Predict Fraud Risk", use_container_width=True):
        try:
            pred_response = requests.post(f"{API_BASE_URL}/predict", json=input_data, timeout=30)
            explain_response = requests.post(f"{API_BASE_URL}/explain", json=input_data, timeout=60)

            if pred_response.status_code == 200:
                result = pred_response.json()

                st.markdown("## Prediction Summary")
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
                with k2:
                    st.metric("Prediction", "Fraud" if result["prediction"] == 1 else "Not Fraud")
                with k3:
                    st.metric("Risk Level", result["risk_level"])

                if result["prediction"] == 1 or result["risk_level"] == "HIGH":
                    st.markdown(
                        """
                    <div class="fraud-banner">
                        <div class="fraud-banner-icon">🚨</div>
                        <div>
                            <div class="fraud-banner-title">Suspicious activity detected</div>
                            <div class="fraud-banner-text">This transaction requires analyst review.</div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("✅ Transaction appears legitimate based on model scoring.")

                left_res, right_res = st.columns([1.05, 1.0], gap="large")

                with left_res:
                    st.markdown("### Submitted Transaction")
                    st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

                with right_res:
                    if explain_response.status_code == 200:
                        explanation = explain_response.json()
                        explain_df = pd.DataFrame(explanation["top_features"])
                        st.markdown("### Explainability")
                        st.dataframe(explain_df, use_container_width=True)
                        plot_shap_bar(explain_df, "Top Feature Contributions")
                    else:
                        st.warning("Explanation could not be generated.")
            else:
                st.error(f"API error: {pred_response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")

with tab2:
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
                payload = input_df.to_dict(orient="records")

                progress.progress(15, text="Serialising uploaded transactions...")
                time.sleep(0.12)
                progress.progress(35, text="Sending request to fraud scoring API...")
                response = requests.post(
                    f"{API_BASE_URL}/predict_batch",
                    json=payload,
                    timeout=300,
                )
                progress.progress(75, text="Receiving results and building analytics...")
                time.sleep(0.12)
                progress.progress(100, text="Completed ✅")

                if response.status_code == 200:
                    st.session_state.batch_result_df = pd.DataFrame(response.json())
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
                max_value=len(input_df) - 1,
                value=0,
                step=1,
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
                        explain_df = pd.DataFrame(explain_response.json()["top_features"])
                        st.markdown(f"### SHAP for Uploaded Row {selected_index}")
                        st.dataframe(explain_df, use_container_width=True)
                        plot_shap_bar(explain_df, f"Top Feature Contributions for Row {selected_index}")
                    else:
                        st.error(f"Explanation API error: {explain_response.text}")

                except Exception as e:
                    st.error(f"Explanation failed: {e}")