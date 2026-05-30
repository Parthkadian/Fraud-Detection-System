"""
Generate EDA Notebook
=====================
Creates a comprehensive Exploratory Data Analysis notebook for the
credit card fraud dataset. Run this script to generate the notebook
and all associated figures.

Usage:
    python notebooks/generate_eda.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

PALETTE = ["#003366", "#D4AF37", "#00563B", "#8B0000", "#4169E1"]

print("Loading dataset...")
df = pd.read_csv("data/raw/creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ═══════════════════════════════════════════════════════════════════
# 1. CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("\n[1/6] Class Distribution Analysis")

class_counts = df["Class"].value_counts()
fraud_pct = class_counts[1] / len(df) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
bars = axes[0].bar(
    ["Legitimate (0)", "Fraud (1)"],
    class_counts.values,
    color=[PALETTE[0], PALETTE[3]],
    edgecolor="white",
    linewidth=1.5,
)
axes[0].set_title("Transaction Class Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
for bar, count in zip(bars, class_counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center", va="bottom", fontweight="bold",
    )

# Pie chart
axes[1].pie(
    class_counts.values,
    labels=[f"Legitimate\n{class_counts[0]:,}", f"Fraud\n{class_counts[1]:,}"],
    colors=[PALETTE[0], PALETTE[3]],
    autopct="%1.3f%%",
    startangle=90,
    explode=(0, 0.08),
    textprops={"fontsize": 11},
)
axes[1].set_title("Class Proportion", fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Fraud: {class_counts[1]:,} ({fraud_pct:.3f}%)")
print(f"  Legitimate: {class_counts[0]:,} ({100-fraud_pct:.3f}%)")
print(f"  Imbalance ratio: 1:{class_counts[0]//class_counts[1]}")

# ═══════════════════════════════════════════════════════════════════
# 2. AMOUNT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("\n[2/6] Amount Distribution Analysis")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall distribution
axes[0].hist(df["Amount"], bins=100, color=PALETTE[0], alpha=0.8, edgecolor="white")
axes[0].set_title("Transaction Amount Distribution", fontweight="bold")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Frequency")
axes[0].set_yscale("log")

# By class
for cls, color, label in [(0, PALETTE[0], "Legitimate"), (1, PALETTE[3], "Fraud")]:
    subset = df[df["Class"] == cls]["Amount"]
    axes[1].hist(subset, bins=80, alpha=0.6, color=color, label=label, edgecolor="white")
axes[1].set_title("Amount Distribution by Class", fontweight="bold")
axes[1].set_xlabel("Amount")
axes[1].set_ylabel("Frequency")
axes[1].set_yscale("log")
axes[1].legend()

# Box plot
fraud_amounts = df[df["Class"] == 1]["Amount"]
legit_amounts = df[df["Class"] == 0]["Amount"]
bp = axes[2].boxplot(
    [legit_amounts, fraud_amounts],
    labels=["Legitimate", "Fraud"],
    patch_artist=True,
)
bp["boxes"][0].set_facecolor(PALETTE[0])
bp["boxes"][1].set_facecolor(PALETTE[3])
for box in bp["boxes"]:
    box.set_alpha(0.7)
axes[2].set_title("Amount Box Plot by Class", fontweight="bold")
axes[2].set_ylabel("Amount")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_amount_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Fraud amount: mean={fraud_amounts.mean():.2f}, median={fraud_amounts.median():.2f}, max={fraud_amounts.max():.2f}")
print(f"  Legit amount: mean={legit_amounts.mean():.2f}, median={legit_amounts.median():.2f}, max={legit_amounts.max():.2f}")

# ═══════════════════════════════════════════════════════════════════
# 3. TIME DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("\n[3/6] Time Distribution Analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time histogram by class
for cls, color, label in [(0, PALETTE[0], "Legitimate"), (1, PALETTE[3], "Fraud")]:
    subset = df[df["Class"] == cls]
    axes[0].hist(subset["Time"] / 3600, bins=48, alpha=0.6, color=color, label=label, edgecolor="white")
axes[0].set_title("Transaction Time Distribution (Hours)", fontweight="bold")
axes[0].set_xlabel("Time (hours)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Fraud rate over time
df["hour_bucket"] = (df["Time"] // 3600).astype(int)
hourly = df.groupby("hour_bucket").agg(
    total=("Class", "count"),
    fraud=("Class", "sum"),
).reset_index()
hourly["fraud_rate"] = hourly["fraud"] / hourly["total"] * 100

axes[1].bar(hourly["hour_bucket"], hourly["fraud_rate"], color=PALETTE[1], edgecolor="white", alpha=0.8)
axes[1].set_title("Fraud Rate Over Time Buckets", fontweight="bold")
axes[1].set_xlabel("Time Bucket (hours)")
axes[1].set_ylabel("Fraud Rate (%)")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_time_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Time range: {df['Time'].min():.0f}s to {df['Time'].max():.0f}s ({df['Time'].max()/3600:.1f} hours)")

# ═══════════════════════════════════════════════════════════════════
# 4. FEATURE CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════
print("\n[4/6] Feature Correlation Heatmap")

# Top correlations with Class
v_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
correlations = df[v_cols + ["Class"]].corr()["Class"].drop("Class").abs().sort_values(ascending=False)

# Top 15 features
top_features = correlations.head(15).index.tolist() + ["Class"]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Correlation with Class
axes[0].barh(
    correlations.head(15).index[::-1],
    correlations.head(15).values[::-1],
    color=PALETTE[0],
    edgecolor="white",
)
axes[0].set_title("Top 15 Features Correlated with Fraud", fontweight="bold")
axes[0].set_xlabel("|Correlation with Class|")

# Heatmap of top features
corr_matrix = df[top_features].corr()
sns.heatmap(
    corr_matrix,
    ax=axes[1],
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    annot_kws={"size": 7},
)
axes[1].set_title("Correlation Heatmap (Top Features)", fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Top 5 correlated features: {list(correlations.head(5).index)}")

# ═══════════════════════════════════════════════════════════════════
# 5. PCA COMPONENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n[5/6] PCA Component Analysis")

# The V features ARE PCA components, so we analyse their variance
v_cols_only = [f"V{i}" for i in range(1, 29)]
variance_explained = df[v_cols_only].var().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Variance per component
axes[0].bar(
    range(len(variance_explained)),
    variance_explained.values,
    color=PALETTE[0],
    edgecolor="white",
    alpha=0.8,
)
axes[0].set_title("Variance Per PCA Component", fontweight="bold")
axes[0].set_xlabel("Component Index")
axes[0].set_ylabel("Variance")
axes[0].set_xticks(range(0, 28, 2))
axes[0].set_xticklabels([variance_explained.index[i] for i in range(0, 28, 2)], rotation=45, ha="right")

# Cumulative explained variance (normalised)
total_var = variance_explained.sum()
cumulative = np.cumsum(variance_explained.values) / total_var * 100
axes[1].plot(range(len(cumulative)), cumulative, "o-", color=PALETTE[2], linewidth=2)
axes[1].axhline(y=95, color=PALETTE[3], linestyle="--", alpha=0.7, label="95% threshold")
axes[1].set_title("Cumulative Variance Explained", fontweight="bold")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# 2D scatter of top 2 components
top2 = variance_explained.index[:2].tolist()
sample = df.sample(n=min(5000, len(df)), random_state=42)
for cls, color, label in [(0, PALETTE[0], "Legit"), (1, PALETTE[3], "Fraud")]:
    subset = sample[sample["Class"] == cls]
    axes[2].scatter(subset[top2[0]], subset[top2[1]], alpha=0.4, s=8, c=color, label=label)
axes[2].set_title(f"Top 2 PCA Components: {top2[0]} vs {top2[1]}", fontweight="bold")
axes[2].set_xlabel(top2[0])
axes[2].set_ylabel(top2[1])
axes[2].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_pca_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

n95 = int(np.argmax(cumulative >= 95)) + 1
print(f"  Components for 95% variance: {n95}")
print(f"  Top 3 variance components: {list(variance_explained.index[:3])}")

# ═══════════════════════════════════════════════════════════════════
# 6. STATISTICAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n[6/6] Statistical Summary")

summary = df.describe().T
summary["null_pct"] = df.isnull().mean() * 100
summary["skew"] = df.skew()
summary = summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "null_pct", "skew"]]
summary.to_csv(FIGURES_DIR / "statistical_summary.csv")
print(f"  Summary saved to {FIGURES_DIR / 'statistical_summary.csv'}")

# Cleanup temp column
df.drop(columns=["hour_bucket"], inplace=True, errors="ignore")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print(f"All figures saved to: {FIGURES_DIR.absolute()}")
print("=" * 60)
