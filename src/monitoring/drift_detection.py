"""
src/monitoring/drift_detection.py
==================================
Evidently-based data drift monitor.

Reference data loading strategy (in order):
  1. Raw dataset CSV (data/raw/creditcard.csv) — ideal, available after training.
  2. Committed reference snapshot (data/drift_reference.csv) — lightweight CSV
     generated once and committed to the repo; works in any environment.
  3. Synthetic fallback — generates statistically representative data from
     the known feature statistics of the UCI credit-card fraud dataset so the
     drift monitor is *always* operational, even in a fresh clone.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.monitoring.logger import setup_logger
from src.utils.config_loader import load_yaml_file

# ── Feature column definitions ─────────────────────────────────────────────
# V1–V28 are PCA-transformed; their means are ~0, stds vary per feature.
# These approximate stds are derived from the public UCI dataset description.
_V_STD = {
    "V1": 1.96, "V2": 1.65, "V3": 1.52, "V4": 1.42, "V5": 1.38,
    "V6": 1.33, "V7": 1.24, "V8": 1.19, "V9": 1.10, "V10": 1.09,
    "V11": 1.02, "V12": 0.999, "V13": 0.995, "V14": 0.958, "V15": 0.915,
    "V16": 0.876, "V17": 0.849, "V18": 0.838, "V19": 0.814, "V20": 0.771,
    "V21": 0.735, "V22": 0.726, "V23": 0.624, "V24": 0.606, "V25": 0.522,
    "V26": 0.482, "V27": 0.404, "V28": 0.330,
}
_ALL_FEATURE_COLS = ["Time", "Amount"] + list(_V_STD.keys())
_REFERENCE_SNAPSHOT_PATH = "data/drift_reference.csv"
_SYNTHETIC_N = 1000  # rows to generate for the synthetic fallback


def _build_synthetic_reference(n: int = _SYNTHETIC_N) -> pd.DataFrame:
    """
    Generate a statistically representative synthetic reference dataset.

    Uses the published feature statistics of the UCI credit-card fraud
    dataset (284,807 transactions, 48-hour window).  The result is not
    the real data — it is a *distribution approximation* that lets Evidently
    detect meaningful drift against genuine production data.
    """
    rng = np.random.default_rng(42)

    data: dict[str, np.ndarray] = {
        # Time: uniform over 48-hour window (in seconds)
        "Time": rng.uniform(0, 172_800, n),
        # Amount: log-normal (mean ≈ £88, heavy right tail)
        "Amount": np.clip(rng.lognormal(mean=3.0, sigma=1.5, size=n), 0, 25_000),
    }
    for col, std in _V_STD.items():
        data[col] = rng.normal(loc=0.0, scale=std, size=n)

    return pd.DataFrame(data)


class DriftMonitor:
    """Evidently data-drift monitor with robust reference data loading."""

    def __init__(self) -> None:
        self.config = load_yaml_file("configs/config.yaml")
        self.logger = setup_logger(self.config["paths"]["log_file"])
        self.report_dir = self.config.get("monitoring", {}).get("drift_report_dir", "reports")
        os.makedirs(self.report_dir, exist_ok=True)

        self.reference_df = self._load_reference()

    # ── Reference loading ──────────────────────────────────────────────────

    def _load_reference(self) -> pd.DataFrame:
        """
        Load reference data using a 3-tier fallback strategy:
          1. Raw CSV (data/raw/creditcard.csv)
          2. Committed snapshot (data/drift_reference.csv)
          3. Synthetic data generated on the fly
        """
        # Tier 1 — raw CSV (available when user has run training)
        raw_path = self.config["paths"]["raw_data"]
        if os.path.exists(raw_path):
            try:
                df = pd.read_csv(raw_path)
                df = df.drop(columns=[c for c in ["Class", "transaction_memo"] if c in df.columns])
                # Sample for performance — Evidently works well with 500-1000 rows
                if len(df) > 1000:
                    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
                self.logger.info(f"Drift reference loaded from raw CSV ({len(df)} rows).")
                return df
            except Exception as exc:
                self.logger.warning(f"Could not load raw CSV for drift reference: {exc}")

        # Tier 2 — committed reference snapshot
        if os.path.exists(_REFERENCE_SNAPSHOT_PATH):
            try:
                df = pd.read_csv(_REFERENCE_SNAPSHOT_PATH)
                if len(df) > 1000:
                    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
                self.logger.info(
                    f"Drift reference loaded from committed snapshot "
                    f"({_REFERENCE_SNAPSHOT_PATH}, {len(df)} rows)."
                )
                return df
            except Exception as exc:
                self.logger.warning(f"Could not load drift reference snapshot: {exc}")

        # Tier 3 — synthetic fallback (always works)
        self.logger.info(
            "Drift reference CSV not found. "
            "Generating synthetic reference from UCI dataset statistics."
        )
        df = _build_synthetic_reference()

        # Persist as the committed snapshot so next startup uses Tier 2
        try:
            os.makedirs(os.path.dirname(_REFERENCE_SNAPSHOT_PATH), exist_ok=True)
            df.to_csv(_REFERENCE_SNAPSHOT_PATH, index=False)
            self.logger.info(f"Synthetic reference saved to {_REFERENCE_SNAPSHOT_PATH}.")
        except Exception as exc:
            self.logger.warning(f"Could not persist synthetic reference: {exc}")

        return df

    # ── Report generation ──────────────────────────────────────────────────

    def generate_drift_report(
        self,
        production_data: list[dict],
        report_name: str = "data_drift.html",
    ) -> str:
        """
        Generate an Evidently HTML drift report.

        Args:
            production_data: List of transaction dicts (same schema as /predict).
            report_name:     Output filename inside ``reports/``.

        Returns:
            Absolute path to the generated HTML file, or an ``"Error: …"``
            string if generation fails.
        """
        if not production_data:
            return "Error: No production data provided."

        prod_df = pd.DataFrame(production_data)
        # Drop class/label columns if present
        prod_df = prod_df.drop(
            columns=[c for c in ["Class", "transaction_memo"] if c in prod_df.columns]
        )

        # Align columns — compare only features present in both frames
        common_cols = sorted(set(self.reference_df.columns) & set(prod_df.columns))
        if not common_cols:
            return (
                "Error: No common columns between production data and reference. "
                "Ensure the uploaded CSV contains the same features used during training "
                "(Time, Amount, V1–V28)."
            )

        # Minimum column count guard — Evidently needs at least 2 columns
        if len(common_cols) < 2:
            return "Error: At least 2 common columns are required for drift analysis."

        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_df[common_cols],
                current_data=prod_df[common_cols],
            )

            report_path = os.path.join(self.report_dir, report_name)
            report.save_html(report_path)
            self.logger.info(f"Drift report generated at {report_path}")
            return report_path

        except Exception as exc:
            self.logger.error(f"Evidently report generation failed: {exc}")
            return f"Error: Drift report generation failed — {exc}"
