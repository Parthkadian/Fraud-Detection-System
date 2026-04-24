"""
Drift Report Utilities
======================
Helper functions for generating and formatting Evidently drift reports.
Used by both the API endpoint and the dashboard's drift monitoring tab.
"""

import os
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("fraud_detection_logger")


def format_drift_summary(report_json: dict) -> dict:
    """
    Extract a human-readable summary from an Evidently report JSON.

    Parameters
    ----------
    report_json : dict
        Parsed Evidently report output.

    Returns
    -------
    dict
        Simplified drift summary with drifted column count and names.
    """
    try:
        metrics = report_json.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            if "number_of_drifted_columns" in result:
                return {
                    "drifted_columns": result.get("number_of_drifted_columns", 0),
                    "total_columns": result.get("number_of_columns", 0),
                    "dataset_drift": result.get("dataset_drift", False),
                    "drift_share": round(result.get("share_of_drifted_columns", 0), 4),
                }
    except Exception as exc:
        logger.error(f"Failed to parse drift report: {exc}")

    return {"error": "Could not parse drift report."}
