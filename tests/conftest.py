"""
Shared Pytest Fixtures
======================
Provides reusable test data across all test modules.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_transaction() -> dict:
    """A single valid transaction dictionary."""
    return {
        "Time": 10000.0,
        "V1": -1.2, "V2": 0.3, "V3": 1.1, "V4": 0.5, "V5": -0.2,
        "V6": 0.1, "V7": 0.2, "V8": -0.1, "V9": 0.4, "V10": -0.3,
        "V11": 0.2, "V12": -0.5, "V13": 0.1, "V14": -0.2, "V15": 0.3,
        "V16": -0.1, "V17": 0.2, "V18": 0.1, "V19": -0.3, "V20": 0.05,
        "V21": -0.02, "V22": 0.1, "V23": -0.03, "V24": 0.2, "V25": -0.1,
        "V26": 0.05, "V27": 0.02, "V28": -0.01,
        "Amount": 150.50,
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """A small DataFrame suitable for unit-testing preprocessing and features."""
    np.random.seed(42)
    n = 20
    data = {
        "Time": np.random.uniform(0, 172800, n),
        "Amount": np.random.uniform(0, 5000, n),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)

    data["Class"] = [0] * 18 + [1] * 2
    data["transaction_memo"] = (
        ["Coffee shop"] * 5
        + ["Gas station"] * 5
        + ["Online purchase"] * 5
        + ["Grocery store"] * 3
        + ["Suspicious transfer"] * 2
    )
    return pd.DataFrame(data)


@pytest.fixture
def sample_predictions():
    """Pre-computed arrays for testing evaluation."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.1, 0.05, 0.2, 0.15, 0.7, 0.9, 0.85, 0.35, 0.1, 0.05])
    return y_true, y_pred, y_prob
