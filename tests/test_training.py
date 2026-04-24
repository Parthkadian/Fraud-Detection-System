"""
Tests — Feature Engineering
============================
Tests for the FeatureEngineering class.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineering


def _make_df(n: int = 20) -> pd.DataFrame:
    """Create a minimal DataFrame with required columns for feature engineering."""
    np.random.seed(42)
    data = {
        "Amount": np.random.uniform(10, 500, n),
        "Time": np.random.uniform(0, 172800, n),
    }
    for i in range(1, 6):
        data[f"V{i}"] = np.random.randn(n)
    # Use diverse memos so TF-IDF produces >= 5 unique tokens for SVD
    diverse_memos = [
        "Coffee shop transaction morning",
        "Online electronics store purchase",
        "Gas station fuel payment",
        "Grocery store weekend shopping",
        "Suspicious overseas wire transfer",
        "Pharmacy medicine purchase local",
        "Restaurant dinner evening meal",
        "Airline ticket booking travel",
        "Hotel accommodation business trip",
        "Cryptocurrency exchange transfer funds",
    ]
    data["transaction_memo"] = [diverse_memos[i % len(diverse_memos)] for i in range(n)]
    return pd.DataFrame(data)


class TestFeatureEngineering:
    """Tests for the FeatureEngineering class."""

    def test_log_amount_created(self):
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert "log_amount" in transformed.columns

    def test_time_features_created(self):
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert "hour" in transformed.columns
        assert "is_night_transaction" in transformed.columns

    def test_domain_features_created(self):
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert "amount_zscore" in transformed.columns
        assert "amount_time_ratio" in transformed.columns

    def test_v_magnitude_with_enough_v_cols(self):
        """v_features_magnitude requires at least one V column."""
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert "v_features_magnitude" in transformed.columns

    def test_memo_dropped_after_nlp(self):
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert "transaction_memo" not in transformed.columns

    def test_nlp_svd_features_created(self):
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert "nlp_svd_0" in transformed.columns

    def test_no_missing_values_after_transform(self):
        df = _make_df()
        fe = FeatureEngineering()
        transformed = fe.transform(df, is_train=True)
        assert not transformed.isnull().any().any()

