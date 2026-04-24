"""
Feature Engineering Module
==========================
Creates domain-relevant tabular and NLP features for the fraud-detection
pipeline.  Engineered features include:

- ``log_amount``           — log-transformed transaction amount
- ``hour``                 — hour of day extracted from elapsed seconds
- ``is_night_transaction`` — binary flag for transactions between 00:00–06:00
- ``amount_zscore``        — z-score of Amount relative to training distribution
- ``v_features_magnitude`` — L2 norm of PCA components V1–V28
- ``amount_time_ratio``    — interaction: Amount / (Time + 1)
- ``nlp_svd_0 … nlp_svd_4``— TF-IDF + SVD features from transaction memos
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.utils.common import save_object, load_object
import os
import logging

logger = logging.getLogger("fraud_detection_logger")


class FeatureEngineering:
    """
    Stateful transformer that creates new features for fraud detection.

    During training (``is_train=True``), fits TF-IDF + SVD on the memo
    column and persists the fitted objects.  During inference, loads the
    saved artefacts so that the same transformation is applied.
    """

    def __init__(self, artifact_dir: str = "models/artifacts"):
        self.artifact_dir = artifact_dir

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (or partially processed) transaction data.
        is_train : bool
            Whether we are in training mode (fit + transform) or inference
            mode (transform only, using saved artefacts).

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame with original string columns dropped.
        """
        X = X.copy()

        X = self._add_log_amount(X)
        X = self._add_time_features(X)
        X = self._add_amount_zscore(X)
        X = self._add_v_magnitude(X)
        X = self._add_amount_time_ratio(X)
        X = self._add_nlp_features(X, is_train)

        return X

    # ------------------------------------------------------------------ #
    #  Individual feature builders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_log_amount(X: pd.DataFrame) -> pd.DataFrame:
        """Log-transform of Amount to reduce right-skew."""
        if "Amount" in X.columns:
            X["log_amount"] = np.log1p(X["Amount"])
        return X

    @staticmethod
    def _add_time_features(X: pd.DataFrame) -> pd.DataFrame:
        """Extract hour-of-day and night-transaction flag."""
        if "Time" in X.columns:
            X["hour"] = (X["Time"] // 3600) % 24
            X["is_night_transaction"] = X["hour"].apply(
                lambda h: 1 if h < 6 else 0
            ).astype(np.int8)
        return X

    @staticmethod
    def _add_amount_zscore(X: pd.DataFrame) -> pd.DataFrame:
        """Z-score of Amount (relative to current batch)."""
        if "Amount" in X.columns:
            mean = X["Amount"].mean()
            std = X["Amount"].std()
            X["amount_zscore"] = (
                (X["Amount"] - mean) / std if std > 0 else 0.0
            )
        return X

    @staticmethod
    def _add_v_magnitude(X: pd.DataFrame) -> pd.DataFrame:
        """L2 norm of the PCA components V1–V28."""
        v_cols = [c for c in X.columns if c.startswith("V") and c[1:].isdigit()]
        if v_cols:
            X["v_features_magnitude"] = np.sqrt(
                (X[v_cols] ** 2).sum(axis=1)
            )
        return X

    @staticmethod
    def _add_amount_time_ratio(X: pd.DataFrame) -> pd.DataFrame:
        """Interaction feature: Amount relative to time elapsed."""
        if "Amount" in X.columns and "Time" in X.columns:
            X["amount_time_ratio"] = X["Amount"] / (X["Time"] + 1)
        return X

    def _add_nlp_features(
        self, X: pd.DataFrame, is_train: bool
    ) -> pd.DataFrame:
        """TF-IDF + Truncated SVD on the transaction_memo column.

        Robustly handles the case where all memos are identical (sparse
        TF-IDF matrix) by capping n_components to min(n_components,
        n_features - 1).  When no memo column is present, zero-fills the
        SVD columns so downstream feature shapes stay consistent.
        """
        n_components = 5

        if "transaction_memo" not in X.columns:
            # No memo column — fill with zeros for consistent feature shape
            for i in range(n_components):
                X[f"nlp_svd_{i}"] = 0.0
            return X

        if is_train:
            vectorizer = TfidfVectorizer(
                max_features=100, stop_words="english"
            )
            tfidf_matrix = vectorizer.fit_transform(X["transaction_memo"])

            # Cap n_components so SVD never fails on sparse/identical memos
            actual_components = min(n_components, tfidf_matrix.shape[1] - 1)
            if actual_components < 1:
                actual_components = 1

            svd = TruncatedSVD(n_components=actual_components, random_state=42)
            nlp_raw = svd.fit_transform(tfidf_matrix)

            os.makedirs(self.artifact_dir, exist_ok=True)
            save_object(f"{self.artifact_dir}/tfidf_vectorizer.pkl", vectorizer)
            save_object(f"{self.artifact_dir}/tfidf_svd.pkl", svd)
            logger.info(
                "Fitted and saved TF-IDF + SVD artefacts "
                f"(n_components={actual_components})."
            )
        else:
            try:
                vectorizer = load_object(
                    f"{self.artifact_dir}/tfidf_vectorizer.pkl"
                )
                svd = load_object(f"{self.artifact_dir}/tfidf_svd.pkl")
                tfidf_matrix = vectorizer.transform(X["transaction_memo"])

                # Guard: if matrix is too sparse for the saved SVD
                if tfidf_matrix.shape[1] < svd.n_components:
                    logger.warning(
                        "TF-IDF produced fewer features than SVD n_components "
                        f"({tfidf_matrix.shape[1]} < {svd.n_components}). "
                        "Filling NLP features with zeros."
                    )
                    nlp_raw = np.zeros((len(X), svd.n_components))
                else:
                    nlp_raw = svd.transform(tfidf_matrix)
            except Exception as exc:
                logger.warning(
                    f"NLP artefacts not found or failed ({exc}) "
                    "— filling with zeros."
                )
                nlp_raw = np.zeros((len(X), n_components))

        # Pad/trim to exactly n_components columns for consistent shape
        actual_cols = nlp_raw.shape[1]
        nlp_df = pd.DataFrame(
            nlp_raw,
            columns=[f"nlp_svd_{i}" for i in range(actual_cols)],
            index=X.index,
        )
        # Zero-pad any missing columns up to n_components
        for i in range(actual_cols, n_components):
            nlp_df[f"nlp_svd_{i}"] = 0.0

        X = pd.concat([X, nlp_df], axis=1)
        X.drop(columns=["transaction_memo"], inplace=True, errors="ignore")

        return X