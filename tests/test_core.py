"""
Unit tests for core components.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.woe_encoder import WoEEncoder, compute_psi
from monitoring.drift_detector import compute_psi, interpret_psi


# ── WoE Encoder Tests ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(18, 70, 500),
        "income": np.random.uniform(1000, 10000, 500),
        "utilization": np.random.uniform(0, 1, 500),
    })
    y = pd.Series(np.random.binomial(1, 0.12, 500))
    return X, y


def test_woe_encoder_fit_transform(sample_data):
    X, y = sample_data
    encoder = WoEEncoder(n_bins=5)
    X_woe = encoder.fit_transform(X, y)
    assert X_woe.shape == X.shape
    assert not X_woe.isnull().any().any()


def test_iv_summary_has_all_features(sample_data):
    X, y = sample_data
    encoder = WoEEncoder()
    encoder.fit(X, y)
    iv_df = encoder.get_iv_summary()
    assert set(iv_df["Feature"]) == set(X.columns)


def test_feature_selection_removes_useless(sample_data):
    X, y = sample_data
    encoder = WoEEncoder()
    encoder.fit(X, y)
    selected = encoder.select_features(min_iv=0.02)
    assert isinstance(selected, list)
    assert all(encoder.iv_values[f] >= 0.02 for f in selected)


def test_woe_transform_new_data(sample_data):
    X, y = sample_data
    encoder = WoEEncoder()
    encoder.fit(X, y)
    X_new = X.sample(50, random_state=1)
    X_new_woe = encoder.transform(X_new)
    assert X_new_woe.shape == X_new.shape


# ── PSI Tests ─────────────────────────────────────────────────────────────────

def test_psi_identical_distributions():
    """PSI of identical distributions should be near 0."""
    data = np.random.beta(2, 8, 1000)
    psi = compute_psi(data, data)
    assert psi < 0.01


def test_psi_very_different_distributions():
    """PSI of very different distributions should be high."""
    ref = np.random.beta(2, 20, 1000)   # Mostly low scores
    cur = np.random.beta(10, 2, 1000)   # Mostly high scores
    psi = compute_psi(ref, cur)
    assert psi > 0.25


def test_psi_interpret_stable():
    assert "STABLE" in interpret_psi(0.05)


def test_psi_interpret_moderate():
    assert "MODERATE" in interpret_psi(0.15)


def test_psi_interpret_major():
    assert "MAJOR" in interpret_psi(0.30)
