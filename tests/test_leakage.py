"""
tests/test_leakage.py
=====================

Proves that the trajectory feature pipeline raises ValueError when fed
any row with month_idx >= 12 (the observation month boundary).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.trajectory import build_trajectory_features, OBSERVATION_MONTH


# ── Helper ───────────────────────────────────────────────────────────────────

def _valid_behavior(n_loans: int = 3, n_months: int = 12) -> pd.DataFrame:
    """Build a clean behavior DataFrame with month_idx in [0, n_months)."""
    rows = []
    for i in range(n_loans):
        lid = f"L{i:03d}"
        for m in range(n_months):
            rows.append({
                "loan_id": lid,
                "month_idx": m,
                "payment_received": 1,
                "dishonored": 0,
                "payment_amount": 3000.0,
                "days_past_due": float(np.random.randint(0, 10)),
                "arrear_balance": 0.0,
                "effective_roi": 9.5,
            })
    return pd.DataFrame(rows)


# ── Leakage tests ────────────────────────────────────────────────────────────

def test_no_leakage_on_valid_data():
    """Pipeline should succeed when all month_idx < OBSERVATION_MONTH."""
    df = _valid_behavior(n_loans=5, n_months=12)
    assert (df["month_idx"] < OBSERVATION_MONTH).all()
    result = build_trajectory_features(df)
    assert len(result) == 5
    assert "dpd_slope_3m" in result.columns


def test_raises_on_month_12():
    """A single row with month_idx == 12 must raise ValueError."""
    df = _valid_behavior(n_loans=2, n_months=12)
    # Inject one leaking row
    leaked_row = {
        "loan_id": "L000",
        "month_idx": 12,  # ← exactly at observation boundary
        "payment_received": 1,
        "dishonored": 0,
        "payment_amount": 3000.0,
        "days_past_due": 5.0,
        "arrear_balance": 0.0,
        "effective_roi": 9.5,
    }
    df = pd.concat([df, pd.DataFrame([leaked_row])], ignore_index=True)
    with pytest.raises(ValueError, match="Leakage detected"):
        build_trajectory_features(df)


def test_raises_on_month_beyond_12():
    """Rows with month_idx > 12 (e.g. 17) must also raise ValueError."""
    df = _valid_behavior(n_loans=2, n_months=12)
    leaked_row = {
        "loan_id": "L001",
        "month_idx": 17,  # ← future window
        "payment_received": 0,
        "dishonored": 1,
        "payment_amount": 0.0,
        "days_past_due": 95.0,
        "arrear_balance": 3000.0,
        "effective_roi": 9.5,
    }
    df = pd.concat([df, pd.DataFrame([leaked_row])], ignore_index=True)
    with pytest.raises(ValueError, match="Leakage detected"):
        build_trajectory_features(df)


def test_raises_when_only_future_rows():
    """DataFrame containing ONLY future months must raise, not silently succeed."""
    rows = []
    for m in range(12, 18):
        rows.append({
            "loan_id": "L000",
            "month_idx": m,
            "payment_received": 1,
            "dishonored": 0,
            "payment_amount": 3000.0,
            "days_past_due": 5.0,
            "arrear_balance": 0.0,
            "effective_roi": 9.5,
        })
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="Leakage detected"):
        build_trajectory_features(df)


def test_feature_values_are_finite():
    """All trajectory features should be finite (no NaN/Inf) on clean data."""
    df = _valid_behavior(n_loans=10, n_months=12)
    result = build_trajectory_features(df)
    numeric_cols = [c for c in result.columns if c != "loan_id"]
    assert result[numeric_cols].apply(pd.to_numeric, errors="coerce").notna().all().all(), \
        "Some trajectory features contain NaN/Inf"
