"""
src/features/trajectory.py
==========================

Trajectory feature engineering from behavior_history.csv.

All features are computed using only month_idx < 12 (observation window).
A leakage guard raises ValueError if any row with month_idx >= 12 is present.

Features
--------
1. dpd_slope_3m
   Linear-regression slope of days_past_due over the last 3 months (months 9-11).
   Formula: OLS beta of [DPD_9, DPD_10, DPD_11] ~ [0,1,2].
   Why: A rising DPD trend approaching the observation point signals imminent default.

2. dpd_acceleration
   Second difference of DPD: (DPD_11 - DPD_10) - (DPD_10 - DPD_9).
   Formula: DPD_11 - 2*DPD_10 + DPD_9.
   Why: Acceleration (convexity) catches decliners whose pace of deterioration is
   itself increasing — a more sensitive signal than slope alone.

3. dishonor_slope_3m
   OLS slope of the dishonored flag over the last 3 months (months 9-11).
   Formula: OLS beta of [D_9, D_10, D_11] ~ [0,1,2].
   Why: An increasing rate of payment failures is a leading indicator even before
   DPD crosses the 90-day threshold.

4. stuck_high_indicator
   1 if mean(DPD, months 6-11) >= 30 AND std(DPD, months 6-11) <= 20, else 0.
   Formula: int(mean_dpd_6_11 >= 30 and std_dpd_6_11 <= 20).
   Why: Distinguishes chronic-elevated borrowers (plateau) whose probability of
   suddenly crossing 90 DPD is non-trivial due to any shock.

5. arrear_balance_trend
   OLS slope of arrear_balance over months 6-11 (normalised by EMI = 1000 for scale).
   Formula: OLS beta / 1000.
   Why: A growing arrear balance indicates the borrower is progressively unable to
   service their debt — even if individual months don't yet cross 90 DPD.

6. payment_gap_variance
   Variance of (EMI - payment_amount) over all 12 months, clipped at 0 so only
   shortfalls count. Formula: var(max(0, EMI - payment_amount), months 0-11).
   Why: High variance in payment shortfalls signals erratic repayment behaviour,
   a strong predictor of future default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OBSERVATION_MONTH = 12  # candidate-visible window is [0, 12)


def _ols_slope(y: np.ndarray) -> float:
    """Return OLS slope of y regressed on [0, 1, ..., n-1]."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _safe_std(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    return float(arr.std(ddof=0))


def build_trajectory_features(behavior: pd.DataFrame) -> pd.DataFrame:
    """
    Given behavior_history (months 0..11), return one-row-per-loan DataFrame
    with trajectory features.

    Parameters
    ----------
    behavior : pd.DataFrame
        Columns required: loan_id, month_idx, days_past_due, dishonored,
        payment_amount, arrear_balance.

    Returns
    -------
    pd.DataFrame  — one row per loan_id, columns = feature names + loan_id.

    Raises
    ------
    ValueError if any row has month_idx >= OBSERVATION_MONTH (leakage guard).
    """
    # ── Leakage guard ────────────────────────────────────────────────────────
    leaked = behavior["month_idx"] >= OBSERVATION_MONTH
    if leaked.any():
        bad_months = sorted(behavior.loc[leaked, "month_idx"].unique().tolist())
        raise ValueError(
            f"Leakage detected: behavior contains month_idx >= {OBSERVATION_MONTH}. "
            f"Offending months: {bad_months}"
        )

    records = []

    for loan_id, grp in behavior.sort_values("month_idx").groupby("loan_id", sort=False):
        grp = grp.sort_values("month_idx")

        dpd = grp["days_past_due"].to_numpy(dtype=float)
        dishonored = grp["dishonored"].to_numpy(dtype=float)
        payment_amt = grp["payment_amount"].to_numpy(dtype=float)
        arrear = grp["arrear_balance"].to_numpy(dtype=float)

        n = len(dpd)

        # ── Feature 1: dpd_slope_3m ───────────────────────────────────────
        dpd_last3 = dpd[max(0, n - 3):]
        dpd_slope_3m = _ols_slope(dpd_last3)

        # ── Feature 2: dpd_acceleration ──────────────────────────────────
        if n >= 3:
            dpd_acceleration = float(dpd[-1] - 2 * dpd[-2] + dpd[-3])
        else:
            dpd_acceleration = 0.0

        # ── Feature 3: dishonor_slope_3m ─────────────────────────────────
        dis_last3 = dishonored[max(0, n - 3):]
        dishonor_slope_3m = _ols_slope(dis_last3)

        # ── Feature 4: stuck_high_indicator ──────────────────────────────
        dpd_6_11 = dpd[max(0, n - 6):]
        mean_dpd_6_11 = float(dpd_6_11.mean()) if len(dpd_6_11) > 0 else 0.0
        std_dpd_6_11 = _safe_std(dpd_6_11)
        stuck_high_indicator = int(mean_dpd_6_11 >= 30 and std_dpd_6_11 <= 20)

        # ── Feature 5: arrear_balance_trend ──────────────────────────────
        arrear_6_11 = arrear[max(0, n - 6):]
        arrear_balance_trend = _ols_slope(arrear_6_11) / 1000.0

        # ── Feature 6: payment_gap_variance ──────────────────────────────
        # Need EMI: derive from the first non-zero payment or use arrear changes.
        # We don't have EMI in behavior; we'll use payment_amount directly:
        # gap = max(0, median_payment - payment_amount)  [median as proxy for EMI]
        median_payment = float(np.median(payment_amt[payment_amt > 0])) if (payment_amt > 0).any() else 1.0
        gaps = np.maximum(0.0, median_payment - payment_amt)
        payment_gap_variance = float(gaps.var()) if len(gaps) >= 2 else 0.0

        records.append({
            "loan_id": loan_id,
            "dpd_slope_3m": dpd_slope_3m,
            "dpd_acceleration": dpd_acceleration,
            "dishonor_slope_3m": dishonor_slope_3m,
            "stuck_high_indicator": stuck_high_indicator,
            "arrear_balance_trend": arrear_balance_trend,
            "payment_gap_variance": payment_gap_variance,
            # Snapshot features (contextual, not the primary trajectory signals)
            "dpd_month11": float(dpd[-1]) if n > 0 else 0.0,
            "arrear_month11": float(arrear[-1]) if n > 0 else 0.0,
            "dishonor_rate_all": float(dishonored.mean()) if n > 0 else 0.0,
            "mean_dpd_all": float(dpd.mean()) if n > 0 else 0.0,
        })

    return pd.DataFrame(records)
