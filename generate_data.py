"""
generate_data.py
================

Synthetic loan portfolio generator for the EWS take-home assignment.

Produces:
  - loans_static.csv       : one row per loan, origination features + targets
  - behavior_history.csv   : monthly behavioral records for months 0..11

The model the candidate builds must, at observation_month=12, predict whether
a loan will hit serious arrears (DPD > 90) within the next 3 and 6 months.

Design notes (for the hiring team, not the candidate)
-----------------------------------------------------
Three borrower regimes:
  - "clean"    (70%) : low dishonor (~2%), low DPD. Should be predicted safe.
  - "chronic"  (15%) : stuck-high arrears (~45 DPD mean, 18% dishonor) but
                       does not progress. Tests whether candidate's momentum
                       features over-trigger on plateaus.
  - "decliner" (15%) : clean for the first 6-14 months, then progressive
                       deterioration. This is the EWS-relevant cohort —
                       smart trajectory features should catch them early.

Targets are computed from MONTHS 12..17, which the candidate never sees.

Run:
    python generate_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SEED = 42
N_LOANS = 15_000
N_MONTHS_TOTAL = 24            # full history generated
OBSERVATION_MONTH = 12         # candidate sees months 0..11
HORIZON_3M = 3
HORIZON_6M = 6
DEFAULT_DPD_THRESHOLD = 90     # a month with DPD > 90 = "in serious arrears"

REGIME_PROBS = {
    "clean":    0.70,
    "chronic":  0.15,
    "decliner": 0.15,
}

rng = np.random.default_rng(SEED)


# -----------------------------------------------------------------------------
# Static origination features
# -----------------------------------------------------------------------------
def sample_static_features(n: int) -> pd.DataFrame:
    """Generate origination-time features for n loans."""
    loan_amt = rng.lognormal(mean=12.5, sigma=0.6, size=n).clip(50_000, 5_000_000)
    term = rng.choice([60, 120, 180, 240, 300], size=n,
                      p=[0.05, 0.15, 0.30, 0.30, 0.20])
    roi = rng.normal(9.5, 1.2, n).clip(7.0, 14.0)
    # standard EMI formula
    r = roi / 1200
    emi = (loan_amt * r * (1 + r) ** term) / ((1 + r) ** term - 1)

    monthly_income = rng.lognormal(mean=10.7, sigma=0.5, size=n).clip(20_000, 800_000)
    dti = (emi / monthly_income).clip(0.05, 0.85)

    issue_year = rng.integers(2021, 2024, size=n)
    issue_month = rng.integers(1, 13, size=n)
    issue_date = pd.to_datetime(
        [f"{y}-{m:02d}-01" for y, m in zip(issue_year, issue_month)]
    )

    return pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(n)],
        "issue_date": issue_date,
        "loan_amt": loan_amt.round(0).astype(int),
        "term_months": term,
        "roi_initial": roi.round(2),
        "emi": emi.round(0).astype(int),
        "monthly_income": monthly_income.round(0).astype(int),
        "dti": dti.round(3),
        "borrower_age": rng.integers(25, 60, size=n),
        "emp_type": rng.choice(
            ["E", "S", "M", "R"], size=n, p=[0.55, 0.25, 0.10, 0.10]
        ),
        "loan_purpose": rng.choice(
            ["FLAT_PURCH", "CONSTRUCTION", "PLOT_PURCH", "RENOVATION", "TAKEOVER"],
            size=n, p=[0.45, 0.25, 0.15, 0.10, 0.05]
        ),
        "property_type": rng.choice(
            ["FLAT", "BUNGALOW", "PLOT", "COMMERCIAL"],
            size=n, p=[0.60, 0.20, 0.15, 0.05]
        ),
        "rate_type": rng.choice(["A", "F"], size=n, p=[0.80, 0.20]),
        "loan_scheme": rng.choice(["1", "2"], size=n, p=[0.92, 0.08]),
        "has_coborrower": rng.choice([0, 1], size=n, p=[0.55, 0.45]),
    })


# -----------------------------------------------------------------------------
# Behavioral generation per regime
# -----------------------------------------------------------------------------
def generate_behavior_for_loan(
    loan_id: str, regime: str, emi: float, local_rng: np.random.Generator,
) -> list[dict]:
    """Generate N_MONTHS_TOTAL monthly behavior rows for a single loan."""

    if regime == "clean":
        base_dishonor_p = 0.02
        base_dpd_mean, base_dpd_sigma = 2.0, 3.0
        deterioration_start = None
    elif regime == "chronic":
        base_dishonor_p = 0.18
        base_dpd_mean, base_dpd_sigma = 45.0, 10.0  # stuck-high plateau
        deterioration_start = None
    elif regime == "decliner":
        base_dishonor_p = 0.02
        base_dpd_mean, base_dpd_sigma = 2.0, 3.0
        deterioration_start = int(local_rng.integers(4, 11))
    else:
        raise ValueError(f"Unknown regime: {regime}")

    cur_roi = round(max(7.0, min(14.0, float(local_rng.normal(9.5, 1.2)))), 2)
    arrear_balance = 0.0
    rows: list[dict] = []

    for m in range(N_MONTHS_TOTAL):
        # Adjustable-rate loans occasionally see ROI changes
        if local_rng.random() < 0.03:
            cur_roi = round(cur_roi + float(local_rng.choice([-0.5, -0.25, 0.25, 0.5])), 2)
            cur_roi = max(7.0, min(15.0, cur_roi))

        # Decliner regime ramps up after deterioration_start
        if regime == "decliner" and deterioration_start is not None and m >= deterioration_start:
            ramp = min(1.0, (m - deterioration_start) / 12.0)
            dishonor_p = min(0.90, 0.05 + 0.70 * ramp)
            dpd_mean = 5.0 + 160.0 * ramp
            dpd_sigma = 5.0 + 25.0 * ramp
        else:
            dishonor_p = base_dishonor_p
            dpd_mean = base_dpd_mean
            dpd_sigma = base_dpd_sigma

        dishonored = int(local_rng.random() < dishonor_p)
        if dishonored:
            received = 0
            payment_amt = 0
            # dishonored months tend to push DPD higher
            dpd = max(0, int(local_rng.normal(dpd_mean + 25, dpd_sigma)))
            arrear_balance += emi
        else:
            received = 1
            payment_amt = int(round(emi * float(local_rng.uniform(0.95, 1.02))))
            dpd = max(0, int(local_rng.normal(dpd_mean, dpd_sigma)))
            arrear_balance = max(0.0, arrear_balance - max(0, payment_amt - emi))

        rows.append({
            "loan_id": loan_id,
            "month_idx": m,
            "payment_received": received,
            "dishonored": dishonored,
            "payment_amount": payment_amt,
            "days_past_due": dpd,
            "arrear_balance": int(round(arrear_balance)),
            "effective_roi": cur_roi,
        })

    return rows


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
def compute_targets(behavior_full: pd.DataFrame) -> pd.DataFrame:
    """target_Nm = 1 if any month in [obs, obs+N) has DPD > threshold."""
    obs = OBSERVATION_MONTH
    out_rows = []
    for loan_id, grp in behavior_full.groupby("loan_id", sort=False):
        f3 = grp[(grp["month_idx"] >= obs) & (grp["month_idx"] < obs + HORIZON_3M)]
        f6 = grp[(grp["month_idx"] >= obs) & (grp["month_idx"] < obs + HORIZON_6M)]
        out_rows.append({
            "loan_id": loan_id,
            "target_3m": int((f3["days_past_due"] > DEFAULT_DPD_THRESHOLD).any()),
            "target_6m": int((f6["days_past_due"] > DEFAULT_DPD_THRESHOLD).any()),
        })
    return pd.DataFrame(out_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"Seed: {SEED}")
    print(f"Generating {N_LOANS} loans, {N_MONTHS_TOTAL} months each...")

    loans = sample_static_features(N_LOANS)
    regimes = rng.choice(
        list(REGIME_PROBS.keys()),
        size=N_LOANS,
        p=list(REGIME_PROBS.values()),
    )

    print("Generating behavioral history (this takes ~30s)...")
    all_rows: list[dict] = []
    for i in range(N_LOANS):
        # Each loan gets its own seeded sub-generator for reproducibility
        local_rng = np.random.default_rng(SEED + i)
        rows = generate_behavior_for_loan(
            loan_id=loans.iloc[i]["loan_id"],
            regime=regimes[i],
            emi=float(loans.iloc[i]["emi"]),
            local_rng=local_rng,
        )
        all_rows.extend(rows)

    behavior_full = pd.DataFrame(all_rows)

    print("Computing targets from forward window...")
    targets = compute_targets(behavior_full)
    loans = loans.merge(targets, on="loan_id")
    loans["observation_month"] = OBSERVATION_MONTH

    # Mask future months — candidate only sees months 0..11
    behavior_visible = behavior_full[
        behavior_full["month_idx"] < OBSERVATION_MONTH
    ].copy()

    # -------- Outputs --------
    loans.to_csv("loans_static.csv", index=False)
    behavior_visible.to_csv("behavior_history.csv", index=False)
    print(f"\nWrote loans_static.csv      ({len(loans):,} rows)")
    print(f"Wrote behavior_history.csv  ({len(behavior_visible):,} rows)")

    # Hiring-team-only artifacts (DO NOT share with candidate)
    behavior_full.to_csv("_hiring_only_behavior_full.csv", index=False)
    pd.DataFrame({"loan_id": loans["loan_id"], "regime": regimes}).to_csv(
        "_hiring_only_regimes.csv", index=False
    )

    print("\n--- Summary ---")
    print(f"target_3m default rate: {loans['target_3m'].mean():.2%}")
    print(f"target_6m default rate: {loans['target_6m'].mean():.2%}")
    print("\nRegime distribution (hidden from candidate):")
    print(pd.Series(regimes).value_counts())
    print("\nDefault rate by regime (target_6m):")
    print(loans.assign(regime=regimes).groupby("regime")["target_6m"].mean())


if __name__ == "__main__":
    main()
