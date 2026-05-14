"""
src/serving/predict.py
======================

Inference path only — no model fitting.

Loads pre-trained LightGBM models + isotonic calibrators from models/.
Provides score() and explain() functions consumed by the FastAPI app.
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TRAJ_FEATURES = [
    "dpd_slope_3m", "dpd_acceleration", "dishonor_slope_3m",
    "stuck_high_indicator", "arrear_balance_trend", "payment_gap_variance",
    "dpd_month11", "arrear_month11", "dishonor_rate_all", "mean_dpd_all",
]

EMP_TYPE_MAP = {"E": 0, "S": 1, "M": 2, "R": 3}
LOAN_PURPOSE_MAP = {
    "FLAT_PURCH": 0, "CONSTRUCTION": 1, "PLOT_PURCH": 2,
    "RENOVATION": 3, "TAKEOVER": 4,
}
PROPERTY_TYPE_MAP = {"FLAT": 0, "BUNGALOW": 1, "PLOT": 2, "COMMERCIAL": 3}
RATE_TYPE_MAP = {"A": 0, "F": 1}
LOAN_SCHEME_MAP = {"1": 0, "2": 1}


@lru_cache(maxsize=1)
def _load_artifacts() -> dict[str, Any]:
    meta = json.load(open(MODELS_DIR / "metadata.json"))
    models, calibrators = {}, {}
    for h in ("3m", "6m"):
        models[h] = pickle.load(open(MODELS_DIR / f"lgbm_{h}.pkl", "rb"))
        calibrators[h] = pickle.load(open(MODELS_DIR / f"calibrator_{h}.pkl", "rb"))
    test_df = pd.read_parquet(MODELS_DIR / "test_features.parquet")
    portfolio_medians = json.load(open(MODELS_DIR / "portfolio_medians.json"))
    return {
        "meta": meta,
        "models": models,
        "calibrators": calibrators,
        "test_df": test_df,
        "portfolio_medians": portfolio_medians,
    }


def _build_feature_vector(static: dict, traj: dict, feature_names: list[str]) -> np.ndarray:
    row = {**static}
    row["emp_type_enc"] = EMP_TYPE_MAP.get(static.get("emp_type", "E"), 0)
    row["loan_purpose_enc"] = LOAN_PURPOSE_MAP.get(static.get("loan_purpose", "FLAT_PURCH"), 0)
    row["property_type_enc"] = PROPERTY_TYPE_MAP.get(static.get("property_type", "FLAT"), 0)
    row["rate_type_enc"] = RATE_TYPE_MAP.get(static.get("rate_type", "A"), 0)
    row["loan_scheme_enc"] = LOAN_SCHEME_MAP.get(str(static.get("loan_scheme", "1")), 0)
    row.update(traj)
    return np.array([row.get(f, 0.0) for f in feature_names], dtype=float).reshape(1, -1)


def _top_reasons(
    model, feature_vector: np.ndarray, feature_names: list[str], top_n: int = 3
) -> list[dict]:
    """
    Derive top_n feature contributions using the difference between
    the loan's predicted probability and the model's base (mean) score.

    We use SHAP-light approach: feature importances weighted by feature deviation
    from the training mean (approximated by portfolio medians).

    For a production system, full SHAP would be used; here we use a fast
    LightGBM leaf-path contribution approximation via predict_proba
    with each feature ablated to its median (leave-one-in style).
    """
    arts = _load_artifacts()
    medians = arts["portfolio_medians"]
    fn = feature_names

    base_prob = model.predict_proba(feature_vector)[0, 1]
    contributions = []
    for i, feat in enumerate(fn):
        ablated = feature_vector.copy()
        ablated[0, i] = medians.get(feat, 0.0)
        ablated_prob = model.predict_proba(ablated)[0, 1]
        contributions.append((feat, float(feature_vector[0, i]), base_prob - ablated_prob))

    # Sort by absolute contribution, descending
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    result = []
    for feat, value, contrib in contributions[:top_n]:
        direction = "raises risk" if contrib > 0 else "lowers risk"
        result.append({"feature": feat, "value": round(value, 4), "direction": direction})
    return result


def score(static_features: dict, behavior_rows: list[dict]) -> dict:
    """
    Score a single loan.

    Parameters
    ----------
    static_features : dict  — origination features
    behavior_rows   : list of dicts — up to 12 monthly records (month_idx 0..11)

    Returns
    -------
    dict with prob_3m, prob_6m, top_reasons_3m, top_reasons_6m
    """
    from src.features.trajectory import build_trajectory_features

    arts = _load_artifacts()
    feature_names: list[str] = arts["meta"]["feature_names"]

    # Build trajectory
    beh_df = pd.DataFrame(behavior_rows)
    if beh_df.empty or "loan_id" not in beh_df.columns:
        beh_df["loan_id"] = "TEMP"
        static_features = dict(static_features)

    # Ensure loan_id present
    loan_id = static_features.get("loan_id", "TEMP")
    if "loan_id" not in beh_df.columns:
        beh_df["loan_id"] = loan_id

    for col in ["days_past_due", "dishonored", "payment_amount", "arrear_balance",
                "payment_received", "effective_roi", "month_idx"]:
        if col not in beh_df.columns:
            beh_df[col] = 0

    traj_df = build_trajectory_features(beh_df)
    traj = traj_df.iloc[0].to_dict() if len(traj_df) > 0 else {f: 0.0 for f in TRAJ_FEATURES}

    fv = _build_feature_vector(static_features, traj, feature_names)

    probs, reasons = {}, {}
    for h in ("3m", "6m"):
        model = arts["models"][h]
        calibrator = arts["calibrators"][h]
        raw_prob = model.predict_proba(fv)[0, 1]
        cal_prob = float(calibrator.transform([raw_prob])[0])
        probs[h] = round(cal_prob, 4)
        reasons[h] = _top_reasons(model, fv, feature_names)

    return {
        "prob_3m": probs["3m"],
        "prob_6m": probs["6m"],
        "top_reasons_3m": reasons["3m"],
        "top_reasons_6m": reasons["6m"],
    }


def explain(loan_id: str) -> dict:
    """
    Explain a test-set loan. Returns reason codes + portfolio median comparison.

    Raises KeyError if loan_id not in test set.
    """
    arts = _load_artifacts()
    test_df: pd.DataFrame = arts["test_df"]
    feature_names: list[str] = arts["meta"]["feature_names"]
    medians: dict = arts["portfolio_medians"]

    row = test_df[test_df["loan_id"] == loan_id]
    if row.empty:
        raise KeyError(f"loan_id '{loan_id}' not found in test set")

    fv = row[feature_names].values.astype(float)

    result = {"loan_id": loan_id}
    for h in ("3m", "6m"):
        model = arts["models"][h]
        calibrator = arts["calibrators"][h]
        raw_prob = model.predict_proba(fv)[0, 1]
        cal_prob = float(calibrator.transform([raw_prob])[0])
        top = _top_reasons(model, fv, feature_names)
        # Augment with portfolio median
        for item in top:
            item["portfolio_median"] = round(medians.get(item["feature"], 0.0), 4)
        result[f"prob_{h}"] = round(cal_prob, 4)
        result[f"top_reasons_{h}"] = top

    return result
