"""
src/training/train.py
=====================

Train two LightGBM models (3m and 6m horizons), apply isotonic calibration,
evaluate on the test set, and persist all artifacts to models/.

Split: sort by issue_date, first 70% → train, last 30% → test. No random split.
Calibration: isotonic regression on a 20% hold-out from the training split.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.trajectory import build_trajectory_features

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

STATIC_FEATURES = [
    "loan_amt", "term_months", "roi_initial", "emi", "monthly_income",
    "dti", "borrower_age", "has_coborrower",
]
EMP_TYPE_MAP = {"E": 0, "S": 1, "M": 2, "R": 3}
LOAN_PURPOSE_MAP = {
    "FLAT_PURCH": 0, "CONSTRUCTION": 1, "PLOT_PURCH": 2,
    "RENOVATION": 3, "TAKEOVER": 4,
}
PROPERTY_TYPE_MAP = {"FLAT": 0, "BUNGALOW": 1, "PLOT": 2, "COMMERCIAL": 3}
RATE_TYPE_MAP = {"A": 0, "F": 1}
LOAN_SCHEME_MAP = {"1": 0, "2": 1}

TRAJ_FEATURES = [
    "dpd_slope_3m", "dpd_acceleration", "dishonor_slope_3m",
    "stuck_high_indicator", "arrear_balance_trend", "payment_gap_variance",
    "dpd_month11", "arrear_month11", "dishonor_rate_all", "mean_dpd_all",
]

ALL_FEATURES = STATIC_FEATURES + [
    "emp_type_enc", "loan_purpose_enc", "property_type_enc",
    "rate_type_enc", "loan_scheme_enc",
] + TRAJ_FEATURES


def precision_recall_at_top_k(y_true, y_prob, k_pct=0.10):
    """Precision and recall among the top k% by score."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    top_idx = np.argsort(y_prob)[-k:]
    tp = y_true[top_idx].sum()
    precision = tp / k
    recall = tp / max(1, y_true.sum())
    return precision, recall


def build_feature_matrix(loans: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    df = loans.merge(traj, on="loan_id", how="left")
    df["emp_type_enc"] = df["emp_type"].map(EMP_TYPE_MAP).fillna(0).astype(int)
    df["loan_purpose_enc"] = df["loan_purpose"].map(LOAN_PURPOSE_MAP).fillna(0).astype(int)
    df["property_type_enc"] = df["property_type"].map(PROPERTY_TYPE_MAP).fillna(0).astype(int)
    df["rate_type_enc"] = df["rate_type"].map(RATE_TYPE_MAP).fillna(0).astype(int)
    df["loan_scheme_enc"] = df["loan_scheme"].map(LOAN_SCHEME_MAP).fillna(0).astype(int)
    return df


def isotonic_calibrate(raw_probs_train: np.ndarray, y_train: np.ndarray,
                        raw_probs_apply: np.ndarray) -> tuple[IsotonicRegression, np.ndarray]:
    """Fit isotonic regression calibrator; return (calibrator, calibrated_probs)."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_probs_train, y_train)
    return ir, ir.transform(raw_probs_apply)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, label: str,
             threshold: float = 0.5) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    prec10, rec10 = precision_recall_at_top_k(y_true, y_prob, 0.10)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"\n── {label} ──────────────────────────────")
    print(f"  AUC-ROC              : {auc:.4f}")
    print(f"  Brier score          : {brier:.4f}")
    print(f"  Precision@Top10%     : {prec10:.4f}")
    print(f"  Recall@Top10%        : {rec10:.4f}")
    print(f"  Confusion matrix (threshold={threshold:.3f}):")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    return {"auc": auc, "brier": brier, "prec10": prec10, "rec10": rec10, "cm": cm}


def operational_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            portfolio_size: int = 15_000,
                            calls_per_week: int = 200) -> float:
    """
    Find threshold such that at most calls_per_week / portfolio_size * len(y_prob)
    loans are flagged — matching credit officer call capacity.

    We scale the budget proportionally to the test-set size.
    """
    n_test = len(y_prob)
    n_flag = max(1, int(round(calls_per_week * n_test / portfolio_size)))
    sorted_probs = np.sort(y_prob)
    threshold = float(sorted_probs[-n_flag])
    # Nudge slightly below so exactly n_flag loans meet or exceed threshold
    threshold = max(0.0, threshold - 1e-9)
    return threshold


def main() -> None:
    print("Loading data...")
    loans = pd.read_csv("loans_static.csv")
    behavior = pd.read_csv("behavior_history.csv")

    print("Building trajectory features...")
    traj = build_trajectory_features(behavior)

    print("Building feature matrix...")
    df = build_feature_matrix(loans, traj)

    # ── Time-based train/test split ──────────────────────────────────────────
    df = df.sort_values("issue_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

    # Within train, hold out last 20% for calibration
    cal_idx = int(len(train_df) * 0.80)
    fit_df = train_df.iloc[:cal_idx]
    cal_df = train_df.iloc[cal_idx:]

    X_fit = fit_df[ALL_FEATURES].values
    X_cal = cal_df[ALL_FEATURES].values
    X_test = test_df[ALL_FEATURES].values

    artifacts = {}

    for horizon in ("3m", "6m"):
        target_col = f"target_{horizon}"
        y_fit = fit_df[target_col].values
        y_cal = cal_df[target_col].values
        y_test = test_df[target_col].values

        print(f"\nTraining model_{horizon}...")
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        model.fit(X_fit, y_fit)

        # Raw probs on cal + test
        raw_cal = model.predict_proba(X_cal)[:, 1]
        raw_test = model.predict_proba(X_test)[:, 1]

        print(f"Calibrating model_{horizon} (isotonic)...")
        calibrator, cal_test_probs = isotonic_calibrate(raw_cal, y_cal, raw_test)

        # Brier before/after calibration
        brier_raw = brier_score_loss(y_test, raw_test)
        brier_cal = brier_score_loss(y_test, cal_test_probs)
        print(f"  Brier (raw)       : {brier_raw:.4f}")
        print(f"  Brier (calibrated): {brier_cal:.4f}")

        # Operational threshold
        op_thresh = operational_threshold(y_test, cal_test_probs)
        metrics = evaluate(y_test, cal_test_probs, f"model_{horizon} (calibrated)", op_thresh)
        metrics["brier_raw"] = brier_raw
        metrics["brier_calibrated"] = brier_cal
        metrics["op_threshold"] = op_thresh

        # Coverage at operational threshold
        flagged = (cal_test_probs >= op_thresh).sum()
        tp = ((cal_test_probs >= op_thresh) & (y_test == 1)).sum()
        precision_op = tp / max(1, flagged)
        recall_op = tp / max(1, y_test.sum())
        coverage = flagged / len(y_test)
        print(f"  Operational threshold: {op_thresh:.4f}")
        print(f"  Coverage : {coverage:.2%}  Precision: {precision_op:.4f}  Recall: {recall_op:.4f}")

        metrics.update({
            "coverage_op": float(coverage),
            "precision_op": float(precision_op),
            "recall_op": float(recall_op),
        })

        # Save model + calibrator + probs for test set
        pickle.dump(model, open(MODELS_DIR / f"lgbm_{horizon}.pkl", "wb"))
        pickle.dump(calibrator, open(MODELS_DIR / f"calibrator_{horizon}.pkl", "wb"))
        np.save(MODELS_DIR / f"test_probs_{horizon}.npy", cal_test_probs)

        artifacts[horizon] = {
            "metrics": metrics,
            "feature_names": ALL_FEATURES,
            "op_threshold": op_thresh,
        }

    # Save feature names + thresholds for serving
    json.dump(
        {
            "feature_names": ALL_FEATURES,
            "op_threshold_3m": artifacts["3m"]["op_threshold"],
            "op_threshold_6m": artifacts["6m"]["op_threshold"],
            "metrics": {k: v["metrics"] for k, v in artifacts.items()},
        },
        open(MODELS_DIR / "metadata.json", "w"),
        indent=2,
    )

    # Save test loan_ids for /explain endpoint
    test_df[["loan_id"] + ALL_FEATURES].to_parquet(MODELS_DIR / "test_features.parquet", index=False)

    # Save portfolio medians for /explain endpoint
    portfolio_medians = df[ALL_FEATURES].median().to_dict()
    json.dump(portfolio_medians, open(MODELS_DIR / "portfolio_medians.json", "w"), indent=2)

    print("\n✓ All artifacts saved to models/")


if __name__ == "__main__":
    main()
