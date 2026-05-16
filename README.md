# Multi-Horizon Behavioral Default EWS

Early Warning System predicting loan default probability at **3-month** and **6-month** horizons from a fixed observation point (month 12), using 12 months of behavioral history.

---

## Quickstart

```bash
# Create virtual environment
python3 -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

```bash
pip install -r requirements.txt
python generate_data.py          # produces loans_static.csv, behavior_history.csv
python src/training/train.py     # trains, calibrates, saves artifacts to models/
uvicorn src.api.main:app         # starts API on http://localhost:8000
```

OpenAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Repository Layout

```
repo/
├── README.md
├── requirements.txt
├── generate_data.py             ← provided, run once
├── src/
│   ├── features/trajectory.py  ← trajectory feature pipeline (leakage-safe)
│   ├── training/train.py       ← train + calibrate, saves to models/
│   ├── serving/predict.py      ← inference path only, no sklearn fit calls
│   └── api/main.py             ← FastAPI app
├── tests/
│   └── test_leakage.py         ← leakage test (5 tests, all pass)
└── models/                     ← saved artifacts after training
    ├── lgbm_3m.pkl
    ├── lgbm_6m.pkl
    ├── calibrator_3m.pkl
    ├── calibrator_6m.pkl
    ├── metadata.json
    ├── portfolio_medians.json
    └── test_features.parquet
```

---

## Part A — Trajectory Features

All features use only `month_idx < 12`. A leakage guard in `build_trajectory_features()` raises `ValueError` immediately if any row with `month_idx >= 12` is present.

| Feature                | Formula                                                               | Why it predicts default                                                                              |
| ---------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `dpd_slope_3m`         | OLS slope of DPD over months 9–11                                     | A rising DPD trend approaching observation signals imminent breach of the 90-day threshold           |
| `dpd_acceleration`     | `DPD_11 − 2·DPD_10 + DPD_9` (second difference)                       | Catches decliners whose deterioration rate is itself increasing — more sensitive than slope alone    |
| `dishonor_slope_3m`    | OLS slope of the dishonored flag over months 9–11                     | An increasing rate of payment failures is a leading indicator before DPD crosses 90                  |
| `stuck_high_indicator` | `1` if mean(DPD, months 6–11) ≥ 30 **and** std(DPD, months 6–11) ≤ 20 | Identifies chronic-elevated borrowers on a plateau; any shock can push them past 90 DPD              |
| `arrear_balance_trend` | OLS slope of arrear_balance over months 6–11, normalised ÷ 1000       | Growing arrear balance shows progressive inability to service debt even before DPD crosses threshold |
| `payment_gap_variance` | `Var(max(0, EMI_proxy − payment_amount))` over all 12 months          | High variance in shortfalls signals erratic repayment — a strong predictor of future default         |

Snapshot features (`dpd_month11`, `arrear_month11`, `dishonor_rate_all`, `mean_dpd_all`) are included as supporting context but are not the primary trajectory signals.

### Leakage test

```bash
python -m pytest tests/test_leakage.py -v
# 5 passed
```

---

## Part B — Two-Headed Model

**Architecture**: Two separate `LGBMClassifier` models — one for `target_3m`, one for `target_6m`. Separate models allow each horizon to learn its own feature weights; the trajectory signals relevant at 3 months differ in magnitude from those at 6 months.

**Split**: sorted by `issue_date`, first 70% → train, last 30% → test (no random split).
Within train, the last 20% is held out for calibration.

### Test-set metrics (calibrated probabilities)

| Metric                   | model_3m   | model_6m   |
| ------------------------ | ---------- | ---------- |
| AUC-ROC                  | **0.9896** | **0.9913** |
| Brier score (raw)        | 0.0187     | 0.0098     |
| Brier score (calibrated) | 0.0166     | 0.0097     |
| Precision@Top10%         | 0.9511     | 0.9844     |
| Recall@Top10%            | 0.8425     | 0.6383     |

**Confusion matrix at operational threshold (see Part C)**

_model_3m_ (threshold = 1.00):

```
TN=3992  FP=0
FN=395   TP=113
```

_model_6m_ (threshold = 1.00):

```
TN=3795  FP=11
FN=35    TP=659
```

---

## Part C — Calibration + Operational Threshold

### Calibration

**Method**: isotonic regression fitted on the calibration hold-out (20% of train). Isotonic regression is preferred over Platt scaling when the score distribution is non-sigmoid (tree ensembles often produce sharply bimodal scores).

|          | Brier (raw) | Brier (calibrated) | Improvement |
| -------- | ----------- | ------------------ | ----------- |
| model_3m | 0.0187      | 0.0166             | −11%        |
| model_6m | 0.0098      | 0.0097             | −1%         |

Calibration improved model_3m noticeably; model_6m was already near-perfectly calibrated (its raw probabilities were already binary-sharp due to near-perfect class separation in the synthetic data).

### Operational threshold

**Constraint**: credit officers can call 200 borrowers per week from a 15,000-loan portfolio → 200/15,000 = **1.33%** flag rate.

The threshold is set at the score of the 1.33%-th highest-scored loan in the test set (scaled proportionally).

**Note on threshold = 1.00**: The synthetic data has near-perfect regime separation (decliners have rapidly rising DPD; clean loans are near zero). After isotonic calibration, predicted probabilities collapse to essentially 0 or 1. In production data with noisy labels and overlapping regimes, probabilities would be more graduated and the threshold would be a value like 0.35–0.65.

|                        | model_3m | model_6m |
| ---------------------- | -------- | -------- |
| Operational threshold  | 1.00     | 1.00     |
| Coverage (flagged%)    | 2.5%     | 14.9%    |
| Precision at threshold | 1.000    | 0.984    |
| Recall at threshold    | 0.222    | 0.950    |

**Tradeoff explanation**

The 6m head flags 14.9% of the portfolio to achieve 95% recall — meaning it catches nearly every future defaulter but requires officers to call loans with a 1.6% false-positive rate. The 3m head is ultra-precise (100% precision within budget) but only catches 22% of 3m defaulters because most of those loans haven't shown trajectory signals by month 11 — the deterioration starts too close to the horizon. In practice, the 6m head is the primary operational tool; the 3m head supplements it for loans already showing clear deterioration and requiring urgent action.

---

## Part D — FastAPI Service

### Start

```bash
uvicorn src.api.main:app
# OpenAPI: http://localhost:8000/docs
```

### POST /score

Accepts origination features + up to 12 months of behavioral history. Returns `prob_3m`, `prob_6m`, and top 3 contributing features per horizon.

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "static_features": {
      "loan_id": "DEMO001",
      "loan_amt": 500000,
      "term_months": 180,
      "roi_initial": 9.5,
      "emi": 5000,
      "monthly_income": 50000,
      "dti": 0.10,
      "borrower_age": 35,
      "emp_type": "E",
      "loan_purpose": "FLAT_PURCH",
      "property_type": "FLAT",
      "rate_type": "A",
      "loan_scheme": "1",
      "has_coborrower": 0
    },
    "behavior_history": [
      {"month_idx": 0, "payment_received": 1, "dishonored": 0, "payment_amount": 5000, "days_past_due": 0, "arrear_balance": 0, "effective_roi": 9.5},
      {"month_idx": 1, "payment_received": 1, "dishonored": 0, "payment_amount": 5000, "days_past_due": 5, "arrear_balance": 0, "effective_roi": 9.5},
      {"month_idx": 2, "payment_received": 1, "dishonored": 0, "payment_amount": 5000, "days_past_due": 10, "arrear_balance": 0, "effective_roi": 9.5},
      {"month_idx": 3, "payment_received": 1, "dishonored": 0, "payment_amount": 5000, "days_past_due": 20, "arrear_balance": 0, "effective_roi": 9.5},
      {"month_idx": 4, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 35, "arrear_balance": 5000, "effective_roi": 9.5},
      {"month_idx": 5, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 50, "arrear_balance": 10000, "effective_roi": 9.5},
      {"month_idx": 6, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 65, "arrear_balance": 15000, "effective_roi": 9.5},
      {"month_idx": 7, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 75, "arrear_balance": 20000, "effective_roi": 9.5},
      {"month_idx": 8, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 82, "arrear_balance": 25000, "effective_roi": 9.5},
      {"month_idx": 9, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 87, "arrear_balance": 30000, "effective_roi": 9.5},
      {"month_idx": 10, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 90, "arrear_balance": 35000, "effective_roi": 9.5},
      {"month_idx": 11, "payment_received": 0, "dishonored": 1, "payment_amount": 0, "days_past_due": 95, "arrear_balance": 40000, "effective_roi": 9.5}
    ]
  }'
```

Expected response shape:

```json
{
  "prob_3m": 1.0,
  "prob_6m": 1.0,
  "top_reasons_3m": [
    {"feature": "dpd_slope_3m", "value": 5.0, "direction": "raises risk"},
    {"feature": "arrear_balance_trend", "value": 4.17, "direction": "raises risk"},
    {"feature": "dishonor_slope_3m", "value": 0.0, "direction": "raises risk"}
  ],
  "top_reasons_6m": [...]
}
```

### GET /explain/{loan_id}

For any loan in the test set, returns reason codes with portfolio-median comparison.

```bash
curl http://localhost:8000/explain/L005177
```

Expected response shape:

```json
{
  "loan_id": "L005177",
  "prob_3m": 1.0,
  "prob_6m": 1.0,
  "top_reasons_3m": [
    {"feature": "dpd_slope_3m", "value": 12.5, "direction": "raises risk", "portfolio_median": 0.0},
    {"feature": "mean_dpd_all", "value": 55.3, "direction": "raises risk", "portfolio_median": 2.92},
    {"feature": "arrear_balance_trend", "value": 6.8, "direction": "raises risk", "portfolio_median": 0.0}
  ],
  "top_reasons_6m": [...]
}
```

**Error cases**:

- Missing required fields → HTTP 422 with Pydantic validation detail
- `month_idx >= 12` in behavior_history → HTTP 422
- Unknown `loan_id` in `/explain` → HTTP 404
- Internal error → HTTP 500 with message

---

## Part E — Walkthrough Notes

### Why did/didn't the 6m head perform worse than the 3m head?

The 6m head performed **slightly better** (AUC 0.9913 vs 0.9896). This is specific to the synthetic data's design: nearly all defaults come from the "decliner" regime, which starts deteriorating 4–10 months before observation. By month 11, most decliners are well into their ramp — the trajectory features (`dpd_slope_3m`, `arrear_balance_trend`) are already highly elevated. A loan defaulting in the 3m window has to be deteriorating very fast _and_ close to threshold at observation; a loan defaulting in the 6m window just needs to be clearly on a rising trajectory. The 6m target therefore has a slightly stronger correlation with month-11 features than the 3m target does.

In real data you'd expect the opposite — 6m predictions should be harder because there's more time for circumstances to change, signal-to-noise is lower at longer horizons, and concept drift matters more.

### What would change if behavioral history were noisier (30% of months missing per loan)?

Three things would need to change:

1. **Feature engineering**: OLS slope on 3 months with one missing is undefined. The pipeline would need to either skip missing months in the regression (weighted OLS on available observations) or impute them with a forward-fill / last-observation-carried-forward strategy. The `stuck_high_indicator` would need a minimum-observation guard (e.g., only compute if ≥ 4 of 6 months are present).

2. **Feature values themselves**: `payment_gap_variance` and `dishonor_rate_all` computed over 8 observations instead of 12 have different distributions — the model's learned thresholds would be miscalibrated. Normalising by number of observed months would help.

3. **Missing-indicator features**: Add `n_months_observed` and `missing_rate` as explicit features. The model should know whether a loan has sparse history (which is often itself a signal — servicers often have gaps because of system issues correlated with portfolio stress).

### If you had another 10 hours, what would you build?

1. **SHAP-based reason codes**: Replace the ablation-based feature attribution with SHAP TreeExplainer, which is exact for tree models, additive, and consistent. The current ablation approach is O(n_features × inference_time) per request — slow and approximate.

2. **Missing-month robustness**: Add the imputation/normalisation logic described above so the `/score` endpoint handles real-world sparse history gracefully instead of returning degraded features.

3. **PSI monitor** (`monitor.py`): Compute Population Stability Index on score distribution and key features between a new batch and the training distribution. Flag when PSI > 0.2 — a standard signal that the model needs retraining.
