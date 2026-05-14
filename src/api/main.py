"""
src/api/main.py
===============

FastAPI service exposing:
  POST /score      — score a single loan (static + behavior)
  GET  /explain/{loan_id} — explain a test-set loan with portfolio medians
  GET  /docs       — OpenAPI (automatic)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

app = FastAPI(
    title="Multi-Horizon Behavioral EWS",
    description="Early Warning System: predicts P(default) at 3m and 6m horizons.",
    version="1.0.0",
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class BehaviorRecord(BaseModel):
    loan_id: str | None = None
    month_idx: int = Field(..., ge=0, lt=12, description="Month index (0-11 only)")
    payment_received: int = Field(default=0)
    dishonored: int = Field(default=0)
    payment_amount: float = Field(default=0.0)
    days_past_due: float = Field(default=0.0)
    arrear_balance: float = Field(default=0.0)
    effective_roi: float = Field(default=9.5)


class StaticFeatures(BaseModel):
    loan_id: str | None = None
    loan_amt: float
    term_months: int
    roi_initial: float
    emi: float
    monthly_income: float
    dti: float
    borrower_age: int
    emp_type: str = Field(default="E", description="E/S/M/R")
    loan_purpose: str = Field(default="FLAT_PURCH")
    property_type: str = Field(default="FLAT")
    rate_type: str = Field(default="A", description="A (adjustable) or F (fixed)")
    loan_scheme: str = Field(default="1")
    has_coborrower: int = Field(default=0)


class ScoreRequest(BaseModel):
    static_features: StaticFeatures
    behavior_history: list[BehaviorRecord] = Field(
        ..., min_length=1, max_length=12,
        description="Monthly records for months 0-11 only"
    )


class ReasonCode(BaseModel):
    feature: str
    value: float
    direction: str  # "raises risk" | "lowers risk"
    portfolio_median: float | None = None


class ScoreResponse(BaseModel):
    prob_3m: float
    prob_6m: float
    top_reasons_3m: list[ReasonCode]
    top_reasons_6m: list[ReasonCode]


class ExplainResponse(BaseModel):
    loan_id: str
    prob_3m: float
    prob_6m: float
    top_reasons_3m: list[ReasonCode]
    top_reasons_6m: list[ReasonCode]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse, summary="Score a loan at 3m and 6m horizon")
def score_loan(request: ScoreRequest) -> ScoreResponse:
    """
    Accept a loan's static origination features and up to 12 months of behavioral
    history (month_idx 0-11). Returns default probabilities and top contributing
    features for both horizons.
    """
    from src.serving.predict import score

    static_dict = request.static_features.model_dump()
    behavior_list = [r.model_dump() for r in request.behavior_history]

    # Validate month_idx range
    bad_months = [r["month_idx"] for r in behavior_list if r["month_idx"] >= 12]
    if bad_months:
        raise HTTPException(
            status_code=422,
            detail=f"behavior_history contains month_idx >= 12: {bad_months}. "
                   "Only months 0-11 are permitted."
        )

    try:
        result = score(static_dict, behavior_list)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ScoreResponse(
        prob_3m=result["prob_3m"],
        prob_6m=result["prob_6m"],
        top_reasons_3m=[ReasonCode(**r) for r in result["top_reasons_3m"]],
        top_reasons_6m=[ReasonCode(**r) for r in result["top_reasons_6m"]],
    )


@app.get(
    "/explain/{loan_id}",
    response_model=ExplainResponse,
    summary="Explain a test-set loan with portfolio median comparison",
)
def explain_loan(loan_id: str) -> ExplainResponse:
    """
    For any loan in the test set, returns predicted probabilities plus the top 3
    contributing features per horizon, each annotated with the portfolio median
    so a credit officer can see 'DTI 38% vs portfolio median 22%'.
    """
    from src.serving.predict import explain

    try:
        result = explain(loan_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ExplainResponse(
        loan_id=result["loan_id"],
        prob_3m=result["prob_3m"],
        prob_6m=result["prob_6m"],
        top_reasons_3m=[ReasonCode(**r) for r in result["top_reasons_3m"]],
        top_reasons_6m=[ReasonCode(**r) for r in result["top_reasons_6m"]],
    )


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}
