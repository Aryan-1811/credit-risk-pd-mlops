"""
FastAPI PD Scoring Service.

Endpoints:
    GET  /health         — liveness check
    POST /predict        — single borrower PD score
    POST /predict/batch  — batch PD scoring

Run locally:
    uvicorn serving.api:app --reload --port 8000

Docker:
    docker build -t credit-risk-pd .
    docker run -p 8000:8000 credit-risk-pd
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from serving.schemas import (
    BorrowerFeatures, PDResponse,
    BatchPDRequest, BatchPDResponse,
    HealthResponse
)
from serving.model_loader import load_champion_model


# ── Global model state ────────────────────────────────────────────────────────
model = None
model_version = "unknown"

FEATURE_COLUMNS = [
    "revolving_utilization", "age", "num_30_59_days_late",
    "debt_ratio", "monthly_income", "num_open_credit_lines",
    "num_90_days_late", "num_real_estate_loans",
    "num_60_89_days_late", "num_dependents"
]

LGD = 0.45  # Basel II floor for unsecured retail


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global model, model_version
    model, model_version = load_champion_model()
    yield
    model = None


app = FastAPI(
    title="Credit Risk PD Scoring API",
    description="Predicts Probability of Default (PD) for retail borrowers.",
    version="1.0.0",
    lifespan=lifespan,
)


def _to_dataframe(borrower: BorrowerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame row for scoring."""
    data = {
        "revolving_utilization": [borrower.revolving_utilization],
        "age": [borrower.age],
        "num_30_59_days_late": [borrower.num_30_59_days_late],
        "debt_ratio": [borrower.debt_ratio],
        "monthly_income": [borrower.monthly_income if borrower.monthly_income else 5000.0],
        "num_open_credit_lines": [borrower.num_open_credit_lines],
        "num_90_days_late": [borrower.num_90_days_late],
        "num_real_estate_loans": [borrower.num_real_estate_loans],
        "num_60_89_days_late": [borrower.num_60_89_days_late],
        "num_dependents": [borrower.num_dependents if borrower.num_dependents else 0],
    }
    return pd.DataFrame(data)


def _get_risk_band(pd_score: float) -> str:
    """Map PD score to risk band (standard credit risk segmentation)."""
    if pd_score < 0.05:
        return "LOW"
    elif pd_score < 0.15:
        return "MEDIUM"
    elif pd_score < 0.30:
        return "HIGH"
    else:
        return "VERY HIGH"


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=model_version,
    )


@app.post("/predict", response_model=PDResponse)
def predict(borrower: BorrowerFeatures):
    """Score a single borrower and return PD, risk band, and expected loss."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = _to_dataframe(borrower)
        pd_score = float(model.predict_proba(X)[0, 1])
        el = pd_score * LGD * 1.0  # EAD = 1 unit of exposure

        return PDResponse(
            pd_score=round(pd_score, 6),
            risk_band=_get_risk_band(pd_score),
            expected_loss=round(el, 6),
            model_version=model_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPDResponse)
def predict_batch(request: BatchPDRequest):
    """Score multiple borrowers in one request."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for borrower in request.borrowers:
        X = _to_dataframe(borrower)
        pd_score = float(model.predict_proba(X)[0, 1])
        el = pd_score * LGD
        results.append(PDResponse(
            pd_score=round(pd_score, 6),
            risk_band=_get_risk_band(pd_score),
            expected_loss=round(el, 6),
            model_version=model_version,
        ))

    return BatchPDResponse(
        results=results,
        count=len(results),
        model_version=model_version,
    )
