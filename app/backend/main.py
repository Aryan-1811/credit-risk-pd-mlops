"""
Credit Risk PD Scoring API — Standalone FastAPI Backend.
Loads model directly from bundled model_export.pkl.

Deploy to Render:
    1. Push this folder to GitHub
    2. Create new Web Service on Render
    3. Build command: pip install -r requirements.txt
    4. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Credit Risk PD Scoring API",
    description="Basel II-aligned Probability of Default model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfDependents",
]
LGD = 0.45


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_export.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()


class BorrowerInput(BaseModel):
    revolving_utilization: float = Field(..., ge=0.0, le=1.0)
    age: int = Field(..., ge=18, le=100)
    late_30_59: int = Field(..., ge=0)
    debt_ratio: float = Field(..., ge=0.0)
    monthly_income: float = Field(..., ge=0.0)
    open_credit_lines: int = Field(..., ge=0)
    dependents: int = Field(..., ge=0)


class PDResponse(BaseModel):
    pd_score: float
    pd_percent: float
    risk_band: str
    expected_loss: float
    decision: str
    stress: dict


def get_risk_band(pd_score: float):
    if pd_score < 0.05:
        return "LOW", "APPROVE"
    elif pd_score < 0.15:
        return "MEDIUM", "APPROVE WITH CONDITIONS"
    elif pd_score < 0.30:
        return "HIGH", "REFER TO UNDERWRITER"
    else:
        return "VERY HIGH", "DECLINE"


@app.get("/")
def root():
    return {"status": "ok", "model": "credit-risk-pd-lightgbm", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PDResponse)
def predict(borrower: BorrowerInput):
    df = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines":  borrower.revolving_utilization,
        "age":                                    borrower.age,
        "NumberOfTime30-59DaysPastDueNotWorse":  borrower.late_30_59,
        "DebtRatio":                             borrower.debt_ratio,
        "MonthlyIncome":                         borrower.monthly_income,
        "NumberOfOpenCreditLinesAndLoans":       borrower.open_credit_lines,
        "NumberOfDependents":                    borrower.dependents,
    }])[FEATURE_COLS]

    pd_score = float(model.predict_proba(df)[0, 1])
    risk_band, decision = get_risk_band(pd_score)
    el = pd_score * LGD

    stress = {
        "baseline":         round(pd_score, 4),
        "mild_stress":      round(min(pd_score * 1.3, 1.0), 4),
        "adverse":          round(min(pd_score * 1.7, 1.0), 4),
        "severely_adverse": round(min(pd_score * 2.5, 1.0), 4),
    }

    return PDResponse(
        pd_score=round(pd_score, 6),
        pd_percent=round(pd_score * 100, 2),
        risk_band=risk_band,
        expected_loss=round(el, 6),
        decision=decision,
        stress=stress,
    )
