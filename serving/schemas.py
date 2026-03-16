"""
Pydantic schemas for the PD scoring API.
Defines the structure of requests and responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class BorrowerFeatures(BaseModel):
    """
    Input features for a single borrower PD scoring request.
    Based on Give Me Some Credit dataset schema.
    Adapt field names to match your actual dataset.
    """
    revolving_utilization: float = Field(
        ..., ge=0.0, le=1.0,
        description="Revolving lines utilisation rate (0–1)"
    )
    age: int = Field(..., ge=18, le=100, description="Borrower age")
    num_30_59_days_late: int = Field(
        ..., ge=0, description="Times 30–59 days past due (last 2 years)"
    )
    debt_ratio: float = Field(..., ge=0.0, description="Debt ratio")
    monthly_income: Optional[float] = Field(
        None, ge=0.0, description="Monthly income (£)"
    )
    num_open_credit_lines: int = Field(..., ge=0)
    num_90_days_late: int = Field(..., ge=0)
    num_real_estate_loans: int = Field(..., ge=0)
    num_60_89_days_late: int = Field(..., ge=0)
    num_dependents: Optional[int] = Field(None, ge=0)

    @field_validator("monthly_income", "num_dependents", mode="before")
    @classmethod
    def handle_none(cls, v):
        return v  # Allow None; preprocessing handles imputation


class PDResponse(BaseModel):
    """API response containing the predicted Probability of Default."""
    pd_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Predicted Probability of Default (0–1)"
    )
    risk_band: str = Field(
        ..., description="Risk classification: LOW / MEDIUM / HIGH / VERY HIGH"
    )
    expected_loss: float = Field(
        ..., description="Expected Loss = PD × LGD × EAD (assuming LGD=0.45, EAD=1)"
    )
    model_version: str = Field(..., description="MLflow model version used")
    status: str = "success"


class BatchPDRequest(BaseModel):
    """Batch scoring request for multiple borrowers."""
    borrowers: list[BorrowerFeatures]


class BatchPDResponse(BaseModel):
    """Batch scoring response."""
    results: list[PDResponse]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
