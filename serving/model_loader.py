"""
Model Loader: fetches the champion model from MLflow Model Registry.

The MLflow Model Registry stores versioned models with lifecycle stages:
    Staging → Validation → Production (Champion)

We always serve the "Production" stage model.
"""

import mlflow
import mlflow.sklearn
import os
from typing import Tuple


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit-risk-pd-lightgbm")


def load_champion_model() -> Tuple[object, str]:
    """
    Load the current Production (champion) model from MLflow registry.

    Returns
    -------
    model        : fitted sklearn-compatible model
    model_version: version string for logging/response headers
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/Production"
    print(f"Loading model: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    # Get version metadata
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    version = versions[0].version if versions else "unknown"

    print(f"Loaded {MODEL_NAME} v{version}")
    return model, f"v{version}"


def load_model_from_run(run_id: str) -> object:
    """Load a specific model by MLflow run ID (for A/B testing or rollback)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/lightgbm"
    return mlflow.sklearn.load_model(model_uri)
