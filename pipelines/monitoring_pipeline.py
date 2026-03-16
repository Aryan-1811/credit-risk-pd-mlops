"""
Prefect Monitoring Pipeline.

Runs on a weekly schedule to:
    1. Load reference data (training baseline)
    2. Load recent production scoring data
    3. Compute PSI on scores and features
    4. If drift detected → trigger retraining pipeline
    5. Save report to monitoring/reports/

Schedule: Weekly (every Monday 06:00 UTC)

Deploy:
    prefect deployment build pipelines/monitoring_pipeline.py:monitoring_pipeline \
        --name "weekly-drift-check" --cron "0 6 * * 1"
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from pathlib import Path
from prefect import flow, task, get_run_logger

from monitoring.drift_detector import run_psi_report, PSI_THRESHOLDS
from pipelines.training_pipeline import training_pipeline


REFERENCE_FEATURES_PATH = "data/reference/reference_features.parquet"
REFERENCE_SCORES_PATH = "data/reference/reference_scores.npy"
PRODUCTION_SCORES_PATH = "data/processed/production_scores.npy"
PRODUCTION_FEATURES_PATH = "data/processed/production_features.parquet"


@task(name="load-reference-data")
def load_reference_data():
    logger = get_run_logger()

    if not Path(REFERENCE_FEATURES_PATH).exists():
        logger.warning("Reference data not found — skipping drift check")
        return None, None

    ref_features = pd.read_parquet(REFERENCE_FEATURES_PATH)
    ref_scores = np.load(REFERENCE_SCORES_PATH)
    logger.info(f"Reference data loaded: {len(ref_scores):,} samples")
    return ref_features, ref_scores


@task(name="load-production-data")
def load_production_data():
    """
    In a real deployment, this task would query a feature store or
    data warehouse for the last week's scoring requests.
    Here we simulate with saved numpy arrays.
    """
    logger = get_run_logger()

    if not Path(PRODUCTION_SCORES_PATH).exists():
        logger.warning("No production data found — generating synthetic data for demo")
        # Simulate mild drift for demonstration
        prod_scores = np.random.beta(2, 10, size=1000)
        prod_features = None
    else:
        prod_scores = np.load(PRODUCTION_SCORES_PATH)
        prod_features = (
            pd.read_parquet(PRODUCTION_FEATURES_PATH)
            if Path(PRODUCTION_FEATURES_PATH).exists()
            else None
        )

    logger.info(f"Production data loaded: {len(prod_scores):,} samples")
    return prod_features, prod_scores


@task(name="run-drift-detection")
def detect_drift(ref_features, ref_scores, prod_features, prod_scores) -> dict:
    logger = get_run_logger()

    if ref_scores is None or prod_scores is None:
        logger.warning("Missing data — skipping drift detection")
        return {"retrain_recommended": False, "score_psi": 0.0}

    report = run_psi_report(
        reference_scores=ref_scores,
        current_scores=prod_scores,
        reference_features=ref_features,
        current_features=prod_features,
        output_dir="monitoring/reports",
    )

    logger.info(f"Score PSI: {report['score_psi']:.4f} — {report['score_status']}")
    return report


@task(name="trigger-retraining-if-needed")
def trigger_retraining(drift_report: dict):
    logger = get_run_logger()

    if drift_report.get("retrain_recommended", False):
        logger.warning(
            f"PSI={drift_report['score_psi']:.4f} exceeds threshold "
            f"{PSI_THRESHOLDS['moderate']}. Triggering retraining pipeline."
        )
        training_pipeline()  # Trigger retraining as a subflow
    else:
        logger.info("No retraining needed. Model is stable.")


@flow(name="credit-risk-monitoring-pipeline", log_prints=True)
def monitoring_pipeline():
    """Weekly drift monitoring flow."""
    ref_features, ref_scores = load_reference_data()
    prod_features, prod_scores = load_production_data()
    drift_report = detect_drift(ref_features, ref_scores, prod_features, prod_scores)
    trigger_retraining(drift_report)
    print(f"\nMonitoring complete. Retrain triggered: {drift_report.get('retrain_recommended')}")


if __name__ == "__main__":
    monitoring_pipeline()
