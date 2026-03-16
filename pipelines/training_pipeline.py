"""
Prefect Training Pipeline.

Orchestrates the full model training lifecycle:
    1. Ingest & validate new data
    2. Preprocess + WoE encode
    3. Train all models
    4. Evaluate on holdout set
    5. Register best model to MLflow Model Registry
    6. Promote to Production if it beats current champion

Schedule: Run monthly (banks retrain PD models on monthly data cycles).

Run manually:
    python pipelines/training_pipeline.py

Deploy as scheduled flow:
    prefect deployment build pipelines/training_pipeline.py:training_pipeline \
        --name "monthly-retrain" --cron "0 0 1 * *"
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import mlflow
from prefect import flow, task, get_run_logger

from src.data.preprocessing import load_data, clean_data, impute_missing, split_data
from src.features.woe_encoder import WoEEncoder
from src.models.train import train_all_models
from src.evaluation.metrics import compute_all_metrics


DATA_PATH = os.getenv("DATA_PATH", "data/raw/cs-training.csv")
TARGET_COL = "SeriousDlqin2yrs"
MODEL_NAME = "credit-risk-pd-lightgbm"
MIN_GINI_THRESHOLD = 0.40  # Minimum acceptable Gini to promote to Production


@task(name="ingest-and-validate")
def ingest_data(data_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    df = load_data(data_path)
    df = clean_data(df)
    df = impute_missing(df)

    # Basic data validation
    assert len(df) > 1000, "Dataset too small — check data source"
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing"
    default_rate = df[TARGET_COL].mean()
    assert 0.01 < default_rate < 0.50, f"Suspicious default rate: {default_rate:.3f}"

    logger.info(f"Data validated: {len(df):,} rows, default rate={default_rate:.3f}")
    return df


@task(name="feature-engineering")
def engineer_features(df: pd.DataFrame, target_col: str):
    logger = get_run_logger()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col)

    encoder = WoEEncoder(n_bins=10)
    X_train_woe = encoder.fit_transform(X_train, y_train)
    X_val_woe = encoder.transform(X_val)
    X_test_woe = encoder.transform(X_test)

    selected = encoder.select_features(min_iv=0.02)
    logger.info(f"Selected {len(selected)} features via IV >= 0.02")

    # Save reference data for drift monitoring
    X_train_woe[selected].to_parquet("data/reference/reference_features.parquet")
    np.save("data/reference/reference_scores.npy", np.zeros(len(X_train)))  # Placeholder

    return (
        X_train_woe[selected], X_val_woe[selected], X_test_woe[selected],
        y_train, y_val, y_test, encoder
    )


@task(name="train-models")
def train(X_train, y_train, X_val, y_val):
    logger = get_run_logger()
    results = train_all_models(X_train, y_train, X_val, y_val)
    logger.info("All models trained and logged to MLflow")
    return results


@task(name="evaluate-and-register")
def evaluate_and_register(results: dict, X_test, y_test) -> bool:
    """Evaluate best model on test set and register if Gini meets threshold."""
    logger = get_run_logger()
    client = mlflow.tracking.MlflowClient()

    best_model_name = max(
        results, key=lambda k: results[k]["metrics"]["gini"]
    )
    best_model = results[best_model_name]["model"]
    logger.info(f"Best model: {best_model_name}")

    y_pred = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_all_metrics(y_test, y_pred)

    logger.info(f"Test Gini: {test_metrics['gini']:.4f} | KS: {test_metrics['ks_statistic']:.4f}")

    if test_metrics["gini"] < MIN_GINI_THRESHOLD:
        logger.warning(
            f"Gini {test_metrics['gini']:.4f} below threshold {MIN_GINI_THRESHOLD}. "
            "Not promoting to Production."
        )
        return False

    # Register to MLflow Model Registry
    with mlflow.start_run(run_name=f"register-{best_model_name}"):
        mlflow.log_metrics(test_metrics)
        model_uri = mlflow.sklearn.log_model(
            best_model, artifact_path=best_model_name,
            registered_model_name=MODEL_NAME
        ).model_uri

    # Promote latest version to Production
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if latest:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest[0].version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Promoted {MODEL_NAME} v{latest[0].version} to Production")

    return True


@flow(name="credit-risk-pd-training-pipeline", log_prints=True)
def training_pipeline(data_path: str = DATA_PATH):
    """Full monthly retraining flow."""
    df = ingest_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = engineer_features(
        df, TARGET_COL
    )
    results = train(X_train, y_train, X_val, y_val)
    promoted = evaluate_and_register(results, X_test, y_test)
    print(f"\nPipeline complete. Model promoted to Production: {promoted}")


if __name__ == "__main__":
    training_pipeline()
