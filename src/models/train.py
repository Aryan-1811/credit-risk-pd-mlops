"""
Model Training with MLflow Experiment Tracking.

Models trained:
  1. Logistic Regression  — industry baseline / scorecard
  2. XGBoost              — gradient boosting benchmark
  3. LightGBM             — fast gradient boosting (your existing stack)

All runs logged to MLflow with params, metrics, and artefacts.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any

from src.evaluation.metrics import compute_all_metrics


MODELS = {
    "logistic_regression": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        random_state=42,
        verbose=-1,
    ),
}


def train_and_log(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    calibrate: bool = True,
    experiment_name: str = "credit-risk-pd",
) -> Dict[str, Any]:
    """
    Train a model, optionally calibrate probabilities, and log everything to MLflow.

    Calibration is important for regulatory use — predicted PDs must be
    reliable point estimates, not just rankings.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        model = MODELS[model_name]

        # Calibrate with Platt scaling for well-calibrated probabilities
        if calibrate and model_name == "logistic_regression":
            # LR is already probabilistic; skip extra calibration
            final_model = model
        elif calibrate:
            final_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
        else:
            final_model = model

        final_model.fit(X_train, y_train)
        y_pred_proba = final_model.predict_proba(X_val)[:, 1]

        metrics = compute_all_metrics(y_val, y_pred_proba)

        # Log to MLflow
        mlflow.log_params({"model": model_name, "calibrated": calibrate})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(final_model, artifact_path=model_name)

        print(f"\n{'='*50}")
        print(f"Model: {model_name.upper()}")
        print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
        print(f"  KS Stat : {metrics['ks_statistic']:.4f}")
        print(f"  Gini    : {metrics['gini']:.4f}")
        print(f"  Brier   : {metrics['brier_score']:.4f}")
        print(f"{'='*50}")

        return {"model": final_model, "metrics": metrics}


def train_all_models(
    X_train, y_train, X_val, y_val
) -> Dict[str, Dict]:
    """Train and compare all three models."""
    results = {}
    for name in MODELS:
        results[name] = train_and_log(name, X_train, y_train, X_val, y_val)
    return results
