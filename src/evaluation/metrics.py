"""
Credit Risk Model Evaluation Metrics.

Industry-standard metrics used in Basel IRB model validation:
  - AUC-ROC      : Discriminatory power
  - KS Statistic : Max separation between default/non-default score distributions
  - Gini         : 2 × AUC − 1 (Lorenz curve metric)
  - Brier Score  : Calibration quality (lower = better)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve
from typing import Dict


def compute_ks_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic.
    Max distance between CDF of defaulters and non-defaulters.
    Industry benchmark: KS > 0.30 is acceptable, > 0.50 is strong.
    """
    df = pd.DataFrame({"score": y_scores, "target": y_true})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["cum_events"] = df["target"].cumsum() / df["target"].sum()
    df["cum_non_events"] = (1 - df["target"]).cumsum() / (1 - df["target"]).sum()
    ks = (df["cum_events"] - df["cum_non_events"]).abs().max()
    return float(ks)


def compute_gini(auc: float) -> float:
    """Gini coefficient = 2 × AUC − 1."""
    return 2 * auc - 1


def compute_all_metrics(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Dict[str, float]:
    """Compute full suite of credit risk validation metrics."""
    auc = roc_auc_score(y_true, y_scores)
    ks = compute_ks_statistic(np.array(y_true), np.array(y_scores))
    gini = compute_gini(auc)
    brier = brier_score_loss(y_true, y_scores)

    return {
        "auc_roc": round(auc, 4),
        "ks_statistic": round(ks, 4),
        "gini": round(gini, 4),
        "brier_score": round(brier, 4),
    }


def plot_roc_curve(y_true, y_scores, model_name: str, ax=None):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_ks_chart(y_true, y_scores, model_name: str):
    """
    KS chart: cumulative % of events vs non-events by score decile.
    The max gap between the two curves = KS statistic.
    """
    df = pd.DataFrame({"score": y_scores, "target": y_true})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["cum_events"] = df["target"].cumsum() / df["target"].sum()
    df["cum_non_events"] = (1 - df["target"]).cumsum() / (1 - df["target"]).sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.index / len(df), df["cum_events"], label="Defaulters", color="red")
    ax.plot(df.index / len(df), df["cum_non_events"], label="Non-Defaulters", color="blue")
    ax.set_title(f"KS Chart — {model_name}")
    ax.set_xlabel("Population %")
    ax.set_ylabel("Cumulative %")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_calibration(y_true, y_scores, model_name: str):
    """
    Calibration plot: predicted PD vs actual default rate.
    Well-calibrated model = diagonal line.
    Critical for regulatory use.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted PD")
    ax.set_ylabel("Fraction of Defaults")
    ax.set_title("Calibration Plot")
    ax.legend()
    plt.tight_layout()
    return fig
