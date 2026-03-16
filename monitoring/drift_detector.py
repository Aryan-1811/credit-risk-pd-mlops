"""
Model Monitoring: Data Drift & Population Stability Index (PSI).

PSI is the credit industry standard for detecting score/feature drift.
It's used by risk teams to decide when a model needs recalibration.

PSI Thresholds (industry standard):
    PSI < 0.10  → No significant change, model stable
    PSI 0.10-0.25 → Moderate shift, investigate
    PSI > 0.25  → Major shift, model likely needs retraining
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import ColumnDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: evidently not installed. PSI-only mode active.")


PSI_THRESHOLDS = {
    "stable":   0.10,
    "moderate": 0.25,
}


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Parameters
    ----------
    expected : reference distribution (training data scores/feature)
    actual   : current distribution (production data scores/feature)
    n_bins   : number of buckets (10 = deciles, standard in credit)
    """
    # Create bins from reference distribution
    breakpoints = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(expected, breakpoints)
    bins = np.unique(bins)  # Remove duplicates

    # Compute bucket frequencies
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    # Convert to proportions, smooth to avoid log(0)
    expected_pct = (expected_counts + 0.0001) / len(expected)
    actual_pct = (actual_counts + 0.0001) / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def interpret_psi(psi: float) -> str:
    if psi < PSI_THRESHOLDS["stable"]:
        return "STABLE — No action needed"
    elif psi < PSI_THRESHOLDS["moderate"]:
        return "MODERATE DRIFT — Investigate feature distributions"
    else:
        return "MAJOR DRIFT — Model retraining recommended"


def run_psi_report(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    reference_features: Optional[pd.DataFrame] = None,
    current_features: Optional[pd.DataFrame] = None,
    output_dir: str = "monitoring/reports",
) -> dict:
    """
    Run full PSI report on model scores and optionally all features.
    Saves JSON summary + HTML report (if Evidently available).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Score-level PSI
    score_psi = compute_psi(reference_scores, current_scores)
    result = {
        "timestamp": timestamp,
        "score_psi": round(score_psi, 4),
        "score_status": interpret_psi(score_psi),
        "feature_psi": {},
        "retrain_recommended": score_psi > PSI_THRESHOLDS["moderate"],
    }

    # Feature-level PSI
    if reference_features is not None and current_features is not None:
        for col in reference_features.columns:
            if col in current_features.columns:
                feat_psi = compute_psi(
                    reference_features[col].values,
                    current_features[col].values
                )
                result["feature_psi"][col] = {
                    "psi": round(feat_psi, 4),
                    "status": interpret_psi(feat_psi)
                }

    # Save JSON summary
    summary_path = Path(output_dir) / f"psi_report_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"PSI report saved: {summary_path}")

    # Evidently HTML report (richer visualisation)
    if EVIDENTLY_AVAILABLE and reference_features is not None:
        _run_evidently_report(
            reference_features, current_features, output_dir, timestamp
        )

    # Print summary
    print(f"\n{'='*50}")
    print(f"DRIFT MONITORING REPORT — {timestamp}")
    print(f"  Score PSI : {score_psi:.4f}  →  {interpret_psi(score_psi)}")
    if result["feature_psi"]:
        drifted = [
            f for f, v in result["feature_psi"].items()
            if v["psi"] > PSI_THRESHOLDS["stable"]
        ]
        print(f"  Drifted features ({len(drifted)}): {', '.join(drifted) or 'None'}")
    print(f"  Retrain recommended: {result['retrain_recommended']}")
    print(f"{'='*50}\n")

    return result


def _run_evidently_report(ref_df, cur_df, output_dir, timestamp):
    """Generate rich Evidently HTML drift report."""
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    html_path = Path(output_dir) / f"evidently_report_{timestamp}.html"
    report.save_html(str(html_path))
    print(f"Evidently report saved: {html_path}")
