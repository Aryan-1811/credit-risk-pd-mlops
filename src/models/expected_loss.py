"""
Expected Loss (EL) Computation and Macro Stress Testing.

Basel II formula:
    EL = PD × LGD × EAD

Stress Testing:
    Shift macro variables (unemployment, interest rate) and observe
    how PD changes under adverse scenarios — directly mirrors what
    Statistical Modelling Quants do for IFRS 9 / CCAR stress tests.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List


def compute_expected_loss(
    pd_scores: np.ndarray,
    lgd: float = 0.45,       # Basel II foundation IRB assumption for unsecured
    ead: np.ndarray = None,  # Exposure at Default; defaults to £1 per borrower
) -> pd.DataFrame:
    """
    Compute Expected Loss for each borrower.

    Parameters
    ----------
    pd_scores : array of model-predicted PDs (0–1)
    lgd       : Loss Given Default (scalar or array). 0.45 = Basel II floor.
    ead       : Exposure at Default in £. If None, uses 1.0 per borrower.

    Returns
    -------
    DataFrame with PD, LGD, EAD, EL columns
    """
    if ead is None:
        ead = np.ones(len(pd_scores))

    el = pd_scores * lgd * ead

    return pd.DataFrame({
        "PD": pd_scores,
        "LGD": lgd,
        "EAD": ead,
        "EL": el,
        "EL_pct": el / ead  # EL as % of exposure
    })


def stress_test_pd(
    base_pd: np.ndarray,
    scenarios: Dict[str, float],
) -> pd.DataFrame:
    """
    Shift PDs under macro stress scenarios.

    Simple linear stress: stressed_PD = base_PD × multiplier
    In practice, use a macro-conditional PD model (e.g. Merton/Vasicek).

    Parameters
    ----------
    base_pd   : Baseline model PD array
    scenarios : Dict of scenario_name -> PD multiplier
                e.g. {"Baseline": 1.0, "Adverse": 1.5, "Severely Adverse": 2.5}

    Returns
    -------
    DataFrame with mean PD per scenario
    """
    results = []
    for scenario_name, multiplier in scenarios.items():
        stressed_pd = np.clip(base_pd * multiplier, 0, 1)
        results.append({
            "Scenario": scenario_name,
            "Mean PD": stressed_pd.mean(),
            "Median PD": np.median(stressed_pd),
            "P95 PD": np.percentile(stressed_pd, 95),
            "PD Multiplier": multiplier,
        })
    return pd.DataFrame(results)


def vasicek_stress_pd(
    pd_base: float,
    rho: float = 0.15,       # Asset correlation (Basel II IRBA retail = 0.15)
    confidence: float = 0.999  # 99.9% VaR — Basel II requirement
) -> float:
    """
    Vasicek (2002) single-factor conditional PD model.
    Used in Basel II IRBA capital requirement formula.

        PD_stressed = N( (N^-1(PD) - sqrt(rho) × N^-1(confidence)) / sqrt(1 - rho) )

    This converts through-the-cycle PD to point-in-time stressed PD.
    """
    from scipy.stats import norm
    N = norm.cdf
    N_inv = norm.ppf

    pd_stressed = N(
        (N_inv(pd_base) - np.sqrt(rho) * N_inv(confidence)) / np.sqrt(1 - rho)
    )
    return pd_stressed


def plot_el_distribution(el_df: pd.DataFrame, title: str = "Expected Loss Distribution"):
    """Plot histogram of Expected Loss across portfolio."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(el_df["PD"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("PD Distribution")
    axes[0].set_xlabel("Probability of Default")
    axes[0].set_ylabel("Count")

    axes[1].hist(el_df["EL"], bins=50, color="coral", edgecolor="white")
    axes[1].set_title("Expected Loss Distribution")
    axes[1].set_xlabel("Expected Loss (£)")
    axes[1].set_ylabel("Count")

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_stress_scenarios(stress_df: pd.DataFrame):
    """Bar chart comparing mean PD across stress scenarios."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#2ecc71", "#f39c12", "#e74c3c"][:len(stress_df)]
    ax.bar(stress_df["Scenario"], stress_df["Mean PD"], color=colours, edgecolor="white")
    ax.set_ylabel("Mean Portfolio PD")
    ax.set_title("PD Under Macro Stress Scenarios")
    for i, row in stress_df.iterrows():
        ax.text(i, row["Mean PD"] + 0.001, f"{row['Mean PD']:.3f}", ha="center")
    plt.tight_layout()
    return fig
