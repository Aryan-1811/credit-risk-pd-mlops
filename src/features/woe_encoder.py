"""
Weight of Evidence (WoE) Encoding and Information Value (IV) Feature Selection.

WoE is the industry-standard transformation for credit risk scorecards.
It linearises the relationship between features and log-odds of default.

    WoE_i = ln(Distribution of Events_i / Distribution of Non-Events_i)
    IV = Σ (Events% - NonEvents%) × WoE
"""

import pandas as pd
import numpy as np
from typing import Dict, List


IV_THRESHOLDS = {
    "Useless":      (0.00, 0.02),
    "Weak":         (0.02, 0.10),
    "Medium":       (0.10, 0.30),
    "Strong":       (0.30, 0.50),
    "Suspiciously strong": (0.50, float("inf")),
}


class WoEEncoder:
    """
    Fits WoE bins per feature and transforms data.
    Supports both numeric (quantile-binned) and categorical features.
    """

    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.woe_maps: Dict[str, pd.DataFrame] = {}
        self.iv_values: Dict[str, float] = {}
        self.bin_edges: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Compute WoE and IV for each feature."""
        for col in X.columns:
            woe_df, edges = self._compute_woe(X[col], y)
            self.woe_maps[col] = woe_df
            self.iv_values[col] = woe_df["IV"].sum()
            if edges is not None:
                self.bin_edges[col] = edges
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace feature values with their WoE scores."""
        X_woe = X.copy()
        for col in X.columns:
            if col in self.woe_maps:
                X_woe[col] = self._apply_woe(X[col], col)
        return X_woe

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _compute_woe(self, series: pd.Series, y: pd.Series):
        """Compute WoE table for a single feature. Returns (woe_df, bin_edges)."""
        total_events = y.sum()
        total_non_events = (1 - y).sum()
        edges = None

        if pd.api.types.is_numeric_dtype(series):
            binned, retbins = pd.qcut(series, q=self.n_bins, duplicates="drop", retbins=True)
            edges = retbins.copy()
            edges[0] = -np.inf
            edges[-1] = np.inf
            bins_str = binned.astype(str)
        else:
            bins_str = series.astype(str)

        df = pd.DataFrame({"bin": bins_str, "target": y})
        grouped = df.groupby("bin")["target"].agg(
            events="sum", total="count"
        ).reset_index()
        grouped["non_events"] = grouped["total"] - grouped["events"]

        grouped["events"] = grouped["events"].clip(lower=0.5)
        grouped["non_events"] = grouped["non_events"].clip(lower=0.5)

        grouped["dist_events"] = grouped["events"] / total_events
        grouped["dist_non_events"] = grouped["non_events"] / total_non_events
        grouped["WoE"] = np.log(
            grouped["dist_events"] / grouped["dist_non_events"]
        )
        grouped["IV"] = (
            grouped["dist_events"] - grouped["dist_non_events"]
        ) * grouped["WoE"]

        return grouped, edges

    def _apply_woe(self, series: pd.Series, col: str) -> pd.Series:
        """Map raw values to WoE scores using stored bin edges."""
        woe_df = self.woe_maps[col]

        if pd.api.types.is_numeric_dtype(series) and col in self.bin_edges:
            edges = self.bin_edges[col]
            binned = pd.cut(series, bins=edges, include_lowest=True)
            bin_str = binned.astype(str)
            mapping = dict(zip(woe_df["bin"], woe_df["WoE"]))
            return bin_str.map(mapping).fillna(0).astype(float)
        else:
            mapping = dict(zip(woe_df["bin"].astype(str), woe_df["WoE"]))
            return series.astype(str).map(mapping).fillna(0).astype(float)

    def get_iv_summary(self) -> pd.DataFrame:
        """Return IV values with predictive power labels."""
        rows = []
        for feature, iv in sorted(
            self.iv_values.items(), key=lambda x: x[1], reverse=True
        ):
            label = next(
                (k for k, (lo, hi) in IV_THRESHOLDS.items() if lo <= iv < hi),
                "Unknown"
            )
            rows.append({"Feature": feature, "IV": round(iv, 4), "Power": label})
        return pd.DataFrame(rows)

    def select_features(self, min_iv: float = 0.02) -> List[str]:
        """Return features with IV above threshold (drop useless ones)."""
        return [f for f, iv in self.iv_values.items() if iv >= min_iv]
