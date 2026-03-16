"""
Data preprocessing for Credit Risk PD Modelling.
Handles cleaning, imputation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset (Give Me Some Credit or German Credit)."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Remove duplicates
    - Cap extreme outliers
    - Handle impossible values
    """
    df = df.drop_duplicates()

    # Example: cap age at realistic range
    if "age" in df.columns:
        df = df[df["age"].between(18, 100)]

    # Cap revolving utilisation at 1 (Give Me Some Credit specific)
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["RevolvingUtilizationOfUnsecuredLines"] = df[
            "RevolvingUtilizationOfUnsecuredLines"
        ].clip(0, 1)

    print(f"After cleaning: {len(df):,} rows")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numeric: median imputation (robust to outliers)
    - Categorical: mode imputation
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed '{col}' with median={median_val:.4f}")

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified train/val/test split preserving default rate.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_relative,
        stratify=y_train, random_state=random_state
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Default rate — Train: {y_train.mean():.3f} | "
          f"Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test
