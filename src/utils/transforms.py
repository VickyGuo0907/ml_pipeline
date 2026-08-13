"""Composable feature transform functions.

Each function is stateless and returns a transformed copy.
Enable/disable via features.yaml — no code changes needed.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def frequency_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace each category with its count in the dataset.

    SVG Stage 1: "Freq encode State — 52 levels → numeric"

    Args:
        df: DataFrame containing col.
        col: Column name to encode in-place.

    Returns:
        Copy of df with col replaced by integer frequencies.
    """
    df = df.copy()
    freq_map = df[col].value_counts().to_dict()
    df[col] = df[col].map(freq_map).fillna(0).astype(int)
    return df


def fit_boxcox(y: pd.Series) -> dict[str, float]:
    """Fit Box-Cox parameters (lambda + shift offset) on training data only.

    The offset shifts the series so every value is > 0 before Box-Cox. Fit on
    the training set and reuse the returned params via apply_boxcox on both
    train and test so test data never contributes to the fitted statistics.

    Args:
        y: Training target series (must be numeric).

    Returns:
        Dict with the fitted "lambda" and "offset".
    """
    from scipy.stats import boxcox

    y_arr = y.to_numpy(dtype=float)
    offset = max(0.0, float(-y_arr.min())) + 1e-6
    _, lambda_val = boxcox(y_arr + offset)
    logger.info("Box-Cox fitted: λ=%.4f, offset=%.6f", lambda_val, offset)
    return {"lambda": float(lambda_val), "offset": float(offset)}


def apply_boxcox(y: pd.Series, params: dict[str, float]) -> pd.Series:
    """Apply a Box-Cox transform previously fitted by fit_boxcox.

    Args:
        y: Target series to transform (train or test).
        params: Dict with "lambda" and "offset" as returned by fit_boxcox.

    Returns:
        Transformed series.
    """
    from scipy.stats import boxcox

    transformed = boxcox(y.to_numpy(dtype=float) + params["offset"], lmbda=params["lambda"])
    return pd.Series(transformed, index=y.index, name=y.name)


def boxcox_transform(y: pd.Series) -> tuple[pd.Series, float]:
    """Apply Box-Cox power transform; handles zeros/negatives via offset.

    Convenience single-pass wrapper over fit_boxcox/apply_boxcox for callers
    that only need the fitted lambda.

    Args:
        y: Target series (must be numeric).

    Returns:
        Tuple of (transformed series, fitted lambda).
    """
    params = fit_boxcox(y)
    return apply_boxcox(y, params), params["lambda"]


def compute_vif(X: pd.DataFrame) -> dict[str, float]:
    """Compute Variance Inflation Factor for each predictor column.

    Args:
        X: Feature matrix (numeric only, no target).

    Returns:
        Dict mapping column name → VIF value.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    values = X.values.astype(float)
    return {col: variance_inflation_factor(values, i) for i, col in enumerate(X.columns)}


def drop_high_vif(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Iteratively drop the highest-VIF column until all VIF ≤ threshold.

    SVG Stage 2: "VIF cleanup — Drop 5 HCAHPS > 5"

    Args:
        X: Feature matrix.
        threshold: Maximum acceptable VIF.

    Returns:
        Tuple of (pruned DataFrame, list of dropped column names).
    """
    X = X.copy()
    dropped: list[str] = []

    while len(X.columns) > 1:
        vifs = compute_vif(X)
        max_col = max(vifs, key=lambda c: vifs[c])
        if vifs[max_col] <= threshold:
            break
        logger.info("Dropping '%s' (VIF=%.2f > %.1f)", max_col, vifs[max_col], threshold)
        X = X.drop(columns=[max_col])
        dropped.append(max_col)

    logger.info("VIF cleanup removed %d columns: %s", len(dropped), dropped)
    return X, dropped


def iterative_impute(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Impute missing values using MICE (sklearn IterativeImputer).

    Equivalent to R's missForest on numeric columns.
    Non-numeric columns are left untouched.

    SVG Stage 1: "missForest impute — Fill all NAs"

    Args:
        df: DataFrame potentially containing NaN values.
        random_state: Seed for reproducibility.

    Returns:
        Copy of df with numeric NaN values imputed.
    """
    # Import here because IterativeImputer is still experimental in sklearn 1.3
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    imputer = IterativeImputer(random_state=random_state, max_iter=10)
    df = df.copy()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    logger.info("IterativeImputer filled NAs in %d numeric columns", len(numeric_cols))
    return df


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaN with column median; categorical with mode.

    Args:
        df: DataFrame to impute.

    Returns:
        Copy of df with NaN values filled.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
    return df


def drop_pattern_columns(df: pd.DataFrame, patterns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose names contain any of the given substrings (case-insensitive).

    SVG Stage 1: "Drop bad imputations — Payment, ED volume"

    Args:
        df: Input DataFrame.
        patterns: List of substrings to match against column names.

    Returns:
        Tuple of (filtered DataFrame, list of dropped column names).
    """
    if not patterns:
        return df, []

    lower_patterns = [p.lower() for p in patterns]
    to_drop = [
        col for col in df.columns
        if any(p in col.lower() for p in lower_patterns)
    ]
    logger.info("Dropping %d pattern-matched columns: %s", len(to_drop), to_drop)
    return df.drop(columns=to_drop), to_drop


# Registry for imputation strategies — add new strategies by extending this dict
IMPUTE_REGISTRY: dict[str, Any] = {
    "iterative": iterative_impute,
    "median": median_impute,
}
