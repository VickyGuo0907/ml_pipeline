"""Shared finance evaluation utilities: cross-sectional rank correlation
(Spearman Information Coefficient) and the per-model-type MLflow reload
contract Plan 3 locked in (see src/finance/train_finance.py's module
docstring for the full reasoning).

Used in two places: this plan's evaluate_finance.py (reload each trained
model, score the test period, compute IC per model type) and a later
serving plan (reload the champion model to answer live prediction
requests) - reload_model() is written once here so both reuse it unchanged.
"""
from typing import Any

import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.statsmodels
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_ic(
    long_df: pd.DataFrame,
    predicted_col: str = "predicted",
    actual_col: str = "actual",
    month_col: str = "month",
) -> float:
    """Cross-sectional Spearman rank correlation per month, averaged across months.

    Directly operationalizes the brief's rank-driven long/short industry
    insight: hedge funds care more about which asset will do better (rank)
    than the absolute predicted return. A month where every prediction is
    identical (e.g. Random Walk's constant-0 forecast, or too few assets to
    rank) has an undefined correlation - scipy returns NaN for these
    (ConstantInputWarning, not an exception), and such months are excluded
    from the average rather than counted as 0.

    Args:
        long_df: Columns [month_col, predicted_col, actual_col], one row per
            (month, asset) pair.
        predicted_col: Column name for predicted values.
        actual_col: Column name for actual values.
        month_col: Column name identifying each cross-sectional group.

    Returns:
        Mean Spearman correlation across months with a defined correlation,
        or NaN if no month has one.
    """
    month_ics = []
    for _, group in long_df.groupby(month_col):
        result = spearmanr(group[predicted_col], group[actual_col])
        if not np.isnan(result.statistic):
            month_ics.append(result.statistic)
    if not month_ics:
        return float("nan")
    return float(np.mean(month_ics))


def reload_model(model_type: str, mlflow_run_id: str, mlflow_tracking_uri: str) -> Any:
    """Reload a trained finance model from MLflow using Plan 3's locked
    per-model-type flavor contract.

    Args:
        model_type: One of random_walk, mean, arima, cross_sectional_gbm.
        mlflow_run_id: The MLflow run ID the model was logged under.
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        random_walk/mean -> a PyFuncModel (.predict(model_input) works;
            .forecast() does NOT survive the mlflow round-trip).
        arima -> the native ARIMAResultsWrapper (.forecast(steps) works).
        cross_sectional_gbm -> the native LGBMRegressor (.predict(X) works).

    Raises:
        ValueError: If model_type is not one of the four finance model types.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = f"runs:/{mlflow_run_id}/model"
    if model_type in ("random_walk", "mean"):
        return mlflow.pyfunc.load_model(model_uri)
    if model_type == "arima":
        return mlflow.statsmodels.load_model(model_uri)
    if model_type == "cross_sectional_gbm":
        return mlflow.lightgbm.load_model(model_uri)
    raise ValueError(f"Unknown finance model type '{model_type}' for reload")
