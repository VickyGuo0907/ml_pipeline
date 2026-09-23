"""Model registry for finance returns/risk pipelines: Random Walk, Mean
(shrinkage), ARIMA, and cross-sectional GBM fit functions.

Random Walk and Mean are implemented as mlflow.pyfunc.PythonModel subclasses
(rather than lightweight dataclasses) so train_finance.py (this plan) and a
later serving plan can log/reload them via the standard mlflow.pyfunc
flavor - the pipeline's design spec already names mlflow.pyfunc as the
serving-detection dispatch target for these two model types specifically,
so this interface choice belongs here, not invented ad hoc later.

ARIMA reuses statsmodels' own ARIMAResultsWrapper directly (native
.fittedvalues/.forecast()), mirroring src/forecasting/model_registry.py's
ETS/SARIMAX precedent for pjm_load_forecast. Cross-sectional GBM reuses the
existing tabular src.utils.model_registry "gbm" entry (LGBMRegressor) as-is,
trained via the normal sklearn fit(X, y)/predict(X) interface.
"""
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.utils.model_registry import get_model


class RandomWalkModel(mlflow.pyfunc.PythonModel):
    """Zero-drift random walk: forecast is always 0, the textbook
    EMH-consistent baseline for log-returns.

    Attributes:
        fittedvalues: pd.Series of 0.0, indexed like the training series -
            the "training-period prediction" for every observed return, used
            by train_finance.py's training-error-std-dev calculation.
    """

    def __init__(self, fittedvalues: pd.Series):
        self.fittedvalues = fittedvalues

    def forecast(self, steps: int) -> np.ndarray:
        """Forecast `steps` future values, all 0.0."""
        return np.zeros(steps)

    def predict(self, context, model_input) -> np.ndarray:
        """mlflow.pyfunc contract: output length matches model_input's row count."""
        return np.zeros(len(model_input))


class MeanModel(mlflow.pyfunc.PythonModel):
    """Historical mean forecast - the "shrinkage" estimator the brief means.

    window=None (default) uses the full training history's mean; an int
    window uses only the trailing N observations for both the forecast value
    and (via a shift(1).rolling(window) mean) the training-period fitted
    values.

    Attributes:
        fittedvalues: pd.Series indexed like the training series - constant
            (the full-history mean) if window is None, else a shift(1)
            rolling mean (NaN for the first `window` rows, matching this
            project's other rolling-feature conventions of excluding the
            current row from its own trailing statistic).
    """

    def __init__(self, fittedvalues: pd.Series, forecast_value: float):
        self.fittedvalues = fittedvalues
        self.forecast_value = forecast_value

    def forecast(self, steps: int) -> np.ndarray:
        """Forecast `steps` future values, all equal to forecast_value."""
        return np.full(steps, self.forecast_value)

    def predict(self, context, model_input) -> np.ndarray:
        """mlflow.pyfunc contract: output length matches model_input's row count."""
        return np.full(len(model_input), self.forecast_value)


def fit_random_walk(y: pd.Series) -> RandomWalkModel:
    """Fit a zero-drift random walk model (no fitted parameters).

    Args:
        y: Training target series (log-returns), any index.

    Returns:
        RandomWalkModel whose forecast is always 0.
    """
    return RandomWalkModel(fittedvalues=pd.Series(0.0, index=y.index))


def fit_mean(y: pd.Series, hyperparameters: dict[str, Any]) -> MeanModel:
    """Fit the historical-mean "shrinkage" model.

    Args:
        y: Training target series (log-returns), any index.
        hyperparameters: window (int | None, default None) - trailing
            window in months for the mean; None uses full history.

    Returns:
        MeanModel forecasting the (optionally trailing-window) historical mean.
    """
    window = hyperparameters.get("window")
    if window is None:
        forecast_value = float(y.mean())
        fittedvalues = pd.Series(np.full(len(y), forecast_value), index=y.index)
    else:
        fittedvalues = y.shift(1).rolling(window).mean()
        forecast_value = float(y.tail(window).mean())
    return MeanModel(fittedvalues=fittedvalues, forecast_value=forecast_value)


def fit_arima(y: pd.Series, hyperparameters: dict[str, Any]) -> Any:
    """Fit an ARIMA model on a return series - the "complex model" foil.

    Uses statsmodels.tsa.arima.model.ARIMA (not SARIMAX) - no seasonality
    concept for monthly returns, and no exogenous regressor. order is fixed
    via config, never searched - consistent with the project's ban on
    automated hyperparameter search.

    Args:
        y: Training target series (log-returns), any index.
        hyperparameters: order (list[int], default [1, 0, 0]).

    Returns:
        Fitted ARIMAResultsWrapper (native .fittedvalues/.forecast(steps)).
    """
    order = tuple(hyperparameters.get("order", (1, 0, 0)))
    model = ARIMA(y, order=order)
    return model.fit()


def fit_cross_sectional_gbm(
    panel_df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    hyperparameters: dict[str, Any],
) -> Any:
    """Fit one LightGBM regressor across the whole stacked asset panel.

    Args:
        panel_df: Training feature matrix (all assets stacked), as produced
            by engineer_finance_features - must contain target_col and every
            column in feature_columns.
        target_col: Target column name (e.g. 'log_return').
        feature_columns: Predictor columns (e.g. lag_1_return,
            trailing_12m_vol, ticker_encoded) - already excludes date_ordinal
            (Plan 2's features manifest tracks that separately under
            index_columns, since it's a time axis, not a predictor).
        hyperparameters: LightGBM constructor kwargs (n_estimators,
            max_depth, learning_rate, random_state, n_jobs) from models.yaml.

    Returns:
        Fitted LGBMRegressor (get_model("gbm", ...)).
    """
    model = get_model("gbm", hyperparameters)
    model.fit(panel_df[feature_columns], panel_df[target_col])
    return model
