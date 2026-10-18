"""Chronological, fixed daily cutoff forecasting and shared error metrics."""
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.recursive import recursive_forecast
from src.forecasting.weather import weather_columns


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error, as a percentage (0-100+ scale).

    Safe for PJM load values (always well above zero); no zero-guard needed
    for this pipeline's target column.
    """
    actual_arr = actual.to_numpy(dtype=float)
    predicted_arr = predicted.to_numpy(dtype=float)
    return float(np.mean(np.abs((actual_arr - predicted_arr) / actual_arr)) * 100)


def select_cv_origins(
    index: pd.DatetimeIndex,
    n_windows: int,
    horizon_hours: int,
    min_history_hours: int = 1,
) -> list[pd.Timestamp]:
    """Select up to n_windows midnight cutoffs with complete horizons."""
    if n_windows < 1 or horizon_hours < 1 or len(index) < horizon_hours:
        return []
    usable = index[max(min_history_hours, 0) : len(index) - horizon_hours + 1]
    usable = usable[(usable.hour == 0) & (usable.minute == 0)]
    if len(usable) == 0:
        return []
    positions = np.linspace(0, len(usable) - 1, min(n_windows, len(usable))).astype(int)
    return list(usable[positions])


def score_statsmodels_origins(
    fitted: Any,
    y: pd.Series,
    origins: list[pd.Timestamp],
    horizon_hours: int,
    dynamic: bool = True,
    exog: pd.Series | pd.DataFrame | None = None,
) -> list[float]:
    """Score refreshed pre-cutoff states; dynamic is retained for API compatibility."""
    predictions = predict_statsmodels_origins(fitted, y, origins, horizon_hours, exog)
    return [mape(g.actual, g.predicted) for _, g in predictions.groupby("origin", sort=False)]


def predict_statsmodels_origins(
    fitted: Any,
    y: pd.Series,
    origins: list[pd.Timestamp],
    horizon_hours: int,
    exog: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Refresh state on pre-cutoff history with fixed training-only parameters.

    Reject origins inside the fitted sample: dynamic prediction cannot undo
    parameter estimation on future observations. ETS is smoothed on expanding
    history; SARIMAX applies its fitted parameters without optimization.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    frames = []
    fit_end = pd.Timestamp(fitted.model.data.dates[-1])
    fit_start = pd.Timestamp(fitted.model.data.dates[0])
    for origin in origins:
        if fit_end >= origin:
            raise ValueError("Model must be fitted strictly before every validation cutoff")
        history = y.loc[fit_start : origin - pd.Timedelta(hours=1)]
        actual = y.reindex(pd.date_range(origin, periods=horizon_hours, freq="h"))
        if actual.isna().any():
            raise ValueError("Incomplete evaluation horizon")
        if isinstance(fitted.model, ETSModel):
            source = fitted.model
            refreshed = ETSModel(
                history,
                error=source.error,
                trend=source.trend,
                damped_trend=source.damped_trend,
                seasonal=source.seasonal,
                seasonal_periods=source.seasonal_periods,
                initialization_method="estimated",
            ).smooth(fitted.params)
            values = refreshed.forecast(horizon_hours)
        else:
            past_exog = exog.loc[history.index] if exog is not None else None
            refreshed = fitted.apply(history, exog=past_exog, refit=False)
            future_exog = exog.loc[actual.index] if exog is not None else None
            values = refreshed.forecast(horizon_hours, exog=future_exog)
        frames.append(
            pd.DataFrame(
                {"actual": actual.to_numpy(), "predicted": np.asarray(values), "origin": origin},
                index=actual.index,
            )
        )
    return pd.concat(frames) if frames else pd.DataFrame(columns=["actual", "predicted", "origin"])


def score_gbm_origins(
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    origins: list[pd.Timestamp],
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
) -> list[float]:
    """Score a fitted GBM model at each origin via recursive_forecast.

    Args:
        model: Fitted sklearn-compatible estimator.
        df: Feature matrix (DatetimeIndex) covering both the fitted range and
            every origin/horizon being scored, including target_col — the
            training feature matrix for chronological validation; train+test
            concatenated for out-of-sample test evaluation.
        target_col: Target column name.
        feature_columns: Exact column order the model expects.
        origins: Rolling-origin timestamps from select_cv_origins.
        horizon_hours: Forecast horizon per origin.
        lags, rolling_windows, calendar_features, holiday_features: Feature
            config, passed through to recursive_forecast.

    Returns:
        List of MAPE scores, one per origin that had enough trailing data
        and at least one hour of history before it.
    """
    scores = []
    y = df[target_col]
    # Historical wx_ values give a perfect-weather-information experiment;
    # they are not operationally available forecasts.
    wx_cols = weather_columns(feature_columns)
    exog = df[wx_cols] if wx_cols else None
    for origin in origins:
        end = origin + pd.Timedelta(hours=horizon_hours - 1)
        if end > y.index[-1]:
            continue
        history = y.loc[: origin - pd.Timedelta(hours=1)]
        if history.empty:
            continue
        pred = recursive_forecast(
            model,
            history,
            horizon_hours,
            feature_columns,
            lags,
            rolling_windows,
            calendar_features,
            holiday_features,
            exog=exog,
        )
        actual = y.loc[origin:end]
        scores.append(mape(actual, pred))
    return scores


def score_seasonal_naive_origins(
    y: pd.Series,
    origins: list[pd.Timestamp],
    horizon_hours: int,
    season_length: int,
) -> list[float]:
    """Score a seasonal naive forecast at each origin: every forecast hour
    repeats the value observed season_length hours earlier (168 = same hour
    last week). The benchmark any trained model has to beat.

    For horizon_hours > season_length the pattern repeats from the last
    observed season (never reads an actual value at or after the origin).

    Args:
        y: Full target series covering the history before each origin and the
            scored horizon after it.
        origins: Rolling-origin timestamps from select_cv_origins.
        horizon_hours: Forecast horizon per origin.
        season_length: Seasonal period in hours.

    Returns:
        List of MAPE scores, one per origin with a full season of history and
        a full horizon of actuals.
    """
    scores = []
    for origin in origins:
        end = origin + pd.Timedelta(hours=horizon_hours - 1)
        if end > y.index[-1]:
            continue
        last_season = y.loc[
            origin - pd.Timedelta(hours=season_length) : origin - pd.Timedelta(hours=1)
        ]
        if len(last_season) < season_length:
            continue
        reps = int(np.ceil(horizon_hours / season_length))
        pred_values = np.tile(last_season.to_numpy(dtype=float), reps)[:horizon_hours]
        actual = y.loc[origin:end]
        scores.append(mape(actual, pd.Series(pred_values, index=actual.index)))
    return scores


def forecast_metrics(predictions: pd.DataFrame, scale: float) -> dict[str, float | None]:
    """Pooled hourly errors; MASE uses training-only daily seasonal differences."""
    error = predictions.actual - predictions.predicted
    mae = float(error.abs().mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt((error**2).mean())),
        "mape": mape(predictions.actual, predictions.predicted),
        "mase": mae / scale if np.isfinite(scale) and scale > 0 else None,
    }


def evaluate_origins(
    model: Any,
    model_type: str,
    df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    origins: list[pd.Timestamp],
    horizon_hours: int,
    features_config: Any,
    scale_history: pd.Series,
    season_length: int = 168,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """One evaluation implementation shared by pipeline and notebook.

    Historical wx_ inputs are a perfect-weather-information experiment.
    MASE denominator is fixed from the fitting partition, with season m=24.
    """
    if not origins:
        raise ValueError("No scorable daily cutoffs")
    if scale_history.index[-1] >= min(origins):
        raise ValueError("MASE scale history must precede every evaluation cutoff")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Evaluation requires a unique chronological index")
    if season_length < 1:
        raise ValueError("season_length must be positive")
    y = df[target_col]
    if model_type in ("ets", "sarimax"):
        exog = df["is_holiday"] if getattr(model.model, "k_exog", 0) > 0 else None
        predictions = predict_statsmodels_origins(model, y, origins, horizon_hours, exog)
    else:
        frames = []
        direct = model_type == "gbm_direct"
        base_columns = [c for c in feature_columns if c != "lead_hour"]
        wx = weather_columns(base_columns) if model_type in ("gbm", "gbm_direct") else []
        for origin in origins:
            idx = pd.date_range(origin, periods=horizon_hours, freq="h")
            history = y.loc[: origin - pd.Timedelta(hours=1)]
            if model_type == "seasonal_naive":
                if len(history) < season_length:
                    raise ValueError("Insufficient seasonal-naive history")
                values = np.resize(history.tail(season_length).to_numpy(), horizon_hours)
            elif direct:
                from src.forecasting.direct_forecast import direct_forecast

                values = direct_forecast(
                    model,
                    df,
                    target_col,
                    feature_columns,
                    base_columns,
                    origin,
                    horizon_hours,
                    features_config.lags,
                    features_config.rolling_windows,
                    features_config.calendar_features,
                    features_config.holiday_features,
                ).to_numpy()
            else:
                values = recursive_forecast(
                    model,
                    history,
                    horizon_hours,
                    feature_columns,
                    features_config.lags,
                    features_config.rolling_windows,
                    features_config.calendar_features,
                    features_config.holiday_features,
                    exog=df[wx] if wx else None,
                ).to_numpy()
            frames.append(
                pd.DataFrame(
                    {"actual": y.reindex(idx).to_numpy(), "predicted": values, "origin": origin},
                    index=idx,
                )
            )
        if not frames:
            raise ValueError("No scorable daily cutoffs")
        predictions = pd.concat(frames)
    if predictions.empty or predictions[["actual", "predicted"]].isna().any().any():
        raise ValueError("Incomplete evaluation predictions")
    scale = float(scale_history.diff(24).abs().mean())
    metrics: dict[str, Any] = dict(forecast_metrics(predictions, scale))
    scores = [mape(g.actual, g.predicted) for _, g in predictions.groupby("origin")]
    metrics.update(
        test_mape_mean=float(np.mean(scores)),
        test_mape_std=float(np.std(scores)),
        n_windows_scored=len(scores),
        mase_season_length=24,
        mase_scale=scale,
        weather_information=(
            "perfect-weather-information experiment"
            if model_type in ("gbm", "gbm_direct")
            and weather_columns([c for c in feature_columns if c != "lead_hour"])
            else "no weather inputs"
        ),
    )
    predictions["hour"] = predictions.index.hour
    predictions["lead_hour"] = (
        (predictions.index - pd.DatetimeIndex(predictions.origin)) / pd.Timedelta(hours=1) + 1
    ).astype(int)
    predictions["season"] = predictions.index.month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    # Holiday labels follow the same federal calendar as feature engineering.
    from pandas.tseries.holiday import USFederalHolidayCalendar

    holidays = USFederalHolidayCalendar().holidays(
        predictions.index.min().normalize(), predictions.index.max().normalize()
    )
    predictions["holiday"] = predictions.index.normalize().isin(holidays).astype(int)
    metrics["errors_by"] = {
        field: {
            str(label): {**forecast_metrics(group, scale), "n_hours": len(group)}
            for label, group in predictions.groupby(field)
        }
        for field in ("hour", "season", "holiday", "lead_hour")
    }
    return metrics, predictions
