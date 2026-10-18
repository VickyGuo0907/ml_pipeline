"""Descriptive weather exploration and paired regression error comparisons.

These analyses do not select features or tune models using test outcomes.
Observed horizon weather represents a perfect-weather-information experiment.
"""
from typing import Any

import numpy as np
import pandas as pd


def weather_training_summary(
    train_df: pd.DataFrame,
    target_col: str,
    temperature_col: str = "wx_temp_f",
) -> pd.DataFrame:
    """Summarize all training hours in fixed 10 F bins, separately by season.

    Bins are fixed before inspecting outcomes. They describe association, not
    a causal weather effect; season, hour, and activity also influence load.
    Pass only the training partition, never the concatenated train/test frame.
    """
    frame = train_df[[target_col, temperature_col]].copy()
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Training load and temperature must be nonempty and finite")
    frame["season"] = frame.index.month.map(
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
    edges = [-np.inf, *range(0, 111, 10), np.inf]
    frame["temperature_bin_f"] = pd.cut(frame[temperature_col], bins=edges, right=False)
    summary = (
        frame.groupby(["season", "temperature_bin_f"], observed=True)
        .agg(
            n_hours=(target_col, "size"),
            temperature_mean_f=(temperature_col, "mean"),
            load_mean_mw=(target_col, "mean"),
            load_median_mw=(target_col, "median"),
        )
        .reset_index()
    )
    summary["temperature_bin_f"] = summary.temperature_bin_f.astype(str)
    return summary


def compare_weather_predictions(
    weather: pd.DataFrame,
    no_weather: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pair predictions by cutoff AND forecast timestamp before comparing errors.

    Reject mismatched horizons, targets, or group labels rather than silently
    dropping rows in a join. Positive MAE gains and MAPE percentage-point gains
    favor weather. A day wins only if its mean absolute error is strictly lower.
    """
    required = ["actual", "predicted", "origin", "hour", "season", "holiday"]

    def keyed(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[required].copy()
        if result.empty or not np.isfinite(result[["actual", "predicted"]].to_numpy()).all():
            raise ValueError("Predictions must be nonempty and finite")
        if (result.actual <= 0).any():
            raise ValueError("Percentage comparison requires positive load")
        result["timestamp"] = result.index
        result = result.set_index(["origin", "timestamp"]).sort_index()
        if not result.index.is_unique:
            raise ValueError("Duplicate cutoff/timestamp prediction keys")
        return result

    wx, base = keyed(weather), keyed(no_weather)
    if not wx.index.equals(base.index):
        raise ValueError("Weather comparison requires identical cutoffs and forecast timestamps")
    if not wx[["actual", "hour", "season", "holiday"]].equals(
        base[["actual", "hour", "season", "holiday"]]
    ):
        raise ValueError("Weather comparison requires identical targets and group labels")
    paired = wx.drop(columns="predicted").copy()
    paired["weather_abs_error_mw"] = (wx.actual - wx.predicted).abs()
    paired["no_weather_abs_error_mw"] = (base.actual - base.predicted).abs()
    paired["mae_gain_mw"] = paired.no_weather_abs_error_mw - paired.weather_abs_error_mw
    paired["mape_gain_pp"] = paired.mae_gain_mw / paired.actual * 100
    cutoff = paired.index.get_level_values("origin")
    timestamp = paired.index.get_level_values("timestamp")
    elapsed_hours = (timestamp - cutoff) / pd.Timedelta(hours=1)
    if (elapsed_hours < 0).any() or not np.equal(elapsed_hours, np.floor(elapsed_hours)).all():
        raise ValueError("Forecast timestamps must be hourly and at or after their cutoffs")
    paired["lead_hour"] = elapsed_hours.astype(int) + 1

    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        wx_mae = float(group.weather_abs_error_mw.mean())
        base_mae = float(group.no_weather_abs_error_mw.mean())
        return {
            "n_hours": len(group),
            "n_days": group.index.get_level_values("origin").nunique(),
            "weather_mae_mw": wx_mae,
            "no_weather_mae_mw": base_mae,
            "mae_gain_mw": base_mae - wx_mae,
            "mae_reduction_pct": (base_mae - wx_mae) / base_mae * 100 if base_mae else None,
            "mape_gain_pp": float(group.mape_gain_pp.mean()),
        }

    daily = paired.groupby(level="origin").agg(
        weather_mae_mw=("weather_abs_error_mw", "mean"),
        no_weather_mae_mw=("no_weather_abs_error_mw", "mean"),
        mae_gain_mw=("mae_gain_mw", "mean"),
        mape_gain_pp=("mape_gain_pp", "mean"),
    )
    summary = {
        "weather_information": "perfect-weather-information experiment",
        "overall": summarize(paired),
        "days_improved": int((daily.mae_gain_mw > 0).sum()),
        "days_worse": int((daily.mae_gain_mw < 0).sum()),
        "days_tied": int((daily.mae_gain_mw == 0).sum()),
        "errors_by": {
            field: {str(label): summarize(group) for label, group in paired.groupby(field)}
            for field in ("hour", "season", "holiday", "lead_hour")
        },
        "interpretation": (
            "Descriptive paired comparison of fixed models on the sampled cutoffs. "
            "Historical weather is not an operational weather forecast. Holiday results "
            "may cover very few independent days. No causal claim or test-based tuning."
        ),
    }
    return summary, daily
