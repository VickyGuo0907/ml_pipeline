"""Hourly weather support for forecasting pipelines: raw-file routing,
station aggregation (clean stage), and weather-derived features (feature
stage).

Regional electric load is driven mostly by HVAC: air conditioning on hot,
humid days and heating on cold, windy ones. Load against temperature is
U-shaped (lowest around a comfort temperature, rising in both directions),
so the features built here linearise that shape (cooling/heating degree
hours) and add the two comfort indices the NWS publishes for exactly these
conditions (heat index, wind chill), plus trailing temperature means for
thermal inertia (buildings keep absorbing heat/cold over multi-day spells).

Every weather column carries the WEATHER_PREFIX ("wx_"). That prefix is the
contract the rest of the forecasting package relies on: recursive_forecast
reads wx_ columns from a supplied exogenous frame instead of rebuilding them
from the target, train_forecast can drop them for a no-weather ablation
model, and serving refuses to forecast with a wx_ model until a weather
forecast feed exists.

CAVEAT: training and scoring use observed (historical) weather for future
hours: a perfect-weather-information experiment. This does not establish
operational accuracy with a genuine day-ahead weather forecast, nor a strict
mathematical bound on production accuracy. A forecast feed needs its own
evaluation; the no-weather serving fallback remains available.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEATHER_PREFIX = "wx_"

# Raw per-station measurement columns the landing CSV must carry
# (written by scripts/stage_pjm_load_forecast_weather_landing.py), and the
# wx_ column each one becomes after station aggregation.
RAW_WEATHER_COLUMNS: dict[str, str] = {
    "temp_f": "wx_temp_f",
    "dewpoint_f": "wx_dewpoint_f",
    "rel_humidity": "wx_rel_humidity",
    "wind_mph": "wx_wind_mph",
}


def is_weather_column(column: str) -> bool:
    """True for any weather-derived feature column (the wx_ prefix contract)."""
    return column.startswith(WEATHER_PREFIX)


def weather_columns(columns: list[str] | pd.Index) -> list[str]:
    """The subset of columns that are weather features, order preserved."""
    return [c for c in columns if is_weather_column(c)]


def split_raw_files(files: list[str], weather_file_pattern: str | None) -> tuple[str, str | None]:
    """Route a forecasting pipeline's raw files into the target file and an
    optional weather file.

    Without weather configured, exactly one raw file is expected (the
    original single-series contract). With weather configured, exactly one
    file must match weather_file_pattern and exactly one other file is the
    target series.

    Args:
        files: Raw filenames from the ingest manifest.
        weather_file_pattern: Substring identifying the weather file, or None
            when weather isn't configured.

    Returns:
        (target_filename, weather_filename or None).

    Raises:
        ValueError: If the file set doesn't match the expected shape.
    """
    if weather_file_pattern is None:
        if len(files) != 1:
            raise ValueError(
                f"Expected exactly one raw file for a forecasting pipeline, found {len(files)}: {files}"
            )
        return files[0], None

    weather = [f for f in files if weather_file_pattern in f]
    target = [f for f in files if weather_file_pattern not in f]
    if len(weather) != 1 or len(target) != 1:
        raise ValueError(
            f"Expected one target file plus one weather file matching '{weather_file_pattern}', "
            f"found weather={weather}, target={target}. Stage the weather landing file with "
            f"scripts/stage_pjm_load_forecast_weather_landing.py, or remove the weather block "
            f"from cleaning.yaml to run without weather."
        )
    return target[0], weather[0]


def aggregate_station_weather(raw: pd.DataFrame, station_weights: dict[str, float]) -> pd.DataFrame:
    """Collapse long per-station hourly rows into one weighted regional series.

    Duplicate (Datetime, station) rows are averaged first: like the PJM load
    file, weather converted to local prevailing time repeats the 1am hour on
    DST fall-back dates. Weights are renormalised per hour over the stations
    that actually reported, so one station's missing hour shifts the average
    toward the others instead of turning the whole hour into NaN.

    Args:
        raw: Long-format frame with Datetime, station, and RAW_WEATHER_COLUMNS.
        station_weights: {station_name: weight}; stations absent from this map
            are ignored (logged once).

    Returns:
        Frame indexed by Datetime with one wx_ column per raw measurement.

    Raises:
        ValueError: If required columns are missing or no configured station
            appears in the data.
    """
    missing = [c for c in ["Datetime", "station", *RAW_WEATHER_COLUMNS] if c not in raw.columns]
    if missing:
        raise ValueError(f"Weather file is missing required columns: {missing}")

    df = raw.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    unknown = sorted(set(df["station"]) - set(station_weights))
    if unknown:
        logger.warning("Ignoring weather stations not configured in cleaning.yaml: %s", unknown)
    df = df[df["station"].isin(station_weights)]
    if df.empty:
        raise ValueError(
            f"No rows for any configured weather station {sorted(station_weights)} in the weather file."
        )

    measures = list(RAW_WEATHER_COLUMNS)
    df = df.groupby(["Datetime", "station"], as_index=False)[measures].mean()
    df["_w"] = df["station"].map(station_weights).astype(float)

    out = {}
    for col in measures:
        valid = df[col].notna()
        weighted = (df[col] * df["_w"]).where(valid)
        num = weighted.groupby(df["Datetime"]).sum(min_count=1)
        den = df["_w"].where(valid).groupby(df["Datetime"]).sum(min_count=1)
        out[RAW_WEATHER_COLUMNS[col]] = num / den
    result = pd.DataFrame(out)
    result.index.name = "Datetime"
    return result.sort_index()


def heat_index_f(temp_f: pd.Series, rel_humidity: pd.Series) -> pd.Series:
    """NWS heat index (degrees F) via the Rothfusz regression with the NWS
    low/high-humidity adjustments; falls back to the NWS simple formula when
    that already averages below 80F (the regression isn't valid there).

    Reference: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    """
    t = temp_f.astype(float)
    rh = rel_humidity.astype(float)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
        - 0.00683783 * t**2 - 0.05481717 * rh**2 + 0.00122874 * t**2 * rh
        + 0.00085282 * t * rh**2 - 0.00000199 * t**2 * rh**2
    )
    low_rh = (rh < 13) & (t >= 80) & (t <= 112)
    hi = hi - np.where(low_rh, ((13 - rh) / 4) * np.sqrt(np.clip((17 - (t - 95).abs()) / 17, 0, None)), 0.0)
    high_rh = (rh > 85) & (t >= 80) & (t <= 87)
    hi = hi + np.where(high_rh, ((rh - 85) / 10) * ((87 - t) / 5), 0.0)
    use_regression = ((simple + t) / 2) >= 80
    return pd.Series(np.where(use_regression, hi, simple), index=temp_f.index)


def wind_chill_f(temp_f: pd.Series, wind_mph: pd.Series) -> pd.Series:
    """NWS (2001) wind chill (degrees F). Defined only for T <= 50F and wind
    > 3 mph; outside that range the air temperature itself is returned.

    Reference: https://www.weather.gov/media/epz/wxcalc/windChill.pdf
    """
    t = temp_f.astype(float)
    v = wind_mph.astype(float)
    v16 = np.power(np.clip(v, 0, None), 0.16)
    wc = 35.74 + 0.6215 * t - 35.75 * v16 + 0.4275 * t * v16
    return pd.Series(np.where((t <= 50) & (v > 3), wc, t), index=temp_f.index)


def build_weather_features(weather: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Derive the wx_ feature set from aggregated hourly weather.

    Only reads weather columns, never the target, so every feature here can
    be computed for future hours from a weather forecast. Trailing rolling
    temperature means include the current hour (the forecast for that hour
    is known at prediction time).

    Args:
        weather: Frame with wx_temp_f, wx_dewpoint_f, wx_rel_humidity,
            wx_wind_mph on an hourly DatetimeIndex.
        config: WeatherFeaturesConfig (comfort_temp_f, rolling_windows,
            heat_index, wind_chill).

    Returns:
        A new frame: the raw wx_ columns plus derived ones.
    """
    out = weather.copy()
    t = out["wx_temp_f"]
    out["wx_cdh"] = (t - config.comfort_temp_f).clip(lower=0)
    out["wx_hdh"] = (config.comfort_temp_f - t).clip(lower=0)
    if config.heat_index:
        out["wx_heat_index_f"] = heat_index_f(t, out["wx_rel_humidity"])
    if config.wind_chill:
        out["wx_wind_chill_f"] = wind_chill_f(t, out["wx_wind_mph"])
    for window in config.rolling_windows:
        out[f"wx_temp_mean_{window}h"] = t.rolling(window).mean()
    return out

