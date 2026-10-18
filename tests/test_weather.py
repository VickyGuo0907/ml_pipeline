"""Tests for hourly weather support in the forecasting pipeline: raw-file
routing, station aggregation, NWS comfort indices, wx_ feature derivation,
the clean/feature stage wiring, exogenous recursive forecasting, the seasonal
naive benchmark, the with/without-weather GBM ablation through train and
evaluate, serving's weather guard, and the weather landing script."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.stage_pjm_load_forecast_landing import _year_chunks, stage_landing
from src.forecasting.clean_forecast import clean_forecast_data
from src.forecasting.evaluate_forecast import register_forecast_models_to_mlflow
from src.forecasting.features_forecast import engineer_forecast_features
from src.forecasting.recursive import recursive_forecast
from src.forecasting.rolling_origin import score_seasonal_naive_origins
from src.forecasting.serve_forecast import forecast_with_gbm
from src.forecasting.train_forecast import train_forecast_models
from src.forecasting.weather import (
    aggregate_station_weather,
    build_weather_features,
    heat_index_f,
    split_raw_files,
    weather_columns,
    wind_chill_f,
)
from src.utils.config import WeatherFeaturesConfig

TARGET = "PJME_MW"


def _station_rows(idx: pd.DatetimeIndex, station: str, temp: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Datetime": idx, "station": station, "temp_f": temp,
        "dewpoint_f": temp - 10, "rel_humidity": 60.0, "wind_mph": 8.0,
    })


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
class TestSplitRawFiles:
    def test_no_weather_requires_single_file(self):
        assert split_raw_files(["PJME_hourly.csv"], None) == ("PJME_hourly.csv", None)
        with pytest.raises(ValueError, match="exactly one raw file"):
            split_raw_files(["a.csv", "b.csv"], None)

    def test_routes_target_and_weather(self):
        files = ["weather_hourly.csv", "PJME_hourly.csv"]
        assert split_raw_files(files, "weather_hourly") == ("PJME_hourly.csv", "weather_hourly.csv")

    def test_missing_weather_file_fails_loud(self):
        with pytest.raises(ValueError, match="weather file"):
            split_raw_files(["PJME_hourly.csv"], "weather_hourly")


class TestAggregateStationWeather:
    def test_weighted_average_and_renormalised_when_station_missing(self):
        idx = pd.date_range("2020-07-01", periods=3, freq="h")
        a = _station_rows(idx, "A", np.array([90.0, 90.0, 90.0]))
        b = _station_rows(idx[:2], "B", np.array([70.0, 70.0]))  # B missing the last hour
        out = aggregate_station_weather(pd.concat([a, b]), {"A": 3.0, "B": 1.0})
        assert out["wx_temp_f"].iloc[0] == pytest.approx((3 * 90 + 70) / 4)
        assert out["wx_temp_f"].iloc[2] == pytest.approx(90.0)  # only A reported
        assert list(out.columns) == ["wx_temp_f", "wx_dewpoint_f", "wx_rel_humidity", "wx_wind_mph"]

    def test_dst_duplicate_hours_averaged_and_unknown_station_ignored(self):
        idx = pd.DatetimeIndex(["2014-11-02 01:00", "2014-11-02 01:00", "2014-11-02 02:00"])
        a = _station_rows(idx, "A", np.array([50.0, 52.0, 48.0]))
        other = _station_rows(idx, "ZZZ", np.array([0.0, 0.0, 0.0]))
        out = aggregate_station_weather(pd.concat([a, other]), {"A": 1.0})
        assert len(out) == 2
        assert out["wx_temp_f"].iloc[0] == pytest.approx(51.0)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing required columns"):
            aggregate_station_weather(pd.DataFrame({"Datetime": [], "station": []}), {"A": 1.0})


class TestComfortIndices:
    def test_heat_index_matches_nws_table(self):
        # NWS heat index chart: 90F at 70% RH -> 106F; 80F at 40% RH -> 80F.
        hi = heat_index_f(pd.Series([90.0, 80.0]), pd.Series([70.0, 40.0]))
        assert hi.iloc[0] == pytest.approx(106, abs=1)
        assert hi.iloc[1] == pytest.approx(80, abs=1)

    def test_heat_index_below_80_uses_simple_formula(self):
        hi = heat_index_f(pd.Series([60.0]), pd.Series([50.0]))
        assert hi.iloc[0] == pytest.approx(0.5 * (60 + 61 + (60 - 68) * 1.2 + 50 * 0.094))

    def test_wind_chill_matches_nws_table_and_passthrough(self):
        # NWS wind chill chart: 0F at 15 mph -> -19F; 30F at 10 mph -> 21F.
        wc = wind_chill_f(pd.Series([0.0, 30.0, 60.0, 20.0]), pd.Series([15.0, 10.0, 20.0, 2.0]))
        assert wc.iloc[0] == pytest.approx(-19, abs=1)
        assert wc.iloc[1] == pytest.approx(21, abs=1)
        assert wc.iloc[2] == 60.0  # too warm: defined as air temperature
        assert wc.iloc[3] == 20.0  # too calm


class TestBuildWeatherFeatures:
    def test_degree_hours_and_trailing_means(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="h")
        weather = pd.DataFrame({
            "wx_temp_f": [55.0, 65.0, 75.0, 85.0], "wx_dewpoint_f": 50.0,
            "wx_rel_humidity": 50.0, "wx_wind_mph": 5.0,
        }, index=idx)
        cfg = WeatherFeaturesConfig(comfort_temp_f=65, rolling_windows=[2])
        out = build_weather_features(weather, cfg)
        assert out["wx_cdh"].tolist() == [0.0, 0.0, 10.0, 20.0]
        assert out["wx_hdh"].tolist() == [10.0, 0.0, 0.0, 0.0]
        assert out["wx_temp_mean_2h"].iloc[1:].tolist() == [60.0, 70.0, 80.0]
        assert {"wx_heat_index_f", "wx_wind_chill_f"} <= set(out.columns)
        assert weather_columns(out.columns) == list(out.columns)


# --------------------------------------------------------------------------
# Clean + feature stages
# --------------------------------------------------------------------------
def _write_raw(raw_dir: Path, run_id: str, load: pd.DataFrame, weather: pd.DataFrame | None) -> None:
    run_path = raw_dir / run_id
    run_path.mkdir(parents=True)
    files = {"PJME_hourly_test.csv": {"format": "csv"}}
    load.to_csv(run_path / "PJME_hourly_test.csv", index=False)
    if weather is not None:
        weather.to_csv(run_path / "weather_hourly.csv", index=False)
        files["weather_hourly.csv"] = {"format": "csv"}
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump({"run_id": run_id, "files": files}, f)


def _write_weather_config(config_dir: Path, weather_features: bool = True) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        f"target:\n  name: {TARGET}\n  type: continuous\nproblem_type: forecasting\n"
        "train_test_split: 0.8\n"
    )
    (config_dir / "cleaning.yaml").write_text(
        "max_gap_hours: 3\nfill_strategy: interpolate\n"
        "weather:\n  file_pattern: weather_hourly\n  max_gap_hours: 3\n  stations:\n"
        "    - {name: A, latitude: 40.0, longitude: -75.0, weight: 1.0}\n"
        "    - {name: B, latitude: 39.0, longitude: -76.0, weight: 1.0}\n"
    )
    wx = (
        f"weather_features:\n  enabled: {str(weather_features).lower()}\n  comfort_temp_f: 65\n"
        "  rolling_windows: [3]\n"
    )
    (config_dir / "features.yaml").write_text(
        "lags: [1, 2]\nrolling_windows: []\ncalendar_features: true\nholiday_features: false\n"
        "snapshot_hours: 4\n" + wx
    )


def _load_and_weather(n: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2020-07-01", periods=n, freq="h")
    temp = 70 + 15 * np.sin(np.arange(n) * 2 * np.pi / 24)
    load = pd.DataFrame({"Datetime": idx.astype(str), TARGET: 30000 + 300 * temp})
    weather = pd.concat([_station_rows(idx, "A", temp + 2), _station_rows(idx, "B", temp - 2)])
    return load, weather


class TestCleanAndFeaturesWithWeather:
    def test_clean_joins_regional_weather_and_features_derive_wx(self, tmp_path):
        run_id = "2026-10-16"
        load, weather = _load_and_weather()
        # A 2h hole in one station's data: other station covers it (renormalised).
        weather = weather.drop(weather[(weather["station"] == "A")].index[10:12])
        _write_raw(tmp_path / "raw", run_id, load, weather)
        _write_weather_config(tmp_path / "config")

        result = clean_forecast_data(tmp_path / "raw", tmp_path / "interim", run_id, config_dir=tmp_path / "config")
        interim = pd.read_parquet(result["output_path"])
        assert {"wx_temp_f", "wx_rel_humidity"} <= set(interim.columns)
        assert interim["wx_temp_f"].notna().all()
        manifest = yaml.safe_load((tmp_path / "interim" / run_id / "manifest.yaml").read_text())
        assert manifest["weather"]["stations"] == ["A", "B"]

        engineer_forecast_features(tmp_path / "interim", tmp_path / "features", run_id, config_dir=tmp_path / "config")
        train = pd.read_parquet(tmp_path / "features" / run_id / "train.parquet")
        assert {"wx_cdh", "wx_hdh", "wx_heat_index_f", "wx_wind_chill_f", "wx_temp_mean_3h"} <= set(train.columns)
        assert train.notna().all().all()

    def test_long_weather_gap_fails_loud(self, tmp_path):
        run_id = "2026-10-16"
        load, weather = _load_and_weather()
        ts = pd.to_datetime(weather["Datetime"])
        weather = weather[(ts < "2020-07-01 10:00") | (ts > "2020-07-01 18:00")]  # 9h gap, both stations
        _write_raw(tmp_path / "raw", run_id, load, weather)
        _write_weather_config(tmp_path / "config")
        with pytest.raises(ValueError, match="gap"):
            clean_forecast_data(tmp_path / "raw", tmp_path / "interim", run_id, config_dir=tmp_path / "config")

    def test_weather_features_disabled_drops_wx_columns(self, tmp_path):
        run_id = "2026-10-16"
        load, weather = _load_and_weather()
        _write_raw(tmp_path / "raw", run_id, load, weather)
        _write_weather_config(tmp_path / "config", weather_features=False)
        clean_forecast_data(tmp_path / "raw", tmp_path / "interim", run_id, config_dir=tmp_path / "config")
        engineer_forecast_features(tmp_path / "interim", tmp_path / "features", run_id, config_dir=tmp_path / "config")
        train = pd.read_parquet(tmp_path / "features" / run_id / "train.parquet")
        assert not weather_columns(train.columns)


# --------------------------------------------------------------------------
# Recursive forecasting with exogenous weather + seasonal naive
# --------------------------------------------------------------------------
class _SumModel:
    """Predicts lag_1h + wx_temp_f, so the output proves exog was read per step."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["lag_1h"] + X["wx_temp_f"]).to_numpy()


class TestRecursiveWithExog:
    def test_exog_read_per_step(self):
        hist_idx = pd.date_range("2020-01-01", periods=3, freq="h")
        history = pd.Series([10.0, 10.0, 10.0], index=hist_idx)
        future = pd.date_range(hist_idx[-1] + pd.Timedelta(hours=1), periods=3, freq="h")
        exog = pd.DataFrame({"wx_temp_f": [1.0, 2.0, 3.0]}, index=future)
        pred = recursive_forecast(_SumModel(), history, 3, ["lag_1h", "wx_temp_f"], [1], [], False, False, exog=exog)
        assert pred.tolist() == [11.0, 13.0, 16.0]

    def test_exog_missing_step_raises(self):
        hist_idx = pd.date_range("2020-01-01", periods=3, freq="h")
        history = pd.Series([10.0, 10.0, 10.0], index=hist_idx)
        exog = pd.DataFrame({"wx_temp_f": [1.0]}, index=[hist_idx[-1] + pd.Timedelta(hours=1)])
        with pytest.raises(ValueError, match="no row for forecast step"):
            recursive_forecast(_SumModel(), history, 2, ["lag_1h", "wx_temp_f"], [1], [], False, False, exog=exog)


class TestSeasonalNaive:
    def test_perfectly_periodic_series_scores_zero(self):
        idx = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
        y = pd.Series(100 + np.tile(np.arange(24), 10), index=idx, dtype=float)
        scores = score_seasonal_naive_origins(y, [idx[100], idx[150]], horizon_hours=12, season_length=24)
        assert scores == [0.0, 0.0]

    def test_horizon_longer_than_season_tiles_last_season(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        y = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], index=idx, dtype=float)
        assert score_seasonal_naive_origins(y, [idx[4]], horizon_hours=5, season_length=2) == [0.0]

    def test_origin_without_full_season_skipped(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        y = pd.Series(np.arange(10, dtype=float) + 1, index=idx)
        assert score_seasonal_naive_origins(y, [idx[1]], horizon_hours=2, season_length=5) == []


# --------------------------------------------------------------------------
# Train + evaluate: seasonal naive baseline and with/without-weather GBM
# --------------------------------------------------------------------------
def _feature_frame(n: int) -> pd.DataFrame:
    """Load driven mostly by an exogenous temperature series (not by its own
    lags), so a weather-aware GBM has something real to learn."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", name="Datetime")
    temp = 65 + 20 * np.sin(np.arange(n) * 2 * np.pi / 37) + rng.normal(0, 2, n)
    df = pd.DataFrame({TARGET: 20000 + 400 * np.abs(temp - 65) + rng.normal(0, 50, n)}, index=idx)
    df["lag_1h"] = df[TARGET].shift(1)
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["year"] = idx.year
    df["wx_temp_f"] = temp
    df["wx_cdh"] = np.clip(temp - 65, 0, None)
    df["wx_hdh"] = np.clip(65 - temp, 0, None)
    return df.dropna()


def _write_train_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        f"target:\n  name: {TARGET}\n  type: continuous\nproblem_type: forecasting\n"
        "pipeline_type: test_forecast\n"
    )
    (config_dir / "features.yaml").write_text(
        "lags: [1]\nrolling_windows: []\ncalendar_features: true\nholiday_features: false\nsnapshot_hours: 1\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - {name: t_naive, type: seasonal_naive, hyperparameters: {season_length: 24}}\n"
        "  - {name: t_gbm, type: gbm, hyperparameters: {n_estimators: 50, random_state: 1, verbose: -1}}\n"
        "  - {name: t_gbm_nowx, type: gbm, hyperparameters: {use_weather: false, n_estimators: 50, "
        "random_state: 1, verbose: -1}}\n"
        "evaluation:\n  horizon_hours: 6\n  n_windows: 3\n"
    )


class TestTrainEvaluateWeatherAblation:
    def test_baseline_scored_not_registered_and_ablation_columns(self, tmp_path):
        run_id = "2026-10-16"
        df = _feature_frame(600)
        run_dir = tmp_path / "features" / run_id
        run_dir.mkdir(parents=True)
        df.iloc[:480].to_parquet(run_dir / "train.parquet")
        df.iloc[480:].to_parquet(run_dir / "test.parquet")
        config_dir = tmp_path / "config"
        _write_train_config(config_dir)
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("weather_test")

        trained = train_forecast_models(tmp_path / "features", run_id, config_dir=config_dir, mlflow_tracking_uri=mlflow_uri)
        assert set(trained["models"]) == {"t_naive", "t_gbm", "t_gbm_nowx"}

        cols = {
            name: mlflow.artifacts.load_dict(f"runs:/{info['mlflow_run_id']}/feature_columns.json")["columns"]
            for name, info in trained["models"].items()
        }
        assert weather_columns(cols["t_gbm"]) == ["wx_temp_f", "wx_cdh", "wx_hdh"]
        assert weather_columns(cols["t_gbm_nowx"]) == []

        result = register_forecast_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids={n: i["mlflow_run_id"] for n, i in trained["models"].items()},
            config_dir=config_dir, run_id=run_id, reports_dir=tmp_path / "reports",
            features_dir=tmp_path / "features",
        )
        statuses = {n: r["status"] for n, r in result["registered_models"].items()}
        assert statuses == {"t_naive": "baseline", "t_gbm": "registered", "t_gbm_nowx": "registered"}
        report = yaml.safe_load(next((tmp_path / "reports").rglob("*evaluation.yaml")).read_text())
        assert report["run_champion"] in {"t_gbm", "t_gbm_nowx"}
        for name, returned in result["registered_models"].items():
            for metric in ("mae", "rmse", "mape", "mase", "test_mape_mean", "test_mape_std", "cv_mape_mean"):
                assert returned[metric] == report["models"][name][metric]
            assert set(returned["errors_by"]) == {"hour", "season", "holiday", "lead_hour"}
            assert returned["n_windows_scored"] == len(result["protocol"]["cutoffs"])
        assert result["registered_models"]["t_gbm"]["weather_information"] == "perfect-weather-information experiment"
        assert result["registered_models"]["t_gbm_nowx"]["weather_information"] == "no weather inputs"
        # Load here is driven by temperature, so weather should clearly help.
        assert report["models"]["t_gbm"]["test_mape_mean"] < report["models"]["t_gbm_nowx"]["test_mape_mean"]


class TestServeGuard:
    def test_weather_model_without_exog_raises(self):
        snapshot = pd.DataFrame({TARGET: [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2, freq="h"))
        with pytest.raises(ValueError, match="weather"):
            forecast_with_gbm(_SumModel(), snapshot, TARGET, ["lag_1h", "wx_temp_f"], 2, [1], [], False, False)


# --------------------------------------------------------------------------
# Landing script
# --------------------------------------------------------------------------
class TestWeatherLandingScript:
    def test_year_chunks(self):
        assert _year_chunks("2001-12-31", "2003-02-01") == [
            ("2001-12-31", "2001-12-31"), ("2002-01-01", "2002-12-31"), ("2003-01-01", "2003-02-01"),
        ]

    def test_stage_landing_converts_utc_to_eastern_wall_clock(self, tmp_path):
        calls = []

        def fake_fetcher(params):
            calls.append(params)
            times = pd.date_range(f"{params['start_date']} 00:00", f"{params['end_date']} 23:00", freq="h")
            n = len(times)
            return {"hourly": {
                "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                "temperature_2m": [50.0] * n, "dew_point_2m": [40.0] * n,
                "relative_humidity_2m": [70.0] * n, "wind_speed_10m": [5.0] * n,
            }}

        stations = [{"name": "PHL", "latitude": 39.87, "longitude": -75.24}]
        result = stage_landing(stations, "2014-11-02", "2014-11-02", tmp_path, fetcher=fake_fetcher, pause_seconds=0)
        out = pd.read_csv(result["output_path"], parse_dates=["Datetime"])
        assert calls[0]["timezone"] == "GMT" and calls[0]["temperature_unit"] == "fahrenheit"
        assert list(out.columns) == ["Datetime", "station", "temp_f", "dewpoint_f", "rel_humidity", "wind_mph"]
        # 2014-11-02 00:00 UTC is 2014-11-01 20:00 EDT; DST fall-back repeats 01:00 local.
        assert out["Datetime"].min() == pd.Timestamp("2014-11-01 20:00")
        assert (out["Datetime"] == pd.Timestamp("2014-11-02 01:00")).sum() == 2


class TestServeEndpointGuard:
    def test_weather_model_rejected_with_501(self):
        from fastapi import HTTPException

        from src.serve import _require_no_weather_features

        _require_no_weather_features(["lag_1h", "hour"])  # no-weather model passes
        with pytest.raises(HTTPException) as exc:
            _require_no_weather_features(["lag_1h", "wx_temp_f"])
        assert exc.value.status_code == 501
