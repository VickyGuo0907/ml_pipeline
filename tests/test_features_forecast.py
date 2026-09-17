"""Tests for the forecasting feature engineering stage: lag/rolling/calendar/
holiday features, the chronological split, and the serving-time snapshot."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.forecasting.features_forecast import engineer_forecast_features
from src.schemas.features import build_forecast_features_schema


def _write_interim_fixture(interim_dir: Path, run_id: str, df: pd.DataFrame) -> Path:
    """Write a cleaned DatetimeIndex parquet + manifest.yaml into interim_dir/run_id/,
    mirroring what clean_forecast_data (Task 1) produces — independent of Task 1's
    implementation, so this test file exercises engineer_forecast_features in isolation.
    """
    run_path = interim_dir / run_id
    run_path.mkdir(parents=True)
    output_path = run_path / "cleaned.parquet"
    df.to_parquet(output_path)
    manifest = {
        "run_id": run_id,
        "stage": "clean_forecast",
        "output_path": str(output_path),
    }
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    return run_path


def _write_forecast_config(
    config_dir: Path,
    lags: list[int],
    rolling_windows: list[int],
    train_test_split: float = 0.8,
    snapshot_hours: int = 5,
    calendar_features: bool = True,
    holiday_features: bool = False,
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: PJME_MW\n  type: continuous\nproblem_type: forecasting\n"
        f"train_test_split: {train_test_split}\n"
    )
    (config_dir / "features.yaml").write_text(
        f"lags: {lags}\n"
        f"rolling_windows: {rolling_windows}\n"
        f"calendar_features: {str(calendar_features).lower()}\n"
        f"holiday_features: {str(holiday_features).lower()}\n"
        f"snapshot_hours: {snapshot_hours}\n"
    )


def _synthetic_series(n_hours: int, start: str = "2020-01-01") -> pd.DataFrame:
    """A simple hourly series with known values (0, 1, 2, ...) for exact lag/rolling math."""
    idx = pd.date_range(start, periods=n_hours, freq="h", name="Datetime")
    return pd.DataFrame({"PJME_MW": range(n_hours)}, index=idx)


class TestLagAndRollingFeatures:
    """Exact-value checks on a small known series."""

    def test_lag_columns_match_shifted_values(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(20)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1, 2], rolling_windows=[], snapshot_hours=3)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        full = pd.concat([
            pd.read_parquet(features_dir / run_id / "train.parquet"),
            pd.read_parquet(features_dir / run_id / "test.parquet"),
        ]).sort_index()

        # Row at hour 5 (value 5.0): lag_1h should be hour 4's value (4.0), lag_2h hour 3's (3.0)
        row = full.loc["2020-01-01 05:00:00"]
        assert row["lag_1h"] == pytest.approx(4.0)
        assert row["lag_2h"] == pytest.approx(3.0)

    def test_rolling_mean_excludes_current_row(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(20)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[], rolling_windows=[3], snapshot_hours=3)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        full = pd.concat([
            pd.read_parquet(features_dir / run_id / "train.parquet"),
            pd.read_parquet(features_dir / run_id / "test.parquet"),
        ]).sort_index()

        # Row at hour 10 (value 10.0): rolling_mean_3h over shift(1) = mean(hours 7,8,9) = mean(7,8,9) = 8.0
        row = full.loc["2020-01-01 10:00:00"]
        assert row["rolling_mean_3h"] == pytest.approx(8.0)


class TestCalendarFeatures:
    def test_hour_day_of_week_month_weekend(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        # 2020-01-04 is a Saturday. Needs >=35 hours from 2020-01-03 00:00 to
        # reach 2020-01-04 10:00 (hour 34) — 40 gives headroom.
        df = _synthetic_series(40, start="2020-01-03")
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1], rolling_windows=[], snapshot_hours=3)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        full = pd.concat([
            pd.read_parquet(features_dir / run_id / "train.parquet"),
            pd.read_parquet(features_dir / run_id / "test.parquet"),
        ]).sort_index()

        row = full.loc["2020-01-04 10:00:00"]
        assert row["hour"] == 10
        assert row["day_of_week"] == 5  # Saturday
        assert row["month"] == 1
        assert bool(row["is_weekend"]) is True


class TestHolidayFeatures:
    def test_is_holiday_on_july_fourth(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(48, start="2019-07-03")  # spans July 3-4, 2019
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1], rolling_windows=[], snapshot_hours=3, holiday_features=True)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        full = pd.concat([
            pd.read_parquet(features_dir / run_id / "train.parquet"),
            pd.read_parquet(features_dir / run_id / "test.parquet"),
        ]).sort_index()

        july_4 = full.loc["2019-07-04 12:00:00"]
        july_3 = full.loc["2019-07-03 12:00:00"]
        assert bool(july_4["is_holiday"]) is True
        assert july_4["days_to_nearest_holiday"] == 0
        assert bool(july_3["is_holiday"]) is False
        assert july_3["days_to_nearest_holiday"] == 1


class TestChronologicalSplit:
    def test_train_test_no_overlap_and_correct_order(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(50)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1], rolling_windows=[], train_test_split=0.8, snapshot_hours=3)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        test_df = pd.read_parquet(features_dir / run_id / "test.parquet")

        assert set(train_df.index).isdisjoint(set(test_df.index))
        assert train_df.index.max() < test_df.index.min()
        assert train_df.index.is_monotonic_increasing
        assert test_df.index.is_monotonic_increasing

    def test_datetime_index_present_and_named(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(30)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1], rolling_windows=[], snapshot_hours=3)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        assert isinstance(train_df.index, pd.DatetimeIndex)


class TestSnapshot:
    def test_snapshot_has_exact_row_count_and_matches_tail(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(50)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1], rolling_windows=[], snapshot_hours=5)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        snapshot_df = pd.read_parquet(features_dir / run_id / "last_window.parquet")
        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        test_df = pd.read_parquet(features_dir / run_id / "test.parquet")
        full = pd.concat([train_df, test_df]).sort_index()

        assert len(snapshot_df) == 5
        assert list(snapshot_df.index) == list(full.tail(5).index)


class TestSchemaCompliance:
    def test_train_output_validates_against_forecast_schema(self, tmp_path):
        """Real output, not a hand-built frame, must satisfy Plan 1's build_forecast_features_schema."""
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(30)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1, 2], rolling_windows=[3], snapshot_hours=3)

        engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        schema = build_forecast_features_schema("PJME_MW")
        validated = schema.validate(train_df)
        assert len(validated) == len(train_df)


class TestManifestAndReturn:
    def test_manifest_and_return_shape(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = _synthetic_series(30)
        _write_interim_fixture(interim_dir, run_id, df)
        _write_forecast_config(config_dir, lags=[1, 2], rolling_windows=[3], snapshot_hours=3)

        result = engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        assert result["train_path"].endswith("train.parquet")
        assert result["test_path"].endswith("test.parquet")
        assert result["snapshot_path"].endswith("last_window.parquet")
        assert result["rows_dropped_warmup"] > 0  # lag_2h/rolling_mean_3h need warm-up rows dropped

        with open(features_dir / run_id / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["stage"] == "feature_engineer_forecast"
        assert "PJME_MW" not in manifest["feature_columns"]
