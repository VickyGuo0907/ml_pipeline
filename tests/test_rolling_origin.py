"""Chronological cutoff, state refresh, and forecast metric regression tests."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.model_registry import fit_ets, fit_sarimax
from src.forecasting.rolling_origin import (
    evaluate_origins,
    mape,
    predict_statsmodels_origins,
    score_gbm_origins,
    score_statsmodels_origins,
    select_cv_origins,
)


def _synthetic_series(n_hours: int = 250, seasonal_period: int = 24) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = (
        50
        + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period)
        + rng.normal(0, 1, n_hours)
    )
    return pd.Series(values, index=idx)


def _synthetic_train_df(n_hours: int = 200, seasonal_period: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h", name="Datetime")
    target = (
        50
        + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period)
        + rng.normal(0, 1, n_hours)
    )
    target = pd.Series(target, index=idx)
    df = pd.DataFrame({"PJME_MW": target})
    df["lag_1h"] = df["PJME_MW"].shift(1)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df.dropna()


class TestMape:
    def test_zero_error_is_zero_percent(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        assert mape(actual, actual) == pytest.approx(0.0)

    def test_known_percentage_error(self):
        actual = pd.Series([100.0, 200.0])
        predicted = pd.Series([110.0, 180.0])  # +10%, -10%
        assert mape(actual, predicted) == pytest.approx(10.0)


class TestSelectCvOrigins:
    def test_origins_leave_room_for_full_horizon(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        origins = select_cv_origins(idx, n_windows=3, horizon_hours=10)
        assert origins
        assert all(o.hour == 0 for o in origins)
        for origin in origins:
            end = origin + pd.Timedelta(hours=9)
            assert end <= idx[-1]

    def test_min_history_hours_pushes_first_origin_past_naive_margin(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        origins = select_cv_origins(idx, n_windows=5, horizon_hours=10, min_history_hours=5)
        assert origins[0] >= idx[5]

    def test_returns_empty_list_when_series_too_short_for_any_window(self):
        """A series shorter than horizon_hours + min_history_hours can't
        support even one origin — select_cv_origins must return [] rather
        than raise, so callers (train_forecast_models, evaluate_forecast's
        registration step) can detect and report this cleanly instead of
        crashing on a malformed index slice."""
        index = pd.date_range("2020-01-01", periods=10, freq="h")

        origins = select_cv_origins(index, n_windows=3, horizon_hours=20, min_history_hours=1)

        assert origins == []


class TestScoreStatsmodelsOrigins:
    def test_rejects_model_fitted_on_validation_observations(self):
        full = _synthetic_series()
        fitted = fit_ets(full, {"seasonal_periods": 24})
        with pytest.raises(ValueError, match="strictly before"):
            score_statsmodels_origins(fitted, full, [full.index[192]], 24)

    def test_ets_refresh_uses_only_pre_cutoff_observations(self):
        full = _synthetic_series(300)
        fitted = fit_ets(full.iloc[:168], {"seasonal_periods": 24})
        origin = full.index[240]
        original = predict_statsmodels_origins(fitted, full, [origin], 24)
        future_changed = full.copy()
        future_changed.loc[origin:] += 500
        changed = predict_statsmodels_origins(fitted, future_changed, [origin], 24)
        np.testing.assert_allclose(original.predicted, changed.predicted)
        recent_changed = full.copy()
        recent_changed.iloc[216:240] += 20
        refreshed = predict_statsmodels_origins(fitted, recent_changed, [origin], 24)
        assert not np.allclose(original.predicted, refreshed.predicted)
        assert len(original) == 24
        assert original.index[0] == origin

    def test_out_of_sample_dynamic_false_works_for_ets(self):
        """The empirically-verified fix: dynamic=origin RAISES for an
        out-of-sample origin with ETSModel ('Cannot anchor simulation outside
        of the sample'). dynamic=False must be used instead, and must work."""
        full = _synthetic_series(250)
        y_train = full.iloc[:200]
        y_test = full.iloc[200:]
        fitted = fit_ets(y_train, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        origins = select_cv_origins(y_test.index, n_windows=2, horizon_hours=5)
        full_y = pd.concat([y_train, y_test])

        scores = score_statsmodels_origins(fitted, full_y, origins, horizon_hours=5, dynamic=False)

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)

    def test_out_of_sample_dynamic_false_works_for_sarimax(self):
        full = _synthetic_series(250)
        y_train = full.iloc[:200]
        y_test = full.iloc[200:]
        fitted = fit_sarimax(y_train, {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 24]})

        origins = select_cv_origins(y_test.index, n_windows=2, horizon_hours=5)
        full_y = pd.concat([y_train, y_test])

        scores = score_statsmodels_origins(fitted, full_y, origins, horizon_hours=5, dynamic=False)

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)


class TestScoreGbmOrigins:
    def test_scores_each_valid_origin(self):
        df = _synthetic_train_df()

        class _MeanModel:
            def predict(self, X):
                return np.full(len(X), df["PJME_MW"].mean())

        origins = select_cv_origins(df.index, n_windows=2, horizon_hours=5)
        scores = score_gbm_origins(
            _MeanModel(),
            df,
            "PJME_MW",
            ["lag_1h", "hour"],
            origins,
            horizon_hours=5,
            lags=[1],
            rolling_windows=[],
            calendar_features=True,
            holiday_features=False,
        )

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)


def test_shared_baselines_metrics_and_mase_training_only():
    from types import SimpleNamespace

    full = _synthetic_series(24 * 20)
    df = full.to_frame("load")
    origins = [full.index[24 * 10], full.index[24 * 11]]
    history = full.iloc[: 24 * 10]
    cfg = SimpleNamespace(
        lags=[1], rolling_windows=[], calendar_features=False, holiday_features=False
    )
    for season in (24, 168):
        metrics, predictions = evaluate_origins(
            None, "seasonal_naive", df, "load", [], origins, 24, cfg, history, season
        )
        error = predictions.actual - predictions.predicted
        assert metrics["mae"] == pytest.approx(error.abs().mean())
        assert metrics["rmse"] == pytest.approx(np.sqrt((error**2).mean()))
        assert metrics["mape"] == pytest.approx(mape(predictions.actual, predictions.predicted))
        assert metrics["mase"] == pytest.approx(error.abs().mean() / history.diff(24).abs().mean())
        assert metrics["n_windows_scored"] == 2
        assert sum(g["n_hours"] for g in metrics["errors_by"]["hour"].values()) == 48
        for origin in origins:
            expected = full.loc[origin - pd.Timedelta(hours=season) :].iloc[:24].to_numpy()
            np.testing.assert_allclose(
                predictions.loc[predictions.origin == origin, "predicted"], expected
            )


def test_validation_fit_boundary_is_before_every_cutoff(tmp_path, monkeypatch):
    import mlflow

    import src.forecasting.train_forecast as training
    from tests.test_train_forecast import _synthetic_train_df, _write_forecast_config

    frame = _synthetic_train_df(400)
    (tmp_path / "test").mkdir()
    frame.to_parquet(tmp_path / "test" / "train.parquet")
    config = tmp_path / "config"
    _write_forecast_config(
        config,
        [{"name": "gbm", "type": "gbm", "hyperparameters": "{n_estimators: 2}"}],
        horizon_hours=24,
        n_windows=2,
    )
    fits = []
    real_get_model = training.get_model

    def tracked_get_model(*args, **kwargs):
        model = real_get_model(*args, **kwargs)
        real_fit = model.fit

        def fit(X, y):
            fits.append(X.index[-1])
            result = real_fit(X, y)
            del model.fit  # Keep the serialized model free of test-only closures.
            return result

        model.fit = fit
        return model

    monkeypatch.setattr(training, "get_model", tracked_get_model)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("chronology")
    result = training.train_forecast_models(tmp_path, "test", config, uri)
    assert "gbm" in result["models"]
    run_id = result["models"]["gbm"]["mlflow_run_id"]
    protocol = mlflow.artifacts.load_dict(f"runs:/{run_id}/validation_protocol.json")
    assert all(fits[0] < pd.Timestamp(o) for o in protocol["cutoffs"])
    assert fits[1] == frame.index[-1]
