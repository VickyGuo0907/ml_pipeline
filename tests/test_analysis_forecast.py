"""Tests for pairing integrity and descriptive weather error comparisons."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.analysis_forecast import compare_weather_predictions, weather_training_summary


def predictions() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=48, freq="h")
    return pd.DataFrame(
        {
            "actual": 100.0,
            "predicted": 110.0,
            "origin": idx.normalize(),
            "hour": idx.hour,
            "season": "winter",
            "holiday": (idx.day == 1).astype(int),
        },
        index=idx,
    )


def test_paired_comparison_counts_wins_losses_and_gains():
    base = predictions()
    wx = base.copy()
    wx.iloc[:24, wx.columns.get_loc("predicted")] = 102.0
    wx.iloc[24:, wx.columns.get_loc("predicted")] = 112.0
    result, daily = compare_weather_predictions(wx, base.sample(frac=1, random_state=42))
    assert result["overall"]["mae_gain_mw"] == pytest.approx(3.0)
    assert result["overall"]["mape_gain_pp"] == pytest.approx(3.0)
    assert result["days_improved"] == 1
    assert result["days_worse"] == 1
    assert result["days_tied"] == 0
    assert len(daily) == 2
    assert result["errors_by"]["holiday"]["1"]["n_days"] == 1
    assert sum(g["n_hours"] for g in result["errors_by"]["hour"].values()) == 48


@pytest.mark.parametrize("change", ["missing", "cutoff", "actual", "duplicate", "label"])
def test_comparison_rejects_noncomparable_predictions(change):
    base = predictions()
    wx = base.copy()
    if change == "missing":
        wx = wx.iloc[:-1]
    elif change == "cutoff":
        wx.origin += pd.Timedelta(hours=1)
    elif change == "actual":
        wx.iloc[0, wx.columns.get_loc("actual")] += 1
    elif change == "duplicate":
        wx = pd.concat([wx, wx.iloc[:1]])
    else:
        wx.iloc[0, wx.columns.get_loc("season")] = "summer"
    with pytest.raises(ValueError):
        compare_weather_predictions(wx, base)


def test_training_temperature_bins_include_extremes_and_count_all_hours():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-07-01", "2020-07-02"])
    train = pd.DataFrame(
        {"load": [100.0, 120.0, 130.0, 150.0], "wx_temp_f": [-5.0, 5.0, 85.0, 115.0]}, index=idx
    )
    result = weather_training_summary(train, "load")
    assert result.n_hours.sum() == len(train)
    assert set(result.season) == {"winter", "summer"}
    assert result.loc[result.temperature_mean_f == -5, "load_mean_mw"].item() == 100
    train.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        weather_training_summary(train, "load")
