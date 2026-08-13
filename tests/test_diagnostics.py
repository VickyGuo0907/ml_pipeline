"""Tests for cross-validation and residual diagnostics (src/utils/diagnostics.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.utils.config import CrossValidationConfig, DiagnosticsConfig, ModelsConfig
from src.utils.diagnostics import cross_validate_model, residual_diagnostics


def _linear_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "group": rng.integers(0, 6, size=n),
    })
    y = pd.Series(2.0 * X["x1"] + 0.5 * X["x2"] + rng.normal(scale=0.2, size=n))
    return X, y


# --------------------------- cross-validation ---------------------------

def test_cv_returns_summary_and_per_fold_scores():
    X, y = _linear_data()
    out = cross_validate_model(LinearRegression(), X, y, folds=5)
    assert out["folds"] == 5
    assert out["strategy"] == "kfold_shuffled"
    assert len(out["cv_r2_folds"]) == 5
    # A well-specified linear model on linear data should score high.
    assert out["cv_r2_mean"] > 0.9
    assert out["cv_r2_std"] >= 0.0
    assert out["cv_rmse_mean"] > 0.0


def test_cv_grouped_keeps_groups_whole():
    X, y = _linear_data()
    out = cross_validate_model(LinearRegression(), X, y, folds=3, group_column="group")
    assert out["strategy"] == "grouped_kfold[group]"
    assert out["folds"] == 3
    assert len(out["cv_r2_folds"]) == 3


def test_cv_does_not_mutate_caller_model():
    """The estimator passed in must remain unfitted — CV clones internally."""
    X, y = _linear_data()
    model = LinearRegression()
    cross_validate_model(model, X, y, folds=3)
    with pytest.raises(Exception):
        model.predict(X)  # unfitted estimators raise NotFittedError


def test_cv_rejects_missing_group_column():
    X, y = _linear_data()
    with pytest.raises(ValueError, match="not found in feature matrix"):
        cross_validate_model(LinearRegression(), X, y, group_column="nope")


def test_cv_rejects_too_few_folds():
    X, y = _linear_data()
    with pytest.raises(ValueError, match="folds must be >= 2"):
        cross_validate_model(LinearRegression(), X, y, folds=1)


def test_cv_reduces_folds_when_groups_are_scarce():
    """More folds than groups is impossible; folds should fall back to n_groups."""
    X, y = _linear_data()
    X["group"] = np.repeat([0, 1, 2], len(X) // 3)
    out = cross_validate_model(LinearRegression(), X, y, folds=5, group_column="group")
    assert out["folds"] == 3


# --------------------------- residual diagnostics ---------------------------

def test_diagnostics_on_clean_residuals_meet_assumptions():
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.normal(size=500))
    y_pred = y_true + rng.normal(scale=0.3, size=500)  # iid gaussian noise
    d = residual_diagnostics(y_true, y_pred.to_numpy())
    assert 1.5 < d["durbin_watson"] < 2.5          # independence
    assert d["breusch_pagan_p"] > 0.05             # constant variance
    assert d["jarque_bera_p"] > 0.05               # normality
    assert abs(d["resid_skew"]) < 0.5


def test_diagnostics_detect_autocorrelation():
    """Strongly autocorrelated residuals should push Durbin-Watson well below 2."""
    n = 400
    rng = np.random.default_rng(2)
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.9 * resid[i - 1] + rng.normal(scale=0.1)
    y_pred = np.zeros(n)
    d = residual_diagnostics(pd.Series(resid), y_pred)
    assert d["durbin_watson"] < 1.0


def test_diagnostics_skipped_on_tiny_sample():
    d = residual_diagnostics(pd.Series([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.9]))
    assert d == {}


# --------------------------- config defaults ---------------------------

def test_cv_and_diagnostics_default_to_disabled():
    """Pipelines that do not opt in must behave exactly as before."""
    cfg = ModelsConfig(models=[{"name": "m", "type": "ols"}])
    assert cfg.cross_validation.enabled is False
    assert cfg.diagnostics.enabled is False
    assert cfg.cross_validation.folds == 5
    assert cfg.cross_validation.group_column is None


def test_diagnostics_config_lists_only_linear_types():
    cfg = DiagnosticsConfig()
    assert "elastic_net" in cfg.linear_types
    assert "random_forest" not in cfg.linear_types
    assert "gbm" not in cfg.linear_types


def test_cv_config_rejects_single_fold():
    with pytest.raises(ValueError):
        CrossValidationConfig(folds=1)


# --------------------------- champion selection ---------------------------

from src.evaluate import _cv_summary, _select_champion  # noqa: E402


def _models(cv_means, cv_stds, rmses):
    """Build a minimal report['models'] dict for champion selection."""
    names = [f"m{i}" for i in range(len(cv_means))]
    return names, {
        n: {
            "test_rmse": rmses[i],
            "cross_validation": {"cv_r2_mean": cv_means[i], "cv_r2_std": cv_stds[i]},
        }
        for i, n in enumerate(names)
    }


def test_champion_defaults_to_lowest_test_rmse():
    names, models = _models([0.05, 0.09], [0.02, 0.01], [0.070, 0.060])
    assert _select_champion(models, names) == "m1"


def test_champion_cv_r2_picks_highest_mean_when_not_tied():
    names, models = _models([0.05, 0.20], [0.09, 0.05], [0.060, 0.070])
    assert _select_champion(models, names, metric="cv_r2") == "m1"


def test_champion_cv_r2_breaks_ties_on_stability():
    """Mirrors the real run: means within 0.003, spreads differ 3x."""
    names, models = _models([0.0505, 0.0529, 0.0506], [0.0266, 0.0552, 0.0801],
                            [0.0652, 0.0652, 0.0644])
    # m1 has the best mean, but all three are tied within tolerance, so the
    # steadiest model (m0) should win.
    assert _select_champion(models, names, metric="cv_r2", tie_tolerance=0.005) == "m0"


def test_champion_cv_r2_falls_back_when_cv_missing():
    names, models = _models([0.05, 0.09], [0.02, 0.01], [0.070, 0.060])
    del models["m0"]["cross_validation"]
    assert _select_champion(models, names, metric="cv_r2") == "m1"


def test_cv_summary_collects_per_fold_metrics_in_order():
    metrics = {
        "cv_r2_mean": 0.05, "cv_r2_std": 0.02, "cv_rmse_mean": 0.06,
        "cv_r2_fold_1": 0.061, "cv_r2_fold_2": -0.002, "cv_r2_fold_3": 0.073,
        "cv_r2_fold_10": 0.9,
    }
    out = _cv_summary(metrics, {"cv_strategy": "grouped_kfold[State]", "cv_folds": "5"})
    # fold_10 must sort after fold_3, not between fold_1 and fold_2
    assert out["cv_r2_per_fold"] == [0.061, -0.002, 0.073, 0.9]
    assert out["strategy"] == "grouped_kfold[State]"


def test_cv_summary_returns_none_without_cv():
    assert _cv_summary({"test_r2": 0.1}, {}) is None
