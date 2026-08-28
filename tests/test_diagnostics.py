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


# --------------------------- feature importance ---------------------------

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from src.utils.config import FeatureImportanceConfig  # noqa: E402
from src.utils.diagnostics import feature_importance  # noqa: E402


def test_importance_from_linear_model_keeps_sign():
    X, y = _linear_data()
    m = LinearRegression().fit(X, y)
    out = feature_importance(m, list(X.columns))
    assert out["source"] == "coef_"
    assert out["signed"] is True
    assert out["n_features"] == 3
    # x1 has the largest true coefficient (2.0), so it should rank first
    assert out["ranking"][0]["feature"] == "x1"


def test_importance_from_tree_model_is_unsigned():
    X, y = _linear_data()
    m = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    out = feature_importance(m, list(X.columns))
    assert out["source"] == "feature_importances_"
    assert out["signed"] is False
    assert all(r["value"] >= 0 for r in out["ranking"])


def test_importance_ranks_by_magnitude_not_value():
    """A large negative coefficient must outrank a small positive one."""
    class Fake:
        coef_ = np.array([0.1, -0.9, 0.3])
    out = feature_importance(Fake(), ["a", "b", "c"])
    assert [r["feature"] for r in out["ranking"]] == ["b", "c", "a"]
    assert out["ranking"][0]["value"] == -0.9  # sign preserved


def test_importance_counts_nonzero_for_sparse_models():
    class Fake:
        coef_ = np.array([0.0, 0.5, 0.0, -0.2])
    out = feature_importance(Fake(), ["a", "b", "c", "d"])
    assert out["n_nonzero"] == 2
    assert out["n_features"] == 4


def test_importance_returns_none_for_unsupported_model():
    class NoAttrs:
        pass
    assert feature_importance(NoAttrs(), ["a"]) is None


def test_importance_returns_none_on_length_mismatch():
    class Fake:
        coef_ = np.array([0.1, 0.2])
    assert feature_importance(Fake(), ["a", "b", "c"]) is None


def test_feature_importance_config_defaults_to_disabled():
    cfg = FeatureImportanceConfig()
    assert cfg.enabled is False
    assert cfg.top_n == 10


# --------------------------- hyperparameter tuning ---------------------------

from sklearn.linear_model import Ridge  # noqa: E402

from src.utils.config import TuningConfig  # noqa: E402
from src.utils.diagnostics import tune_model  # noqa: E402


def test_tuning_returns_best_params_and_unfitted_estimator():
    X, y = _linear_data()
    tuned, summary = tune_model(
        Ridge(), X, y,
        param_distributions={"alpha": [0.01, 1.0, 100.0]},
        n_iter=3, folds=3,
    )
    assert summary["best_params"]["alpha"] in (0.01, 1.0, 100.0)
    assert summary["n_candidates"] == 3
    assert tuned.get_params()["alpha"] == summary["best_params"]["alpha"]
    # Returned estimator must be unfitted — the caller controls the final fit.
    with pytest.raises(Exception):
        tuned.predict(X)


def test_tuning_picks_the_better_alpha_on_clean_linear_data():
    """With a strong linear signal, heavy regularization should lose."""
    X, y = _linear_data()
    _, summary = tune_model(
        Ridge(), X, y,
        param_distributions={"alpha": [0.01, 10000.0]},
        n_iter=2, folds=3,
    )
    assert summary["best_params"]["alpha"] == 0.01


def test_tuning_does_not_mutate_caller_model():
    X, y = _linear_data()
    model = Ridge(alpha=1.0)
    tune_model(model, X, y, param_distributions={"alpha": [0.01, 100.0]}, n_iter=2, folds=3)
    assert model.get_params()["alpha"] == 1.0


def test_tuning_uses_grouped_folds_when_group_column_given():
    X, y = _linear_data()
    _, summary = tune_model(
        Ridge(), X, y,
        param_distributions={"alpha": [0.01, 1.0]},
        n_iter=2, folds=3, group_column="group",
    )
    assert summary["strategy"] == "grouped_kfold[group]"


def test_tuning_caps_n_iter_at_space_size():
    """A space smaller than n_iter should be searched exhaustively, not resampled."""
    X, y = _linear_data()
    _, summary = tune_model(
        Ridge(), X, y,
        param_distributions={"alpha": [0.1, 1.0]},
        n_iter=50, folds=3,
    )
    assert summary["space_size"] == 2
    assert summary["n_candidates"] == 2


def test_tuning_rejects_empty_search_space():
    X, y = _linear_data()
    with pytest.raises(ValueError, match="param_distributions is empty"):
        tune_model(Ridge(), X, y, param_distributions={}, folds=3)


def test_tuning_is_reproducible_under_a_fixed_seed():
    X, y = _linear_data()
    space = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    _, a = tune_model(Ridge(), X, y, param_distributions=space, n_iter=3, folds=3, random_state=7)
    _, b = tune_model(Ridge(), X, y, param_distributions=space, n_iter=3, folds=3, random_state=7)
    assert a["best_params"] == b["best_params"]
    assert a["best_score"] == pytest.approx(b["best_score"])


def test_tuning_summary_is_yaml_serializable():
    """numpy scalars in best_params would break the evaluation report dump."""
    import yaml
    X, y = _linear_data()
    _, summary = tune_model(
        Ridge(), X, y,
        param_distributions={"alpha": [0.01, 1.0]}, n_iter=2, folds=3,
    )
    yaml.safe_dump(summary)


# --------------------------- tuning config ---------------------------

def test_tuning_defaults_to_disabled():
    cfg = ModelsConfig(models=[{"name": "m", "type": "ols"}])
    assert cfg.tuning.enabled is False
    assert cfg.tuning.param_distributions == {}
    assert cfg.tuning.n_iter == 25
    assert cfg.tuning.scoring == "r2"


def test_tuning_config_rejects_single_fold():
    with pytest.raises(ValueError):
        TuningConfig(folds=1)


def test_tuning_config_rejects_zero_iterations():
    with pytest.raises(ValueError):
        TuningConfig(n_iter=0)


# --------------------------- new model type ---------------------------

def test_hist_gbm_is_registered_and_constructible():
    from sklearn.ensemble import HistGradientBoostingRegressor

    from src.utils.model_registry import MODEL_REGISTRY, get_model
    assert "hist_gbm" in MODEL_REGISTRY
    model = get_model("hist_gbm", {"learning_rate": 0.05, "max_iter": 50})
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.get_params()["learning_rate"] == 0.05
