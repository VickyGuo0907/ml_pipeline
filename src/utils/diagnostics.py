"""Cross-validation and residual diagnostics shared by every pipeline.

Two concerns live here, kept separate because they generalize differently:

1. Cross-validation applies to any model and any regression pipeline. A single
   train/test split gives one estimate with no sense of its variability; k-fold
   CV gives a mean and a spread, which is what should be compared when choosing
   between models.

2. Residual diagnostics test *linear-model* assumptions — independence,
   homoscedasticity, normality of errors. They are meaningful for OLS, ridge,
   lasso and elastic net, and meaningless for tree ensembles, which assume none
   of those things. Callers must therefore gate these on model type rather than
   running them for every model; see DiagnosticsConfig.linear_types.

Both are optional and default to off, so pipelines that do not opt in behave
exactly as they did before this module existed.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

logger = logging.getLogger(__name__)

# Scoring names understood by sklearn's cross_val_score. RMSE is exposed as a
# negated score by sklearn convention; we flip the sign back when reporting.
_R2 = "r2"
_NEG_RMSE = "neg_root_mean_squared_error"


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = 5,
    group_column: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run k-fold cross-validation and summarize R² and RMSE across folds.

    When `group_column` names a column in X, GroupKFold is used so that every row
    sharing a group value stays inside a single fold. This matters when
    observations are not independent — e.g. hospitals in the same state share
    patient populations and referral patterns, so a plain random split can place
    near-neighbours on both sides and produce an optimistic score.

    Args:
        model: Unfitted sklearn-compatible estimator. Cloned before each fit, so
            the caller's instance is left untouched.
        X: Feature matrix.
        y: Target vector.
        folds: Number of folds.
        group_column: Optional column in X whose values define groups.
        random_state: Seed for the shuffled KFold (ignored by GroupKFold, which
            is deterministic).

    Returns:
        Dict with fold count, the strategy used, per-fold R², and mean/std
        summaries for R² and RMSE.

    Raises:
        ValueError: If folds < 2, or if group_column is not a column of X.
    """
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")

    groups = None
    if group_column is not None:
        if group_column not in X.columns:
            raise ValueError(
                f"group_column '{group_column}' not found in feature matrix. "
                f"Set cross_validation.group_column to null, or to one of the "
                f"engineered feature columns."
            )
        groups = X[group_column].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < folds:
            logger.warning(
                "group_column '%s' has only %d distinct values but %d folds were "
                "requested; reducing folds to %d",
                group_column, n_groups, folds, n_groups,
            )
            folds = n_groups
        # GroupKFold.split() returns a generator, which cross_val_score would
        # exhaust on the first metric. Materializing it means R² and RMSE are
        # scored on exactly the same partition.
        cv: Any = list(GroupKFold(n_splits=folds).split(X, y, groups))
        strategy = f"grouped_kfold[{group_column}]"
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        strategy = "kfold_shuffled"

    r2 = cross_val_score(clone(model), X, y, cv=cv, scoring=_R2)
    rmse = -cross_val_score(clone(model), X, y, cv=cv, scoring=_NEG_RMSE)

    return {
        "strategy": strategy,
        "folds": int(folds),
        "cv_r2_mean": float(np.mean(r2)),
        "cv_r2_std": float(np.std(r2)),
        "cv_r2_folds": [float(v) for v in r2],
        "cv_rmse_mean": float(np.mean(rmse)),
        "cv_rmse_std": float(np.std(rmse)),
    }


def residual_diagnostics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Test the classical linear-regression assumptions on a model's residuals.

    Only call this for linear model families. Tree ensembles make no assumptions
    about the distribution or variance of their errors, so these statistics are
    not interpretable for them.

    Tests performed:
      * Durbin-Watson — independence of errors. 2.0 means no autocorrelation;
        roughly 1.5–2.5 is conventionally treated as acceptable.
      * Breusch-Pagan — homoscedasticity. p > 0.05 fails to reject constant
        variance, which is the desired outcome.
      * Shapiro-Wilk and Jarque-Bera — normality of residuals. Shapiro-Wilk is
        sensitive at large n and will flag departures too small to matter, which
        is why a second test and the skew/kurtosis values are reported alongside.

    Args:
        y_true: Observed target values.
        y_pred: Model predictions, aligned with y_true.

    Returns:
        Dict of test statistics and p-values. Returns an empty dict if the
        statistical backend is unavailable or the sample is too small to test.
    """
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if resid.size < 20:
        logger.warning("Only %d residuals — skipping diagnostics", resid.size)
        return {}

    try:
        from scipy import stats
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson
    except ImportError as e:  # pragma: no cover - dependency is declared in pyproject
        logger.warning("Diagnostics unavailable (%s)", e)
        return {}

    out: dict[str, float] = {
        "durbin_watson": float(durbin_watson(resid)),
        "resid_skew": float(stats.skew(resid)),
        "resid_kurtosis": float(stats.kurtosis(resid)),
    }

    # Breusch-Pagan regresses squared residuals on the fitted values; a constant
    # column is required for the auxiliary regression's intercept.
    try:
        exog = np.column_stack([np.ones_like(y_pred, dtype=float), np.asarray(y_pred, dtype=float)])
        out["breusch_pagan_p"] = float(het_breuschpagan(resid, exog)[1])
    except Exception as e:
        logger.warning("Breusch-Pagan failed: %s", e)

    # Shapiro-Wilk is only defined up to n=5000 in scipy.
    if resid.size <= 5000:
        try:
            out["shapiro_p"] = float(stats.shapiro(resid).pvalue)
        except Exception as e:
            logger.warning("Shapiro-Wilk failed: %s", e)
    try:
        out["jarque_bera_p"] = float(stats.jarque_bera(resid).pvalue)
    except Exception as e:
        logger.warning("Jarque-Bera failed: %s", e)

    return out
