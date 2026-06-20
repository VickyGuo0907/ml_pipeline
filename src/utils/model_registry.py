"""Model type registry — add a new model type by inserting one line here."""
from typing import Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

# Map config `type` strings to sklearn-compatible estimator classes.
# To add a new model: MODEL_REGISTRY["my_type"] = MyEstimatorClass
MODEL_REGISTRY: dict[str, type] = {
    "ols": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "random_forest": RandomForestRegressor,
    "gbm": LGBMRegressor,
}


def get_model(model_type: str, hyperparameters: dict[str, Any]) -> Any:
    """Instantiate a model from the registry.

    Args:
        model_type: Registry key matching a model class.
        hyperparameters: Keyword arguments passed to the class constructor.

    Returns:
        Fitted-ready estimator instance.

    Raises:
        ValueError: If model_type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Registered types: {available}. "
            f"Add it to src/utils/model_registry.py to enable."
        )
    return MODEL_REGISTRY[model_type](**hyperparameters)


def register_model(type_key: str, model_class: type) -> None:
    """Register a custom estimator class under a new type key.

    Args:
        type_key: String key to use in models.yaml `type` field.
        model_class: Estimator class (must follow sklearn fit/predict API).
    """
    MODEL_REGISTRY[type_key] = model_class
