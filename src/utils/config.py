"""Configuration models and loaders using Pydantic."""
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ProblemType(str, Enum):
    """Machine learning problem type."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class TargetConfig(BaseModel):
    """Target column configuration."""

    name: str = Field(..., description="Target column name")
    type: str = Field(
        ..., description="Target type: 'continuous' for regression, 'categorical' for classification"
    )


class SourceConfig(BaseModel):
    """Data source configuration."""

    name: str = Field(..., description="Source name identifier")
    path: str = Field(..., description="Path to source data")
    format: str = Field(default="csv", description="Data format (csv, parquet, etc)")


class PipelineConfig(BaseModel):
    """Root pipeline configuration."""

    sources: list[SourceConfig] = Field(..., description="Data sources")
    target: TargetConfig = Field(..., description="Target column configuration")
    problem_type: ProblemType = Field(..., description="ML problem type")
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)
    random_state: int = Field(default=42)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[SourceConfig]) -> list[SourceConfig]:
        """Ensure at least one source is defined."""
        if not v:
            raise ValueError("At least one source must be defined")
        return v


class CleaningStep(BaseModel):
    """Single cleaning transformation step."""

    name: str = Field(..., description="Step identifier")
    type: str = Field(..., description="Cleaning operation type")
    columns: Optional[list[str]] = Field(default=None, description="Columns to apply to")
    params: dict[str, Any] = Field(default_factory=dict, description="Step parameters")


class CleaningConfig(BaseModel):
    """Data cleaning configuration."""

    steps: list[CleaningStep] = Field(..., description="Cleaning steps in order")
    missing_strategy: str = Field(default="drop", description="Strategy for missing values")
    duplicates_subset: Optional[list[str]] = Field(
        default=None, description="Columns to check for duplicates"
    )


class FeatureEngineeringStep(BaseModel):
    """Feature engineering transformation step."""

    name: str = Field(..., description="Feature name/identifier")
    type: str = Field(..., description="Feature type: categorical, numerical, etc")
    source_columns: list[str] = Field(..., description="Input columns")
    operation: str = Field(..., description="Operation to perform")
    params: dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    encoding:dict[str, str] = Field(
        default_factory=dict, description="Column encoding mapping"
    )
    steps: list[FeatureEngineeringStep] = Field(
        default_factory=list, description="Feature engineering steps"
    )
    nzv_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Near-zero variance threshold"
    )
    drop_columns: list[str] = Field(default_factory=list, description="Columns to drop")
    scale: bool = Field(default=True, description="Whether to scale features")


class ModelConfig(BaseModel):
    """Individual model configuration."""

    name: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type: linear, gbm, etc")
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )


class ModelsConfig(BaseModel):
    """Models configuration."""

    models: list[ModelConfig] = Field(..., description="Model definitions")
    random_state: int = Field(default=42)
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Loaded configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config(config_dir: str | Path = "config") -> PipelineConfig:
    """Load and validate pipeline configuration.

    Args:
        config_dir: Directory containing config files

    Returns:
        Validated PipelineConfig
    """
    config_path = Path(config_dir) / "pipeline.yaml"
    config_data = load_config(config_path)
    return PipelineConfig(**config_data)


def load_cleaning_config(config_dir: str | Path = "config") -> CleaningConfig:
    """Load and validate cleaning configuration.

    Args:
        config_dir: Directory containing config files

    Returns:
        Validated CleaningConfig
    """
    config_path = Path(config_dir) / "cleaning.yaml"
    config_data = load_config(config_path)
    return CleaningConfig(**config_data)


def load_features_config(config_dir: str | Path = "config") -> FeaturesConfig:
    """Load and validate features configuration.

    Args:
        config_dir: Directory containing config files

    Returns:
        Validated FeaturesConfig
    """
    config_path = Path(config_dir) / "features.yaml"
    config_data = load_config(config_path)
    return FeaturesConfig(**config_data)


def load_models_config(config_dir: str | Path = "config") -> ModelsConfig:
    """Load and validate models configuration.

    Args:
        config_dir: Directory containing config files

    Returns:
        Validated ModelsConfig
    """
    config_path = Path(config_dir) / "models.yaml"
    config_data = load_config(config_path)
    return ModelsConfig(**config_data)
