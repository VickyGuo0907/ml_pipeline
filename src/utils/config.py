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
    # Identifies the pipeline variant — used for logging/tagging, not logic branching
    pipeline_type: str = Field(default="generic", description="Pipeline variant identifier")

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
    missing_strategy: str = Field(default="drop", description="Legacy: strategy for missing values")
    # Options: median | iterative (MICE/missForest-like) | knn
    impute_strategy: str = Field(default="median", description="Imputation strategy")
    # Column name substrings to drop after imputation (case-insensitive)
    drop_column_patterns: list[str] = Field(
        default_factory=list, description="Drop columns whose names contain these substrings"
    )
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

    encoding: dict[str, str] = Field(
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
    # SVG Stage 2: apply Box-Cox power transform to target before modeling
    boxcox_target: bool = Field(default=False, description="Apply Box-Cox transform to target")
    # SVG Stage 2: drop predictors with VIF > threshold; None disables
    vif_threshold: Optional[float] = Field(
        default=None, description="VIF threshold for collinearity pruning; None = disabled"
    )


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


class OrchestrationDAGConfig(BaseModel):
    """DAG-level orchestration settings."""

    dag_id: str = Field(default="pipeline", description="DAG identifier")
    owner: str = Field(default="data-eng", description="DAG owner")
    description: str = Field(
        default="End-to-end ML pipeline: ingest → train → serve",
        description="DAG description",
    )
    schedule_interval: str = Field(default="@weekly", description="Schedule interval")
    catchup: bool = Field(default=False, description="Enable catchup")
    tags: list[str] = Field(default_factory=lambda: ["ml", "production"], description="DAG tags")


class OrchestrationTaskConfig(BaseModel):
    """Task-level orchestration settings."""

    retries: int = Field(default=1, description="Default task retries")
    retry_delay_minutes: int = Field(default=5, description="Retry delay in minutes")
    train_models_retries: int = Field(default=0, description="Train models task retries")


class OrchestrationDirectoriesConfig(BaseModel):
    """Data directories configuration."""

    landing: str = Field(default="data/landing", description="Landing directory")
    raw: str = Field(default="data/raw", description="Raw data directory")
    interim: str = Field(default="data/interim", description="Interim data directory")
    features: str = Field(default="data/features", description="Features directory")
    reports: str = Field(default="reports", description="Reports directory")
    config: str = Field(default="config", description="Configuration directory")


class OrchestrationMLflowConfig(BaseModel):
    """MLflow configuration."""

    tracking_uri: str = Field(
        default="http://mlflow-server:5000", description="MLflow tracking server URI"
    )


class OrchestrationConfig(BaseModel):
    """Complete orchestration configuration for Airflow DAG."""

    dag: OrchestrationDAGConfig = Field(default_factory=OrchestrationDAGConfig)
    tasks: OrchestrationTaskConfig = Field(default_factory=OrchestrationTaskConfig)
    directories: OrchestrationDirectoriesConfig = Field(
        default_factory=OrchestrationDirectoriesConfig
    )
    mlflow: OrchestrationMLflowConfig = Field(default_factory=OrchestrationMLflowConfig)


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


def load_orchestration_config(config_dir: str | Path = "config") -> OrchestrationConfig:
    """Load and validate orchestration configuration.

    Args:
        config_dir: Directory containing config files

    Returns:
        Validated OrchestrationConfig
    """
    config_path = Path(config_dir) / "orchestration.yaml"
    if not config_path.exists():
        return OrchestrationConfig()
    config_data = load_config(config_path)
    return OrchestrationConfig(**config_data)


def load_pipeline_orchestration_config(
    pipeline_dir: str | Path,
    base_dir: str | Path = "config/base",
) -> OrchestrationConfig:
    """Load orchestration config by merging shared base defaults with pipeline overrides.

    Base defaults (config/base/defaults.yaml) are loaded first; the pipeline's
    orchestration.yaml is then deep-merged on top so pipeline values win.

    Args:
        pipeline_dir: Directory for the specific pipeline (e.g. config/biomedical_clinical)
        base_dir: Directory containing base/defaults.yaml

    Returns:
        Validated OrchestrationConfig
    """
    base_path = Path(base_dir) / "defaults.yaml"
    pipeline_path = Path(pipeline_dir) / "orchestration.yaml"

    merged: dict[str, Any] = {}
    if base_path.exists():
        merged = load_config(base_path)

    if pipeline_path.exists():
        pipeline_data = load_config(pipeline_path)
        for key, value in pipeline_data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

    return OrchestrationConfig(**merged) if merged else OrchestrationConfig()


def discover_pipelines(config_root: str | Path = "config") -> list[Path]:
    """Discover pipeline config directories under config_root.

    A valid pipeline directory contains an orchestration.yaml file.
    The base/ directory is excluded.

    Args:
        config_root: Root config directory to scan

    Returns:
        Sorted list of pipeline config directory paths
    """
    root = Path(config_root)
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name != "base" and (p / "orchestration.yaml").exists()
    )
