"""Tests for configuration loading and validation."""
import pytest

from src.utils.config import (
    CleaningConfig,
    FeaturesConfig,
    ModelsConfig,
    OrchestrationConfig,
    PipelineConfig,
    discover_pipelines,
    load_cleaning_config,
    load_features_config,
    load_models_config,
    load_pipeline_config,
    load_pipeline_orchestration_config,
)

BIOMEDICAL_CONFIG = "config/biomedical_clinical"


def test_load_pipeline_config():
    """Test loading and validating the biomedical_clinical pipeline configuration."""
    config = load_pipeline_config(BIOMEDICAL_CONFIG)
    assert isinstance(config, PipelineConfig)
    assert config.target.name == "Excess Readmission Ratio"
    assert config.target.type == "continuous"
    assert config.problem_type.value == "regression"
    assert config.train_test_split == 0.81
    assert config.pipeline_type == "biomedical_clinical"
    assert len(config.sources) >= 1


def test_load_cleaning_config():
    """Test loading and validating the biomedical_clinical cleaning configuration."""
    config = load_cleaning_config(BIOMEDICAL_CONFIG)
    assert isinstance(config, CleaningConfig)
    assert config.impute_strategy == "iterative"
    assert len(config.drop_column_patterns) > 0
    assert "Payment" in config.drop_column_patterns
    assert len(config.steps) >= 1
    assert config.steps[0].name == "type_coercion"


def test_load_features_config():
    """Test loading and validating the biomedical_clinical features configuration."""
    config = load_features_config(BIOMEDICAL_CONFIG)
    assert isinstance(config, FeaturesConfig)
    assert config.nzv_threshold == 0.95
    assert config.scale is True
    assert "State" in config.encoding
    assert config.encoding["State"] == "frequency"
    assert config.boxcox_target is True
    assert config.vif_threshold is None  # disabled: HCAHPS questions are correlated by design
    assert len(config.drop_columns) >= 1


def test_load_models_config():
    """Test loading and validating the biomedical_clinical models configuration."""
    config = load_models_config(BIOMEDICAL_CONFIG)
    assert isinstance(config, ModelsConfig)
    model_names = [m.name for m in config.models]
    assert "ols_baseline" in model_names
    assert "elastic_net" in model_names
    assert "ridge_l2" in model_names
    assert "lasso_l1" in model_names
    assert "random_forest" in model_names
    assert "lightgbm_gbm" in model_names
    assert config.random_state == 42


def test_discover_pipelines():
    """Test that discover_pipelines finds all pipeline config directories."""
    pipelines = discover_pipelines("config")
    dag_ids = [p.name for p in pipelines]
    assert "biomedical_clinical" in dag_ids
    assert "bioinfo_gene" in dag_ids
    assert "base" not in dag_ids


def test_load_pipeline_orchestration_config_merges_defaults():
    """Test that pipeline orchestration merges base defaults with pipeline overrides."""
    config = load_pipeline_orchestration_config(
        pipeline_dir=BIOMEDICAL_CONFIG,
        base_dir="config/base",
    )
    assert isinstance(config, OrchestrationConfig)
    assert config.dag.dag_id == "biomedical_clinical_pipeline"
    assert config.dag.schedule == "@weekly"
    # Base defaults applied
    assert config.tasks.retries == 1
    assert config.mlflow.tracking_uri == "http://mlflow-server:5000"
    # Pipeline directories applied
    assert config.directories.config == "config/biomedical_clinical"


def test_load_bioinfo_gene_orchestration_config():
    """Test bioinfo_gene pipeline config loads and merges correctly."""
    config = load_pipeline_orchestration_config(
        pipeline_dir="config/bioinfo_gene",
        base_dir="config/base",
    )
    assert config.dag.dag_id == "bioinfo_gene_pipeline"
    assert config.dag.schedule == "@monthly"
    assert config.tasks.retries == 1  # inherited from base


def test_pipeline_config_validation():
    """Test that PipelineConfig enforces constraints."""
    valid_data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "target", "type": "continuous"},
        "problem_type": "regression",
        "train_test_split": 0.8,
    }
    config = PipelineConfig(**valid_data)
    assert config.train_test_split == 0.8
    assert config.pipeline_type == "generic"  # default

    invalid_data = {**valid_data, "train_test_split": 1.5}
    with pytest.raises(ValueError):
        PipelineConfig(**invalid_data)

    no_sources = {**valid_data, "sources": []}
    with pytest.raises(ValueError):
        PipelineConfig(**no_sources)


def test_features_config_nzv_threshold_validation():
    """Test that FeaturesConfig enforces NZV threshold bounds."""
    valid_data = {"encoding": {}, "nzv_threshold": 0.95}
    config = FeaturesConfig(**valid_data)
    assert config.nzv_threshold == 0.95
    assert config.boxcox_target is False  # default
    assert config.vif_threshold is None  # default

    invalid_data = {"encoding": {}, "nzv_threshold": 1.5}
    with pytest.raises(ValueError):
        FeaturesConfig(**invalid_data)


def test_cleaning_config_defaults():
    """Test CleaningConfig defaults are backward-compatible."""
    config = CleaningConfig(steps=[])
    assert config.impute_strategy == "median"
    assert config.drop_column_patterns == []
    assert config.missing_strategy == "drop"


def test_pipeline_config_pipeline_type_default():
    """Test that pipeline_type defaults to 'generic' for new pipelines."""
    data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "y", "type": "continuous"},
        "problem_type": "regression",
    }
    config = PipelineConfig(**data)
    assert config.pipeline_type == "generic"
