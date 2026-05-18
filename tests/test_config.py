"""Tests for configuration loading and validation."""
import pytest

from src.utils.config import (
    CleaningConfig,
    FeaturesConfig,
    ModelsConfig,
    PipelineConfig,
    load_cleaning_config,
    load_features_config,
    load_models_config,
    load_pipeline_config,
)


def test_load_pipeline_config():
    """Test loading and validating pipeline configuration."""
    config = load_pipeline_config("config")
    assert isinstance(config, PipelineConfig)
    assert config.target.name == "ExcessReadmissionRatio"
    assert config.target.type == "continuous"
    assert config.problem_type.value == "regression"
    assert config.train_test_split == 0.8
    assert len(config.sources) >= 1


def test_load_cleaning_config():
    """Test loading and validating cleaning configuration."""
    config = load_cleaning_config("config")
    assert isinstance(config, CleaningConfig)
    assert config.missing_strategy == "drop"
    assert len(config.steps) >= 1
    assert config.steps[0].name == "type_coercion"


def test_load_features_config():
    """Test loading and validating features configuration."""
    config = load_features_config("config")
    assert isinstance(config, FeaturesConfig)
    assert config.nzv_threshold == 0.95
    assert config.scale is True
    assert "State" in config.encoding
    assert len(config.drop_columns) >= 1


def test_load_models_config():
    """Test loading and validating models configuration."""
    config = load_models_config("config")
    assert isinstance(config, ModelsConfig)
    assert len(config.models) >= 2
    assert config.models[0].name == "linear_baseline"
    assert config.models[1].name == "lightgbm_gbm"
    assert config.random_state == 42


def test_pipeline_config_validation():
    """Test that PipelineConfig enforces constraints."""
    # Valid config should instantiate
    valid_data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "target", "type": "continuous"},
        "problem_type": "regression",
        "train_test_split": 0.8,
    }
    config = PipelineConfig(**valid_data)
    assert config.train_test_split == 0.8

    # Invalid split ratio should raise
    invalid_data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "target", "type": "continuous"},
        "problem_type": "regression",
        "train_test_split": 1.5,  # Invalid: > 1.0
    }
    with pytest.raises(ValueError):
        PipelineConfig(**invalid_data)

    # Empty sources should raise
    no_sources = {
        "sources": [],
        "target": {"name": "target", "type": "continuous"},
        "problem_type": "regression",
    }
    with pytest.raises(ValueError):
        PipelineConfig(**no_sources)


def test_features_config_nzv_threshold_validation():
    """Test that FeaturesConfig enforces NZV threshold bounds."""
    valid_data = {
        "encoding": {},
        "nzv_threshold": 0.95,
    }
    config = FeaturesConfig(**valid_data)
    assert config.nzv_threshold == 0.95

    # Invalid threshold should raise
    invalid_data = {"encoding": {}, "nzv_threshold": 1.5}
    with pytest.raises(ValueError):
        FeaturesConfig(**invalid_data)
