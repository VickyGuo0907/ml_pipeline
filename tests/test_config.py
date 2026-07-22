"""Tests for configuration loading and validation."""
import pytest

from src.utils.config import (
    BenchmarkConfig,
    CleaningConfig,
    FeaturesConfig,
    JoinDirectConfig,
    JoinStrategyConfig,
    ModelsConfig,
    OrchestrationConfig,
    PipelineConfig,
    UnsupervisedConfig,
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


def test_load_unsupervised_config():
    """Test unsupervised exploration config is loaded from pipeline.yaml."""
    config = load_pipeline_config(BIOMEDICAL_CONFIG)
    assert isinstance(config.unsupervised, UnsupervisedConfig)
    assert config.unsupervised.enabled is True
    assert config.unsupervised.pca.enabled is True
    assert config.unsupervised.clustering.algorithm == "kmeans"
    assert config.unsupervised.clustering.max_k == 10


def test_unsupervised_config_defaults():
    """Test UnsupervisedConfig defaults are sensible when not specified in YAML."""
    cfg = UnsupervisedConfig()
    assert cfg.enabled is True
    assert cfg.pca.enabled is True
    assert cfg.clustering.algorithm == "kmeans"
    assert cfg.clustering.max_k == 10


def test_load_cleaning_config():
    """Test loading and validating the biomedical_clinical cleaning configuration."""
    config = load_cleaning_config(BIOMEDICAL_CONFIG)
    assert isinstance(config, CleaningConfig)
    assert config.impute_strategy == "median"
    assert len(config.drop_column_patterns) > 0
    assert "Footnote" in config.drop_column_patterns


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
    config = CleaningConfig()
    assert config.impute_strategy == "median"
    assert config.drop_column_patterns == []
    assert config.duplicates_subset is None


def test_join_strategy_config_direct_joins_default_empty():
    """Test JoinStrategyConfig.direct_joins defaults to an empty list."""
    config = JoinStrategyConfig()
    assert config.direct_joins == []


def test_join_strategy_config_accepts_direct_joins():
    """Test JoinStrategyConfig parses a direct_joins list of file patterns."""
    config = JoinStrategyConfig(
        enabled=True,
        id_column="Facility ID",
        direct_joins=[{"file_pattern": "Hospital_General_Information"}],
    )
    assert len(config.direct_joins) == 1
    assert isinstance(config.direct_joins[0], JoinDirectConfig)
    assert config.direct_joins[0].file_pattern == "Hospital_General_Information"


def test_pipeline_config_pipeline_type_default():
    """Test that pipeline_type defaults to 'generic' for new pipelines."""
    data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "y", "type": "continuous"},
        "problem_type": "regression",
    }
    config = PipelineConfig(**data)
    assert config.pipeline_type == "generic"


def test_benchmark_config_defaults():
    """Test BenchmarkConfig defaults to disabled — opt-in per pipeline."""
    cfg = BenchmarkConfig()
    assert cfg.enabled is False


def test_pipeline_config_benchmark_defaults_disabled():
    """Test PipelineConfig.benchmark defaults to disabled when not specified in YAML."""
    data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "y", "type": "continuous"},
        "problem_type": "regression",
    }
    config = PipelineConfig(**data)
    assert config.benchmark.enabled is False


def test_biomedical_clinical_benchmark_enabled():
    """Test the biomedical_clinical pipeline opts into the benchmark set."""
    config = load_pipeline_config(BIOMEDICAL_CONFIG)
    assert config.benchmark.enabled is True


def test_bioinfo_gene_benchmark_disabled():
    """Test bioinfo_gene stays lean and does not opt into the benchmark set."""
    config = load_pipeline_config("config/bioinfo_gene")
    assert config.benchmark.enabled is False


def test_orchestration_directories_config_has_benchmark_field():
    """Test OrchestrationDirectoriesConfig exposes a benchmark directory."""
    config = load_pipeline_orchestration_config(
        pipeline_dir=BIOMEDICAL_CONFIG,
        base_dir="config/base",
    )
    assert config.directories.benchmark == "data/biomedical_clinical/benchmark"


LAGGED_CONFIG = "config/hospital_readmission_lagged"


def test_load_hospital_readmission_lagged_pipeline_config():
    """Test the lagged pipeline targets the 2025 HRRP file with a PN filter downstream."""
    config = load_pipeline_config(LAGGED_CONFIG)
    assert config.target.name == "Excess Readmission Ratio"
    assert config.pipeline_type == "hospital_readmission_lagged"
    assert config.train_test_split == 0.80
    patterns = [pf.file_pattern for pf in config.validation.per_file_schemas]
    assert "FY_2025_Hospital_Readmissions_Reduction_Program" in patterns
    assert "Hospital_General_Information" in patterns


def test_load_hospital_readmission_lagged_features_config():
    """Test the lagged pipeline's features.yaml wires the 2025 spine + 2024 pivots/direct-join."""
    config = load_features_config(LAGGED_CONFIG)
    assert config.join_strategy.enabled is True
    assert config.join_strategy.spine.file_pattern == "FY_2025_Hospital_Readmissions_Reduction_Program"
    assert config.join_strategy.spine.measure_value == "READM-30-PN-HRRP"
    pivot_patterns = [p.file_pattern for p in config.join_strategy.pivots]
    assert "HCAHPS" in pivot_patterns
    assert "Timely_and_Effective_Care" in pivot_patterns
    assert "Complications_and_Deaths" in pivot_patterns
    assert "Healthcare_Associated_Infections" in pivot_patterns
    direct_patterns = [d.file_pattern for d in config.join_strategy.direct_joins]
    assert "Hospital_General_Information" in direct_patterns
    assert "Number of Readmissions" in config.drop_columns
    assert "Predicted Readmission Rate" in config.drop_columns
    assert "Expected Readmission Rate" in config.drop_columns


def test_load_hospital_readmission_lagged_cleaning_config():
    """Test the lagged pipeline protects sparse pivot-source Score/HCAHPS columns from imputation."""
    config = load_cleaning_config(LAGGED_CONFIG)
    assert "Score" in config.protect_columns
    assert "HCAHPS Linear Mean Value" in config.protect_columns


def test_load_hospital_readmission_lagged_models_config():
    """Test the lagged pipeline's model ladder matches the capstone validation plan (2+ supervised)."""
    config = load_models_config(LAGGED_CONFIG)
    model_names = [m.name for m in config.models]
    assert "elastic_net" in model_names
    assert "lightgbm_gbm" in model_names
    assert len(model_names) >= 2


def test_hospital_readmission_lagged_orchestration_config():
    """Test the lagged pipeline's DAG id and directories are wired independently of biomedical_clinical."""
    config = load_pipeline_orchestration_config(
        pipeline_dir=LAGGED_CONFIG,
        base_dir="config/base",
    )
    assert config.dag.dag_id == "hospital_readmission_lagged_pipeline"
    assert config.directories.landing == "data/hospital_readmission_lagged/landing"
    assert config.directories.config == "config/hospital_readmission_lagged"
    assert config.tasks.retries == 1  # inherited from base


def test_discover_pipelines_includes_hospital_readmission_lagged():
    """Test discover_pipelines picks up the new pipeline directory alongside the existing two."""
    pipelines = discover_pipelines("config")
    names = [p.name for p in pipelines]
    assert "hospital_readmission_lagged" in names
    assert "biomedical_clinical" in names
    assert "bioinfo_gene" in names
