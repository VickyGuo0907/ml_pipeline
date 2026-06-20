"""Unsupervised exploration stage: PCA + k-means cluster analysis.

SVG Stage 3a: Unsupervised — PCA + k-means segmentation.

Runs in parallel with supervised training; results inform feature understanding
and hospital segmentation. Does NOT block or feed into the training stage.
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_pipeline_config

logger = logging.getLogger(__name__)

_MAX_K = 10  # Maximum k to test in elbow search


def _run_pca(X: pd.DataFrame) -> dict[str, Any]:
    """Fit PCA on features and return variance explained statistics.

    SVG: "PCA: 18 components — PC1 = 22%, need 11 for 80%"

    Args:
        X: Scaled feature matrix (no target).

    Returns:
        Dict with explained variance ratios and cumulative variance.
    """
    pca = PCA(n_components=min(X.shape[1], X.shape[0]))
    pca.fit(X)
    cumulative = float(pca.explained_variance_ratio_.cumsum()[
        (pca.explained_variance_ratio_.cumsum() >= 0.80).argmax()
    ])
    n_for_80 = int((pca.explained_variance_ratio_.cumsum() >= 0.80).argmax()) + 1

    return {
        "n_components": int(pca.n_components_),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "n_components_for_80pct": n_for_80,
        "cumulative_80pct": cumulative,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _find_optimal_k(X: pd.DataFrame) -> dict[str, Any]:
    """Find optimal k using WSS elbow + silhouette score.

    SVG: "K-means: optimal k = 2 — WSS elbow + silhouette agree"

    Args:
        X: Feature matrix.

    Returns:
        Dict with optimal k, inertias, and silhouette scores.
    """
    inertias: list[float] = []
    silhouettes: list[float] = []
    k_range = range(2, min(_MAX_K + 1, X.shape[0]))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))

    # Optimal k: highest silhouette score (most interpretable single criterion)
    best_idx = silhouettes.index(max(silhouettes))
    optimal_k = list(k_range)[best_idx]

    return {
        "optimal_k": optimal_k,
        "best_silhouette": silhouettes[best_idx],
        "inertias": dict(zip(k_range, inertias)),
        "silhouette_scores": dict(zip(k_range, silhouettes)),
    }


def _characterize_clusters(
    df: pd.DataFrame, labels: list[int], target_col: str
) -> dict[str, Any]:
    """Describe each cluster by target mean to assign high/low performance labels.

    SVG: "High-perf vs Low-perf hospitals"

    Args:
        df: Full feature + target DataFrame.
        labels: Cluster assignment per row.
        target_col: Name of the target column.

    Returns:
        Dict mapping cluster id → characterization stats.
    """
    df = df.copy()
    df["_cluster"] = labels
    summary: dict[str, Any] = {}

    for cluster_id, group in df.groupby("_cluster"):
        target_mean = float(group[target_col].mean()) if target_col in group.columns else None
        summary[str(cluster_id)] = {
            "size": len(group),
            "target_mean": target_mean,
        }

    # Label clusters relative to each other by target mean
    if all(v["target_mean"] is not None for v in summary.values()):
        means = {k: v["target_mean"] for k, v in summary.items()}
        best = min(means, key=lambda k: means[k])
        for k in summary:
            summary[k]["label"] = "high_performance" if k == best else "low_performance"

    return summary


def run_unsupervised_analysis(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Run PCA + k-means on the training feature matrix and save a YAML report.

    Args:
        features_dir: Directory containing train.parquet.
        run_id: Run identifier.
        config_dir: Pipeline config directory.
        reports_dir: Output directory for the exploration report.

    Returns:
        Dict with PCA stats, optimal k, and cluster characterizations.

    Raises:
        FileNotFoundError: If train.parquet is missing.
    """
    train_path = Path(features_dir) / run_id / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train features not found: {train_path}")

    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = pd.read_parquet(train_path)
    X = df.drop(columns=[target_col], errors="ignore").select_dtypes(include="number")

    if X.empty or X.shape[1] < 2:
        logger.warning("Insufficient numeric features for unsupervised analysis — skipping")
        return {"skipped": True, "reason": "insufficient numeric features"}

    # Standardize before PCA/k-means (features may already be scaled, but safe to re-scale)
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns
    )

    logger.info("Running PCA on %s feature matrix...", X_scaled.shape)
    pca_stats = _run_pca(X_scaled)
    logger.info(
        "PCA: %d components, PC1=%.1f%%, need %d for 80%%",
        pca_stats["n_components"],
        pca_stats["pc1_variance"] * 100,
        pca_stats["n_components_for_80pct"],
    )

    logger.info("Searching for optimal k in [2, %d]...", min(_MAX_K, X_scaled.shape[0] - 1))
    kmeans_stats = _find_optimal_k(X_scaled)
    optimal_k = kmeans_stats["optimal_k"]
    logger.info(
        "Optimal k=%d (silhouette=%.4f)",
        optimal_k, kmeans_stats["best_silhouette"],
    )

    # Fit final model with optimal k
    final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = final_km.fit_predict(X_scaled).tolist()

    cluster_info = _characterize_clusters(df, labels, target_col)

    results: dict[str, Any] = {
        "run_id": run_id,
        "pipeline_type": pipeline_config.pipeline_type,
        "n_samples": len(X_scaled),
        "n_features": X_scaled.shape[1],
        "pca": pca_stats,
        "kmeans": kmeans_stats,
        "clusters": cluster_info,
    }

    # Save report
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(reports_dir) / f"{run_id}_unsupervised.yaml"
    with open(report_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    logger.info("Unsupervised report saved: %s", report_path)

    return results
