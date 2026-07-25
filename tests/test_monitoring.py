"""Tests for drift detection helpers."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring import _align_columns, compute_drift_detected, generate_drift_report


def _make_df(n: int, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "a": [i + offset for i in range(n)],
        "b": [(i % 5) + offset for i in range(n)],
    })


class TestAlignColumns:
    """Tests for the column-intersection fix (two runs' feature sets can legitimately differ)."""

    def test_restricts_to_common_columns(self):
        reference = pd.DataFrame({"a": [1, 2], "b": [3, 4], "only_in_reference": [5, 6]})
        current = pd.DataFrame({"a": [1, 2], "b": [3, 4], "only_in_current": [7, 8]})

        aligned_ref, aligned_cur = _align_columns(reference, current)

        assert list(aligned_ref.columns) == ["a", "b"]
        assert list(aligned_cur.columns) == ["a", "b"]

    def test_no_mismatch_leaves_columns_untouched(self):
        reference = _make_df(10)
        current = _make_df(10, offset=1.0)

        aligned_ref, aligned_cur = _align_columns(reference, current)

        assert list(aligned_ref.columns) == ["a", "b"]
        assert list(aligned_cur.columns) == ["a", "b"]


class TestComputeDriftDetected:
    """Tests for the standalone boolean drift check."""

    def test_returns_none_when_evidently_unavailable(self):
        with patch("src.monitoring.Report", None), patch("src.monitoring.DatasetDriftMetric", None):
            result = compute_drift_detected(_make_df(50), _make_df(50))
        assert result is None

    def test_returns_boolean_when_evidently_available(self):
        reference = _make_df(200)
        current = _make_df(200)
        result = compute_drift_detected(reference, current)
        assert result in (True, False)

    def test_returns_none_on_evidently_failure(self):
        with patch("src.monitoring.Report") as mock_report_cls:
            mock_report_cls.side_effect = RuntimeError("boom")
            result = compute_drift_detected(_make_df(10), _make_df(10))
        assert result is None


class TestGenerateDriftReport:
    """Tests for the full HTML-report-producing path (previously untested)."""

    def test_creates_baseline_report_when_no_previous_run(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-01"
        run_dir.mkdir(parents=True)
        _make_df(50).to_parquet(run_dir / "train.parquet")

        result = generate_drift_report(
            features_dir=features_dir,
            run_id="2026-07-01",
            previous_run_id=None,
            reports_dir=tmp_path / "reports",
        )

        assert result["type"] == "baseline"
        assert result["run_id"] == "2026-07-01"

    def test_compares_against_previous_run_when_available(self, tmp_path):
        features_dir = tmp_path / "features"
        for run_id, offset in [("2026-06-01", 0.0), ("2026-07-01", 0.0)]:
            run_dir = features_dir / run_id
            run_dir.mkdir(parents=True)
            _make_df(200, offset=offset).to_parquet(run_dir / "train.parquet")

        result = generate_drift_report(
            features_dir=features_dir,
            run_id="2026-07-01",
            previous_run_id="2026-06-01",
            reports_dir=tmp_path / "reports",
        )

        assert result["comparison_run_id"] == "2026-06-01"
        assert "drift_detected" in result

    def test_does_not_crash_when_runs_have_different_columns(self, tmp_path):
        """Regression test: two real runs produced different NZV-surviving columns
        (e.g. one had 'Emergency Services', the other didn't), which previously
        crashed this task with an unhandled KeyError deep inside Evidently."""
        features_dir = tmp_path / "features"

        previous_dir = features_dir / "2026-06-01"
        previous_dir.mkdir(parents=True)
        previous_df = _make_df(200)
        previous_df["Emergency Services"] = [i % 2 for i in range(200)]
        previous_df.to_parquet(previous_dir / "train.parquet")

        current_dir = features_dir / "2026-07-01"
        current_dir.mkdir(parents=True)
        _make_df(200, offset=1.0).to_parquet(current_dir / "train.parquet")  # no "Emergency Services"

        result = generate_drift_report(
            features_dir=features_dir,
            run_id="2026-07-01",
            previous_run_id="2026-06-01",
            reports_dir=tmp_path / "reports",
        )

        assert result["comparison_run_id"] == "2026-06-01"
        assert "warning" not in result or "drift_detected" in result

    def test_raises_when_current_run_data_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_drift_report(
                features_dir=tmp_path / "features",
                run_id="2026-07-01",
                previous_run_id=None,
                reports_dir=tmp_path / "reports",
            )
