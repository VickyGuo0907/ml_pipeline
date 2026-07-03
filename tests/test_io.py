"""Tests for shared I/O helpers."""
from pathlib import Path

from src.utils.io import find_previous_run_id


class TestFindPreviousRunId:
    """Tests for locating the run directory immediately before a given run_id."""

    def test_finds_the_most_recent_prior_run(self, tmp_path):
        for run_id in ["2026-06-01", "2026-06-15", "2026-07-01"]:
            (tmp_path / run_id).mkdir()

        assert find_previous_run_id(tmp_path, "2026-07-01") == "2026-06-15"

    def test_returns_none_when_no_prior_run_exists(self, tmp_path):
        (tmp_path / "2026-07-01").mkdir()
        assert find_previous_run_id(tmp_path, "2026-07-01") is None

    def test_returns_none_when_base_dir_missing(self, tmp_path):
        assert find_previous_run_id(tmp_path / "does_not_exist", "2026-07-01") is None

    def test_ignores_non_directory_entries(self, tmp_path):
        (tmp_path / "2026-06-01").mkdir()
        (tmp_path / "2026-06-30.txt").write_text("not a run directory")
        (tmp_path / "2026-07-01").mkdir()

        assert find_previous_run_id(tmp_path, "2026-07-01") == "2026-06-01"

    def test_current_run_itself_is_not_returned(self, tmp_path):
        (tmp_path / "2026-07-01").mkdir()
        assert find_previous_run_id(tmp_path, "2026-07-01") is None
