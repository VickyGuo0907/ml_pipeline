"""Tests for the hospital_readmission_lagged landing-zone staging script."""
import tempfile
from pathlib import Path

import pytest

from scripts.stage_hospital_readmission_lagged_landing import (
    PREDICTOR_FILES_2024,
    TARGET_FILE_2025,
    stage_landing,
)


def _touch(path: Path, content: str = "Facility ID,Value\n1,1\n") -> None:
    """Write a minimal CSV so copy2 has real bytes to move."""
    path.write_text(content)


class TestStageLanding:
    """Tests for stage_landing copying the 6 required files into the landing zone."""

    def test_stage_landing_copies_all_six_files(self):
        """All 5 predictor files (2024) and the 1 target file (2025) land in dest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_2024 = tmp / "hospitals_2024-10-30"
            source_2025 = tmp / "hospitals_2025-11-26"
            dest = tmp / "landing"
            source_2024.mkdir()
            source_2025.mkdir()

            for name in PREDICTOR_FILES_2024:
                _touch(source_2024 / name)
            _touch(source_2025 / TARGET_FILE_2025)

            result = stage_landing(source_2024, source_2025, dest)

            assert len(result) == 6
            for name in [*PREDICTOR_FILES_2024, TARGET_FILE_2025]:
                assert (dest / name).exists()

    def test_stage_landing_raises_on_missing_source_file(self):
        """A missing required file raises FileNotFoundError naming the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_2024 = tmp / "hospitals_2024-10-30"
            source_2025 = tmp / "hospitals_2025-11-26"
            dest = tmp / "landing"
            source_2024.mkdir()
            source_2025.mkdir()

            # Omit one predictor file
            for name in PREDICTOR_FILES_2024[1:]:
                _touch(source_2024 / name)
            _touch(source_2025 / TARGET_FILE_2025)

            with pytest.raises(FileNotFoundError, match=PREDICTOR_FILES_2024[0]):
                stage_landing(source_2024, source_2025, dest)
