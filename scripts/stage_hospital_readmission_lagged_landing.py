"""Stage the hospital_readmission_lagged landing zone from the raw CMS quarterly dumps.

Copies the 5 predictor files from the 2024-10-30 refresh and the 1 target file from the
2025-11-26 refresh into data/hospital_readmission_lagged/landing/. Run this once before
triggering the hospital_readmission_lagged_pipeline DAG (or any time the source refresh
changes) — see data/hospitial_readmission_lagged/PIPELINE_HANDOFF.md for why these two
specific refreshes were chosen and why Unplanned_Hospital_Visits-Hospital.csv is
deliberately excluded (it carries a pneumonia readmission rate — leakage).
"""
import argparse
import shutil
from pathlib import Path
from typing import Any

# 2024 quarterly refresh used for every predictor — verified in PIPELINE_HANDOFF.md that
# the HRRP target is identical across all four 2024 refreshes, so the choice of refresh
# only matters for these predictor files, not for reproducing the target.
DEFAULT_SOURCE_2024 = Path("data/hospitial_readmission_lagged/hospitals_annual_2024/hospitals_2024-10-30")
# 2025 refresh that provides the target — one year ahead of the predictors above.
DEFAULT_SOURCE_2025 = Path("data/hospitial_readmission_lagged/hospitals_annual_2025/hospitals_2025-11-26")
DEFAULT_DEST = Path("data/hospital_readmission_lagged/landing")

PREDICTOR_FILES_2024: list[str] = [
    "HCAHPS-Hospital.csv",
    "Timely_and_Effective_Care-Hospital.csv",
    "Complications_and_Deaths-Hospital.csv",
    "Healthcare_Associated_Infections-Hospital.csv",
    "Hospital_General_Information.csv",
]
TARGET_FILE_2025: str = "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"


def stage_landing(source_2024: Path, source_2025: Path, dest: Path) -> dict[str, Any]:
    """Copy the 6 required lagged-pipeline files into the landing zone.

    Args:
        source_2024: Directory containing the 2024 quarterly refresh (predictors).
        source_2025: Directory containing the 2025 quarterly refresh (target).
        dest: Landing-zone directory to copy into (created if absent).

    Returns:
        Dict mapping each staged filename to its destination path.

    Raises:
        FileNotFoundError: If any of the 6 required source files is missing.
    """
    missing = [
        str(source_2024 / name) for name in PREDICTOR_FILES_2024 if not (source_2024 / name).exists()
    ]
    if not (source_2025 / TARGET_FILE_2025).exists():
        missing.append(str(source_2025 / TARGET_FILE_2025))
    if missing:
        raise FileNotFoundError(f"Missing required source file(s): {missing}")

    dest.mkdir(parents=True, exist_ok=True)

    staged: dict[str, Any] = {}
    for name in PREDICTOR_FILES_2024:
        dest_path = dest / name
        shutil.copy2(source_2024 / name, dest_path)
        staged[name] = dest_path

    dest_path = dest / TARGET_FILE_2025
    shutil.copy2(source_2025 / TARGET_FILE_2025, dest_path)
    staged[TARGET_FILE_2025] = dest_path

    extra = [
        p.name for p in dest.iterdir()
        if p.name not in staged and not p.name.startswith(".")
    ]
    if extra:
        print(
            f"Warning: {dest} contains {len(extra)} file(s) not managed by this script: {extra}. "
            "Landing zone must contain only the 6 staged files so file_pattern matches stay unambiguous."
        )

    return staged


def main() -> None:
    """CLI entry point: stage the landing zone from the default (or overridden) paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-2024", type=Path, default=DEFAULT_SOURCE_2024)
    parser.add_argument("--source-2025", type=Path, default=DEFAULT_SOURCE_2025)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    staged = stage_landing(args.source_2024, args.source_2025, args.dest)
    print(f"Staged {len(staged)} file(s) into {args.dest}:")
    for name, path in staged.items():
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
