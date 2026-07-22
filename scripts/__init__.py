"""Standalone, dataset-specific prep scripts that run before the generic pipeline.

Nothing here is imported by src/ — these scripts stage landing zones and are the only
place a pipeline's specific file/year/refresh choices are allowed to live, per
CLAUDE.md's architecture rule that src/ stays dataset-agnostic.
"""
