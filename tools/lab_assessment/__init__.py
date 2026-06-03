"""Lab assessment suite: reusable per-dimension metric computers + report.

Read-only. Never mutates lab state. See
docs/superpowers/specs/2026-06-02-lab-assessment-design.md.
"""
from tools.lab_assessment.types import DimensionResult, VERDICT_LEVELS

__all__ = ["DimensionResult", "VERDICT_LEVELS"]
