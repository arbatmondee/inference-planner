"""Compatibility analysis: model + runtime + hardware -> structured verdict."""

from inference_planner.compatibility.analyzer import (
    CompatibilityAnalyzer,
    CompatibilityResult,
    register_rule,
)
from inference_planner.compatibility.rules import CompatibilityContext, CompatibilityIssue, Rule

__all__ = [
    "CompatibilityAnalyzer",
    "CompatibilityResult",
    "register_rule",
    "CompatibilityContext",
    "CompatibilityIssue",
    "Rule",
]
