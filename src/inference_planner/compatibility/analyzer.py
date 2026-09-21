"""Runs the registered compatibility rules and aggregates their verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field

from inference_planner.compatibility.rules import (
    DEFAULT_RULES,
    CompatibilityContext,
    CompatibilityIssue,
    Rule,
)
from inference_planner.core.enums import Severity
from inference_planner.core.registry import Registry

_registry: Registry[Rule] = Registry()
for _rule in DEFAULT_RULES:
    _registry.register(_rule)


def register_rule(rule: Rule, *, priority: bool = False) -> None:
    """Register an additional compatibility rule."""
    _registry.register(rule, priority=priority)


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[CompatibilityIssue, ...] = field(default_factory=tuple)
    errors: tuple[CompatibilityIssue, ...] = field(default_factory=tuple)


class CompatibilityAnalyzer:
    """Evaluates every registered rule against a :class:`CompatibilityContext`."""

    def __init__(self, rules: Registry[Rule] | None = None) -> None:
        self._rules = rules if rules is not None else _registry

    def analyze(self, ctx: CompatibilityContext) -> CompatibilityResult:
        reasons: list[str] = []
        warnings: list[CompatibilityIssue] = []
        errors: list[CompatibilityIssue] = []

        for rule in self._rules:
            issue = rule.check(ctx)
            if issue is None:
                message = rule.success_message(ctx)
                if message:
                    reasons.append(message)
                continue
            if issue.severity == Severity.ERROR:
                errors.append(issue)
            else:
                warnings.append(issue)

        return CompatibilityResult(
            compatible=len(errors) == 0,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )
