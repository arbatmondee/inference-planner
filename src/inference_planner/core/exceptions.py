"""Package-wide exception hierarchy."""

from __future__ import annotations


class InferencePlannerError(Exception):
    """Base class for all errors raised by inference_planner."""


class NoCompatibleAdapterError(InferencePlannerError):
    """Raised when no registered provider/adapter can handle a given input."""


class ModelInspectionError(InferencePlannerError):
    """Raised when a model's metadata cannot be retrieved or parsed."""


class RuntimeAdapterError(InferencePlannerError):
    """Raised when a runtime adapter fails to introspect the installed engine."""


class HardwareDetectionError(InferencePlannerError):
    """Raised when hardware detection fails in an unrecoverable way."""
