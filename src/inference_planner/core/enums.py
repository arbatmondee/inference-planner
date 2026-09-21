"""Cross-cutting enumerations shared by every analysis axis."""

from __future__ import annotations

from enum import Enum


class Confidence(str, Enum):
    """How a piece of data was obtained.

    Consumers must never treat ``ESTIMATED``/``HEURISTIC`` values as exact
    measurements. This is the mechanism the package uses to keep that
    distinction explicit end-to-end, per the "never present an estimate as
    an exact measurement" requirement.
    """

    MEASURED = "measured"
    ESTIMATED = "estimated"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


class AnalysisStage(str, Enum):
    """Which stage of the pipeline produced a result.

    ``STATIC_ANALYSIS`` only ever reads metadata (model config, hardware
    inventory). ``RUNTIME_VALIDATION`` actually loads/executes the runtime
    to measure real behavior. Only interfaces for the latter exist today.
    """

    STATIC_ANALYSIS = "static_analysis"
    RUNTIME_VALIDATION = "runtime_validation"


class Severity(str, Enum):
    """Severity of a compatibility finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
