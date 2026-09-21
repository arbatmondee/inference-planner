"""Stage-2 runtime validation interfaces.

Static analysis (everything else in this package) only ever reads
metadata. This package defines the shape of an eventual "actually start
the runtime and measure it" stage, without implementing it yet — see
:class:`RuntimeValidator`.
"""

from inference_planner.validation.base import RuntimeValidationResult, RuntimeValidator

__all__ = ["RuntimeValidationResult", "RuntimeValidator"]
