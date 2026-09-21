"""Runtime validator lookup by runtime name.

Separate from :mod:`inference_planner.runtimes.registry` because most
runtimes won't have a validator at all yet (Stage-2 probing is opt-in and
implemented per-engine, incrementally) — :func:`get_validator` returns
``None`` rather than raising, so callers can treat "no validator" as
"can't probe this one yet", not an error.
"""

from __future__ import annotations

from inference_planner.core.registry import Registry
from inference_planner.validation.base import RuntimeValidator

_registry: Registry[tuple[str, RuntimeValidator]] = Registry()


def register_validator(name: str, validator: RuntimeValidator, *, priority: bool = False) -> None:
    """Register a runtime validator under a runtime name (e.g. "vllm")."""
    _registry.register((name, validator), priority=priority)


def get_validator(name: str) -> RuntimeValidator | None:
    for registered_name, validator in _registry:
        if registered_name == name:
            return validator
    return None


def _register_builtin_validators() -> None:
    try:
        from inference_planner.validation.vllm_probe import VLLMSubprocessValidator
    except ImportError:  # pragma: no cover - defensive, no optional deps needed here
        return
    register_validator("vllm", VLLMSubprocessValidator())


_register_builtin_validators()
