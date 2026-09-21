"""Interfaces for the runtime-validation (probe/benchmark) stage.

This is what lets the distinction between ``STATIC_ANALYSIS`` and
``RUNTIME_VALIDATION`` — and between estimated and *measured* figures —
be architected in cleanly: static analysis decides what *should* work;
this stage actually confirms what *works*, by loading the model and
measuring it. See :mod:`inference_planner.validation.vllm_probe` for the
concrete, opt-in vLLM implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from inference_planner.core.enums import AnalysisStage, Confidence
from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo


@dataclass(frozen=True)
class RuntimeValidationResult:
    """Measured (not estimated) figures from actually running the engine.

    Every numeric field is ``None`` until a real probe fills it in; the
    ``confidence`` is always ``Confidence.MEASURED`` for fields that were
    actually observed, distinguishing this from the static estimates in
    :class:`~inference_planner.resources.types.ResourceEstimate`.
    """

    stage: AnalysisStage
    confidence: Confidence
    model_load_time_seconds: float | None = None
    actual_vram_usage_gb: float | None = None
    time_to_first_token_ms: float | None = None
    tokens_per_second: float | None = None
    max_observed_concurrency: int | None = None
    notes: tuple[str, ...] = ()


class RuntimeValidator(ABC):
    """Starts a runtime with a generated config and measures its real behavior.

    This is real, expensive, and opt-in: it downloads/loads actual model
    weights, consumes GPU memory, and can take minutes. Implementations
    should launch the engine out-of-process (never in the caller's own
    process — a CUDA OOM or crash during a probe must not take the caller
    down with it), and return measured values only for what they actually
    observed, leaving the rest ``None``.
    """

    @abstractmethod
    def validate(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        config: dict,
        *,
        timeout_seconds: int = 600,
    ) -> RuntimeValidationResult:
        raise NotImplementedError
