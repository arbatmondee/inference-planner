"""Interfaces for an eventual runtime-validation (probe/benchmark) stage.

Nothing in this module runs a model today. It exists so the distinction
between ``STATIC_ANALYSIS`` and ``RUNTIME_VALIDATION`` — and between
estimated and *measured* figures — is architected in from the start, per
the two-stage design: static analysis decides what *should* work; this
stage would eventually confirm what *actually* works by loading the model
and measuring it.
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
    """Would start a runtime with a generated config and measure its behavior.

    Not implemented by any adapter yet (see package constraints: no
    benchmarking in the initial version). Concrete implementations should
    launch the engine out-of-process, send it representative requests, and
    tear it down, returning measured values only for what they actually
    observed.
    """

    @abstractmethod
    def validate(
        self, model: ModelInfo, hardware: HardwareInfo, config: dict
    ) -> RuntimeValidationResult:
        raise NotImplementedError
