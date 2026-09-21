"""Inference runtime/engine abstraction.

An engine is described entirely through :class:`RuntimeCapabilities` (data)
plus an :class:`OverheadProfile` (data) that parameterizes the shared
resource-estimation formulas in :mod:`inference_planner.resources.estimator`.
Concrete adapters (vLLM, and future SGLang/TensorRT-LLM/ONNX Runtime ones)
supply that data by introspecting the installed package; the planner and
compatibility layer never know which engine they're talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.types import OverheadProfile, ResourceEstimate


@dataclass(frozen=True)
class RuntimeIdentity:
    name: str
    installed: bool
    version: str | None = None
    detection_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What an engine (at the currently installed version) can do.

    All fields are things an adapter determines by introspecting the
    installed package (version string, feature imports), not a hardcoded
    table of "vLLM 0.x supports Y".
    """

    supported_dtypes: tuple[str, ...] = field(default_factory=tuple)
    supported_quantization_methods: tuple[str, ...] = field(default_factory=tuple)
    supports_tensor_parallel: bool = False
    supports_pipeline_parallel: bool = False
    supports_chunked_prefill: bool = False
    max_tensor_parallel_size: int | None = None
    """Hard cap the engine imposes, if any (distinct from what hardware allows)."""


class RuntimeAdapter(ABC):
    """A plugin wrapping one inference engine."""

    @abstractmethod
    def identify(self) -> RuntimeIdentity:
        """Detect whether/what version of this engine is installed."""

    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        """Describe what the installed engine version supports."""

    @abstractmethod
    def overhead_profile(self) -> OverheadProfile:
        """Engine-specific memory-overhead assumptions, fed to the shared estimator."""

    def can_run(self, model: ModelInfo, hardware: HardwareInfo) -> bool:
        """Cheap boolean gate. Prefer :mod:`compatibility` for detailed reasons."""
        caps = self.capabilities()
        if model.dtype and caps.supported_dtypes and model.dtype not in caps.supported_dtypes:
            return False
        if model.quantization.is_quantized and caps.supported_quantization_methods:
            if model.quantization.method not in caps.supported_quantization_methods:
                return False
        return True

    @abstractmethod
    def estimate_resources(
        self, model: ModelInfo, hardware: HardwareInfo, *, tensor_parallel_size: int | None = None
    ) -> ResourceEstimate:
        """Estimate memory/compute resources needed to serve ``model`` on ``hardware``.

        ``tensor_parallel_size=None`` means "recommend one automatically".
        """

    @abstractmethod
    def generate_config(
        self, model: ModelInfo, hardware: HardwareInfo, *, tensor_parallel_size: int | None = None
    ) -> dict:
        """Produce an engine-native launch configuration."""
