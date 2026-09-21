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
class RuntimeCandidate:
    """A runtime version to evaluate, whether or not it's installed locally.

    This is what lets a caller ask "would vLLM 0.6.2 also work?" without
    that version being importable in the current process. Since we can't
    execute code from a version we don't have installed, resolving a
    candidate can only ever confirm the version *exists* and apply
    generically-derived/caller-supplied capabilities — never introspect its
    actual code the way :meth:`RuntimeAdapter.capabilities` does for the
    installed version.
    """

    name: str
    """Engine name, matching a registered adapter, e.g. "vllm"."""
    version: str
    """The version string to evaluate, e.g. "0.6.2"."""
    capabilities_override: RuntimeCapabilities | None = None
    """If the caller already knows this candidate's capabilities (e.g. from
    their own image-build manifest), supply them here so they don't have to
    be (unreliably) guessed."""
    metadata: dict = field(default_factory=dict)
    """Free-form bookkeeping, e.g. a Docker image reference this version
    corresponds to. Not interpreted by this library."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What an engine (at the currently installed version) can do.

    All fields are things an adapter determines by introspecting the
    installed package (version string, feature imports), not a hardcoded
    table of "vLLM 0.x supports Y".
    """

    supported_dtypes: tuple[str, ...] = field(default_factory=tuple)
    supported_quantization_methods: tuple[str, ...] = field(default_factory=tuple)
    supported_architectures: tuple[str, ...] = field(default_factory=tuple)
    """Model architecture class names (e.g. "LlamaForCausalLM") this engine
    build's model registry natively recognizes. Empty means "couldn't be
    introspected", not "supports nothing" — see PrecisionSupportedByHardwareRule-
    style callers, which must treat empty as unknown."""
    supports_tensor_parallel: bool = False
    supports_pipeline_parallel: bool = False
    supports_chunked_prefill: bool = False
    max_tensor_parallel_size: int | None = None
    """Hard cap the engine imposes, if any (distinct from what hardware allows)."""
    required_python_specifier: str | None = None
    """PEP 440 version specifier for the Python versions this engine build supports."""
    required_torch_specifier: str | None = None
    """PEP 440 version specifier for the PyTorch versions this engine build requires."""
    installed_torch_version: str | None = None
    torch_build_cuda_version: str | None = None
    """The CUDA version the installed PyTorch build was compiled against."""


class RuntimeAdapter(ABC):
    """A plugin wrapping one inference engine."""

    @abstractmethod
    def identify(self) -> RuntimeIdentity:
        """Detect whether/what version of this engine is installed."""

    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        """Describe what the installed engine version supports."""

    def identify_candidate(self, candidate: RuntimeCandidate) -> RuntimeIdentity:
        """Resolve a not-necessarily-installed candidate version (see :class:`RuntimeCandidate`).

        Default: unsupported. Adapters that can check version existence
        (e.g. against PyPI) should override this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support candidate evaluation")

    def capabilities_for_candidate(self, candidate: RuntimeCandidate) -> RuntimeCapabilities:
        """Best-effort capabilities for a candidate version, without executing its code.

        Default: unsupported. Adapters that override this should return
        ``candidate.capabilities_override`` verbatim when the caller
        supplied one, since that's strictly more reliable than any guess.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support candidate evaluation")

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
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        *,
        tensor_parallel_size: int | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> ResourceEstimate:
        """Estimate memory/compute resources needed to serve ``model`` on ``hardware``.

        ``tensor_parallel_size=None`` means "recommend one automatically".
        ``device_ids=None`` means "consider all detected GPUs"; when given,
        only those specific GPU indices are considered (and must exist on
        ``hardware`` — validated by :mod:`inference_planner.compatibility`,
        not here).
        """

    @abstractmethod
    def generate_config(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        *,
        tensor_parallel_size: int | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> dict:
        """Produce an engine-native launch configuration."""
