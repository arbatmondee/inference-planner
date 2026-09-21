"""Hardware data model and provider protocol.

The abstraction is deliberately vendor-agnostic: :class:`GPUInfo` has a
``vendor`` field and a free-form ``extra`` bag rather than NVIDIA-specific
fields baked into the base schema, so an AMD/Intel/other provider can be
added later without touching this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from inference_planner.core.enums import Confidence


class GPUVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GPUInfo:
    """A single accelerator device."""

    index: int
    name: str
    vendor: GPUVendor
    memory_total_mb: float | None = None
    memory_free_mb: float | None = None
    compute_capability: str | None = None
    """Vendor-specific compute capability string (e.g. NVIDIA SM version "8.0")."""
    driver_version: str | None = None
    supported_precisions: tuple[str, ...] = field(default_factory=tuple)
    """Precisions this device is known to support, e.g. ("fp32", "fp16", "bf16")."""
    multi_processor_count: int | None = None
    uuid: str | None = None
    mig_enabled: bool | None = None
    """Whether this device is partitioned (e.g. NVIDIA MIG). ``None`` if undetected/inapplicable.
    A MIG slice is not a full peer-to-peer GPU, which matters for tensor parallelism."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Vendor-specific fields that don't belong in the common schema."""


@dataclass(frozen=True)
class CPUInfo:
    model: str | None
    physical_cores: int | None
    logical_cores: int | None
    total_memory_mb: float
    available_memory_mb: float


@dataclass(frozen=True)
class RuntimeStackInfo:
    """Versions of low-level stacks relevant to running accelerated workloads."""

    cuda_version: str | None = None
    cuda_driver_version: str | None = None
    rocm_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareInfo:
    """Full snapshot of the machine's compute resources."""

    cpu: CPUInfo
    gpus: tuple[GPUInfo, ...]
    runtime_stack: RuntimeStackInfo
    confidence: Confidence = Confidence.MEASURED
    detection_warnings: tuple[str, ...] = field(default_factory=tuple)
    """Non-fatal issues encountered during detection, e.g. "pynvml not installed"."""

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def gpu_vendors(self) -> set[GPUVendor]:
        return {gpu.vendor for gpu in self.gpus}

    @property
    def total_vram_mb(self) -> float:
        return sum(gpu.memory_total_mb or 0.0 for gpu in self.gpus)

    @property
    def total_free_vram_mb(self) -> float:
        return sum(gpu.memory_free_mb or 0.0 for gpu in self.gpus)


class HardwareProvider(ABC):
    """A plugin that can detect some slice of the machine's hardware.

    Multiple providers may run and contribute (e.g. a CPU provider and an
    NVIDIA GPU provider); the detector merges their output. A provider
    should never assume it is the only one active.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider's tooling/libraries are usable on this machine."""

    @abstractmethod
    def detect_gpus(self) -> list[GPUInfo]:
        """Return the GPUs this provider is responsible for. Empty list if none."""

    def detect_runtime_stack(self) -> RuntimeStackInfo:
        """Return low-level stack info (CUDA/ROCm versions, etc). Optional override."""
        return RuntimeStackInfo()
