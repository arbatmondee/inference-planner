"""Top-level hardware detection entry point.

Runs every registered :class:`HardwareProvider` and merges their output.
Adding AMD/Intel/Apple support later is a matter of registering another
provider here (or externally, via :func:`register_provider`) — nothing
else in the package needs to change.
"""

from __future__ import annotations

import logging

from inference_planner.core.enums import Confidence
from inference_planner.core.registry import Registry
from inference_planner.hardware.base import (
    GPUInfo,
    HardwareInfo,
    HardwareProvider,
    RuntimeStackInfo,
)
from inference_planner.hardware.cpu import detect_cpu
from inference_planner.hardware.nvidia import NvidiaHardwareProvider

logger = logging.getLogger(__name__)

_registry: Registry[HardwareProvider] = Registry()
_registry.register(NvidiaHardwareProvider())


def register_provider(provider: HardwareProvider, *, priority: bool = False) -> None:
    """Register a hardware provider (e.g. a future AMD/Intel implementation)."""
    _registry.register(provider, priority=priority)


class HardwareDetector:
    """Detects the current machine's hardware by consulting all registered providers."""

    def __init__(self, providers: Registry[HardwareProvider] | None = None) -> None:
        self._providers = providers if providers is not None else _registry

    def detect(self) -> HardwareInfo:
        cpu = detect_cpu()
        gpus: list[GPUInfo] = []
        runtime_stack = RuntimeStackInfo()
        warnings: list[str] = []

        for provider in self._providers:
            name = type(provider).__name__
            try:
                if not provider.is_available():
                    continue
            except Exception:  # pragma: no cover - defensive
                logger.warning("%s.is_available() raised", name, exc_info=True)
                warnings.append(f"{name}: availability check failed")
                continue

            try:
                gpus.extend(provider.detect_gpus())
            except Exception:  # pragma: no cover
                logger.warning("%s.detect_gpus() raised", name, exc_info=True)
                warnings.append(f"{name}: GPU detection failed")

            try:
                stack = provider.detect_runtime_stack()
                runtime_stack = _merge_runtime_stack(runtime_stack, stack)
            except Exception:  # pragma: no cover
                logger.debug("%s.detect_runtime_stack() raised", name, exc_info=True)

        if not gpus:
            warnings.append("No GPUs detected by any registered provider; CPU-only hardware assumed.")

        return HardwareInfo(
            cpu=cpu,
            gpus=tuple(gpus),
            runtime_stack=runtime_stack,
            confidence=Confidence.MEASURED,
            detection_warnings=tuple(warnings),
        )


def _merge_runtime_stack(base: RuntimeStackInfo, new: RuntimeStackInfo) -> RuntimeStackInfo:
    return RuntimeStackInfo(
        cuda_version=base.cuda_version or new.cuda_version,
        cuda_driver_version=base.cuda_driver_version or new.cuda_driver_version,
        rocm_version=base.rocm_version or new.rocm_version,
        extra={**new.extra, **base.extra},
    )


def detect_hardware() -> HardwareInfo:
    """Convenience module-level function: ``analyzer.detect_hardware()``."""
    return HardwareDetector().detect()
