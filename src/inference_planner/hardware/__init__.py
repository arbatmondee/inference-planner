"""Hardware discovery: vendor-agnostic detection of CPU, GPUs, and the accel stack."""

from inference_planner.hardware.base import (
    CPUInfo,
    GPUInfo,
    GPUVendor,
    HardwareInfo,
    HardwareProvider,
    RuntimeStackInfo,
)
from inference_planner.hardware.detector import (
    HardwareDetector,
    detect_hardware,
    register_provider,
)

__all__ = [
    "CPUInfo",
    "GPUInfo",
    "GPUVendor",
    "HardwareInfo",
    "HardwareProvider",
    "RuntimeStackInfo",
    "HardwareDetector",
    "detect_hardware",
    "register_provider",
]
