"""CPU/RAM detection via psutil (the one hard dependency, cross-platform)."""

from __future__ import annotations

import platform

import psutil

from inference_planner.hardware.base import CPUInfo


def detect_cpu() -> CPUInfo:
    mem = psutil.virtual_memory()
    return CPUInfo(
        model=_cpu_model_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        total_memory_mb=mem.total / (1024**2),
        available_memory_mb=mem.available / (1024**2),
    )


def _cpu_model_name() -> str | None:
    name = platform.processor()
    if name:
        return name
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None
