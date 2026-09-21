from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from inference_planner.core.registry import Registry
from inference_planner.hardware.base import GPUInfo, GPUVendor, HardwareProvider, RuntimeStackInfo
from inference_planner.hardware.detector import HardwareDetector
from inference_planner.hardware.nvidia import NvidiaHardwareProvider, _precisions_for_compute_capability


class _FakeProvider(HardwareProvider):
    def __init__(self, gpus: list[GPUInfo], available: bool = True):
        self._gpus = gpus
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def detect_gpus(self) -> list[GPUInfo]:
        return self._gpus

    def detect_runtime_stack(self) -> RuntimeStackInfo:
        return RuntimeStackInfo(cuda_version="12.4")


def test_detector_merges_cpu_and_gpu_info():
    fake_gpu = GPUInfo(index=0, name="Fake GPU", vendor=GPUVendor.NVIDIA, memory_total_mb=40000, memory_free_mb=39000)
    registry: Registry[HardwareProvider] = Registry()
    registry.register(_FakeProvider([fake_gpu]))

    hardware = HardwareDetector(providers=registry).detect()

    assert hardware.gpu_count == 1
    assert hardware.gpus[0].name == "Fake GPU"
    assert hardware.total_vram_mb == 40000
    assert hardware.runtime_stack.cuda_version == "12.4"
    assert hardware.cpu.total_memory_mb > 0


def test_detector_warns_when_no_gpus_found():
    registry: Registry[HardwareProvider] = Registry()
    registry.register(_FakeProvider([], available=False))

    hardware = HardwareDetector(providers=registry).detect()

    assert hardware.gpu_count == 0
    assert any("No GPUs detected" in w for w in hardware.detection_warnings)


def test_detector_survives_provider_that_raises():
    class _BrokenProvider(HardwareProvider):
        def is_available(self) -> bool:
            return True

        def detect_gpus(self) -> list[GPUInfo]:
            raise RuntimeError("boom")

    registry: Registry[HardwareProvider] = Registry()
    registry.register(_BrokenProvider())

    hardware = HardwareDetector(providers=registry).detect()

    assert hardware.gpu_count == 0
    assert any("GPU detection failed" in w for w in hardware.detection_warnings)


def test_nvidia_provider_unavailable_without_pynvml_or_smi():
    provider = NvidiaHardwareProvider()
    with patch.object(NvidiaHardwareProvider, "_has_pynvml", return_value=False), \
         patch.object(NvidiaHardwareProvider, "_has_nvidia_smi", return_value=False):
        assert provider.is_available() is False
        assert provider.detect_gpus() == []


def test_nvidia_provider_parses_smi_csv():
    provider = NvidiaHardwareProvider()
    csv_output = "0, NVIDIA A100, 81920, 81000, 550.54.15, GPU-1234, 8.0"
    fake_result = MagicMock(stdout=csv_output)
    with patch.object(NvidiaHardwareProvider, "_has_pynvml", return_value=False), \
         patch.object(NvidiaHardwareProvider, "_has_nvidia_smi", return_value=True), \
         patch("subprocess.run", return_value=fake_result):
        gpus = provider._detect_via_smi()

    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name == "NVIDIA A100"
    assert gpu.memory_total_mb == 81920
    assert gpu.compute_capability == "8.0"
    assert "bf16" in gpu.supported_precisions


def test_nvidia_provider_falls_back_when_compute_cap_query_unsupported():
    provider = NvidiaHardwareProvider()
    error = subprocess.CalledProcessError(1, "nvidia-smi")
    csv_without_cap = "0, NVIDIA T4, 16384, 16000, 470.00, GPU-abcd"

    call_count = {"n": 0}

    def fake_run(args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise error
        return MagicMock(stdout=csv_without_cap)

    with patch.object(NvidiaHardwareProvider, "_has_nvidia_smi", return_value=True), \
         patch("subprocess.run", side_effect=fake_run):
        gpus = provider._detect_via_smi()

    assert len(gpus) == 1
    assert gpus[0].compute_capability is None


def test_precisions_derived_from_compute_capability_not_gpu_name():
    assert _precisions_for_compute_capability("7.5") == ("fp32", "fp16")
    assert _precisions_for_compute_capability("8.6") == ("fp32", "fp16", "bf16")
    assert _precisions_for_compute_capability("9.0") == ("fp32", "fp16", "bf16", "fp8")
    assert _precisions_for_compute_capability(None) == ("fp32",)
