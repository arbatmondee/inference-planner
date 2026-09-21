"""NVIDIA GPU detection.

Tries, in order of preference:

1. ``pynvml`` (the ``nvidia-ml-py`` package) — richest, structured data.
2. The ``nvidia-smi`` CLI via a CSV query — works with no Python bindings
   installed, which is common on inference hosts.

If neither is available, :meth:`is_available` returns ``False`` and the
provider contributes nothing; it never raises for "no NVIDIA GPU present".
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from inference_planner.hardware.base import (
    GPUInfo,
    GPUVendor,
    HardwareProvider,
    RuntimeStackInfo,
)

logger = logging.getLogger(__name__)

_SMI_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.free",
    "driver_version",
    "uuid",
    "compute_cap",
]


class NvidiaHardwareProvider(HardwareProvider):
    """Detects NVIDIA GPUs via pynvml, falling back to nvidia-smi."""

    def is_available(self) -> bool:
        return self._has_pynvml() or self._has_nvidia_smi()

    def detect_gpus(self) -> list[GPUInfo]:
        if self._has_pynvml():
            try:
                return self._detect_via_pynvml()
            except Exception:  # pragma: no cover - defensive, driver quirks vary
                logger.warning("pynvml detection failed, falling back to nvidia-smi", exc_info=True)
        if self._has_nvidia_smi():
            try:
                return self._detect_via_smi()
            except Exception:  # pragma: no cover
                logger.warning("nvidia-smi detection failed", exc_info=True)
        return []

    def detect_runtime_stack(self) -> RuntimeStackInfo:
        cuda_version = None
        driver_version = None
        if self._has_pynvml():
            try:
                import pynvml

                pynvml.nvmlInit()
                try:
                    raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                    cuda_version = f"{raw // 1000}.{(raw % 1000) // 10}"
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode()
                finally:
                    pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                logger.debug("Could not read CUDA/driver version via pynvml", exc_info=True)
        if cuda_version is None and self._has_nvidia_smi():
            cuda_version, driver_version = self._cuda_driver_from_smi()
        return RuntimeStackInfo(cuda_version=cuda_version, cuda_driver_version=driver_version)

    # -- pynvml backend ----------------------------------------------------

    @staticmethod
    def _has_pynvml() -> bool:
        try:
            import pynvml  # noqa: F401
        except ImportError:
            return False
        return True

    def _detect_via_pynvml(self) -> list[GPUInfo]:
        import pynvml

        gpus: list[GPUInfo] = []
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except Exception:
                    compute_capability = None
                try:
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    if isinstance(uuid, bytes):
                        uuid = uuid.decode()
                except Exception:
                    uuid = None
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode()
                except Exception:
                    driver_version = None
                try:
                    mp_count = pynvml.nvmlDeviceGetNumGpuCores(handle)
                except Exception:
                    mp_count = None

                gpus.append(
                    GPUInfo(
                        index=i,
                        name=name,
                        vendor=GPUVendor.NVIDIA,
                        memory_total_mb=mem.total / (1024**2),
                        memory_free_mb=mem.free / (1024**2),
                        compute_capability=compute_capability,
                        driver_version=driver_version,
                        multi_processor_count=mp_count,
                        uuid=uuid,
                        supported_precisions=_precisions_for_compute_capability(compute_capability),
                    )
                )
        finally:
            pynvml.nvmlShutdown()
        return gpus

    # -- nvidia-smi backend --------------------------------------------------

    @staticmethod
    def _has_nvidia_smi() -> bool:
        return shutil.which("nvidia-smi") is not None

    def _run_smi(self, args: list[str]) -> str:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()

    def _detect_via_smi(self) -> list[GPUInfo]:
        query = ",".join(_SMI_FIELDS)
        try:
            output = self._run_smi([f"--query-gpu={query}", "--format=csv,noheader,nounits"])
        except subprocess.CalledProcessError:
            # Older nvidia-smi builds don't support "compute_cap"; retry without it.
            fields = [f for f in _SMI_FIELDS if f != "compute_cap"]
            query = ",".join(fields)
            output = self._run_smi([f"--query-gpu={query}", "--format=csv,noheader,nounits"])
            return self._parse_smi_rows(output, fields)
        return self._parse_smi_rows(output, _SMI_FIELDS)

    def _parse_smi_rows(self, output: str, fields: list[str]) -> list[GPUInfo]:
        gpus: list[GPUInfo] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(",")]
            row = dict(zip(fields, values))
            compute_cap = row.get("compute_cap")
            gpus.append(
                GPUInfo(
                    index=int(row["index"]),
                    name=row["name"],
                    vendor=GPUVendor.NVIDIA,
                    memory_total_mb=_safe_float(row.get("memory.total")),
                    memory_free_mb=_safe_float(row.get("memory.free")),
                    compute_capability=compute_cap if compute_cap and compute_cap != "N/A" else None,
                    driver_version=row.get("driver_version"),
                    uuid=row.get("uuid"),
                    supported_precisions=_precisions_for_compute_capability(compute_cap),
                )
            )
        return gpus

    def _cuda_driver_from_smi(self) -> tuple[str | None, str | None]:
        try:
            output = self._run_smi(["--query-gpu=driver_version", "--format=csv,noheader"])
            driver_version = output.splitlines()[0].strip() if output else None
        except Exception:  # pragma: no cover
            driver_version = None
        cuda_version = self._cuda_version_from_smi_header()
        return cuda_version, driver_version

    def _cuda_version_from_smi_header(self) -> str | None:
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True
            )
        except Exception:  # pragma: no cover
            return None
        for line in result.stdout.splitlines():
            if "CUDA Version" in line:
                # Example: "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4  |"
                try:
                    return line.split("CUDA Version:")[1].strip().split()[0]
                except IndexError:  # pragma: no cover
                    return None
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _precisions_for_compute_capability(compute_capability: str | None) -> tuple[str, ...]:
    """Derive supported float precisions from compute capability, not GPU name.

    This is architecture-generation logic (what a given SM version supports),
    not a per-model GPU name lookup table.
    """
    if not compute_capability:
        return ("fp32",)
    try:
        major = int(compute_capability.split(".")[0])
    except (ValueError, IndexError):
        return ("fp32",)
    precisions = ["fp32", "fp16"]
    if major >= 8:
        precisions.append("bf16")
    if major >= 9:
        precisions.append("fp8")
    return tuple(precisions)
