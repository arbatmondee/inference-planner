"""vLLM runtime adapter.

Everything version-specific about the *installed* engine is discovered by
introspecting the installed ``vllm`` package (its ``__version__``, its
``EngineArgs``/``LLM`` constructor signatures, its quantization/model
registries) rather than a hardcoded per-release feature table. If vLLM
isn't installed at all, the adapter degrades to reporting
``installed=False`` with empty capabilities; it never pretends to know
what an absent engine can do.

For *candidate* versions (see :meth:`identify_candidate`/
:meth:`capabilities_for_candidate`) — versions other than whatever happens
to be installed right now — we can't execute their code, so we only ever
confirm the version exists (via PyPI) and fall back to a conservative,
long-stable baseline capability set unless the caller supplies its own
``capabilities_override``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import logging
import urllib.error
import urllib.request

from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.estimator import ResourceEstimator
from inference_planner.resources.types import OverheadProfile, ResourceEstimate
from inference_planner.runtimes.base import (
    RuntimeAdapter,
    RuntimeCandidate,
    RuntimeCapabilities,
    RuntimeIdentity,
)

logger = logging.getLogger(__name__)

# Generic dtype spellings vLLM has accepted across its history. Not tied to
# any model or GPU; refined below by inspecting the actual installation.
_BASELINE_DTYPES = ("auto", "float16", "bfloat16", "float32")

_PYPI_LOOKUP_TIMEOUT_SECONDS = 5.0


class VLLMAdapter(RuntimeAdapter):
    """Adapter for the vLLM inference engine."""

    def __init__(self) -> None:
        self._estimator = ResourceEstimator()

    def identify(self) -> RuntimeIdentity:
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            return RuntimeIdentity(
                name="vllm", installed=False,
                detection_notes=("vllm package not found",),
            )
        return RuntimeIdentity(name="vllm", installed=True, version=version)

    def capabilities(self) -> RuntimeCapabilities:
        identity = self.identify()
        if not identity.installed:
            return RuntimeCapabilities()

        engine_args_cls = self._import_engine_args()
        supports_tp = self._accepts_param(engine_args_cls, "tensor_parallel_size")
        supports_pp = self._accepts_param(engine_args_cls, "pipeline_parallel_size")
        supports_chunked_prefill = self._accepts_param(engine_args_cls, "enable_chunked_prefill")
        quant_methods = self._quantization_methods()
        architectures = self._supported_architectures()
        python_specifier, torch_specifier = self._declared_requirements()
        installed_torch_version, torch_build_cuda_version = self._installed_torch_info()

        dtypes = list(_BASELINE_DTYPES)
        if "fp8" in quant_methods and "fp8" not in dtypes:
            dtypes.append("fp8")

        return RuntimeCapabilities(
            supported_dtypes=tuple(dtypes),
            supported_quantization_methods=tuple(quant_methods),
            supported_architectures=architectures,
            supports_tensor_parallel=supports_tp,
            supports_pipeline_parallel=supports_pp,
            supports_chunked_prefill=supports_chunked_prefill,
            required_python_specifier=python_specifier,
            required_torch_specifier=torch_specifier,
            installed_torch_version=installed_torch_version,
            torch_build_cuda_version=torch_build_cuda_version,
        )

    def identify_candidate(self, candidate: RuntimeCandidate) -> RuntimeIdentity:
        installed = self.identify()
        if installed.installed and installed.version == candidate.version:
            return installed  # exact match: the real, introspected identity is strictly better

        exists = _pypi_version_exists("vllm", candidate.version)
        if exists is None:
            return RuntimeIdentity(
                name=candidate.name, installed=False, version=candidate.version,
                detection_notes=(
                    "Could not verify this version exists on PyPI (no network access, "
                    "or the lookup failed); treating as unavailable rather than guessing.",
                ),
            )
        if not exists:
            return RuntimeIdentity(
                name=candidate.name, installed=False, version=candidate.version,
                detection_notes=(f"vllm=={candidate.version} was not found on PyPI.",),
            )
        return RuntimeIdentity(
            name=candidate.name, installed=True, version=candidate.version,
            detection_notes=(
                "Confirmed to exist on PyPI. Not the currently-installed version, so its "
                "code was not introspected — see capabilities_for_candidate().",
            ),
        )

    def capabilities_for_candidate(self, candidate: RuntimeCandidate) -> RuntimeCapabilities:
        if candidate.capabilities_override is not None:
            return candidate.capabilities_override

        installed = self.identify()
        if installed.installed and installed.version == candidate.version:
            return self.capabilities()  # exact match: real introspection is strictly better

        # A version we can't execute: only claim what's been true across
        # essentially every vLLM release, never version-specific features
        # (quantization methods, architecture registry, chunked prefill...).
        return RuntimeCapabilities(
            supported_dtypes=_BASELINE_DTYPES,
            supports_tensor_parallel=True,
        )

    def overhead_profile(self) -> OverheadProfile:
        # vLLM's PagedAttention engine keeps CUDA-graph buffers, block tables,
        # and a CUDA context alive per process/GPU. These are order-of-magnitude
        # defaults describing the engine in general, refined by the runtime
        # validation stage in a future version rather than by GPU/model name.
        return OverheadProfile(
            fixed_overhead_gb=1.0,
            per_gpu_fixed_overhead_gb=0.5,
            activation_overhead_fraction=0.10,
            default_gpu_memory_utilization=0.90,
        )

    def estimate_resources(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        *,
        tensor_parallel_size: int | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> ResourceEstimate:
        return self._estimator.estimate(
            model,
            hardware,
            self.overhead_profile(),
            tensor_parallel_size=tensor_parallel_size,
            device_ids=device_ids,
        )

    def generate_config(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        *,
        tensor_parallel_size: int | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> dict:
        estimate = self.estimate_resources(
            model, hardware, tensor_parallel_size=tensor_parallel_size, device_ids=device_ids
        )
        tp = tensor_parallel_size or estimate.recommended_tensor_parallel_size

        if device_ids is not None:
            devices = list(device_ids)
        else:
            # Use the actually-detected GPU indices, not a blind range(tp) --
            # a node can have non-contiguous visible indices (e.g. under
            # CUDA_VISIBLE_DEVICES).
            devices = sorted(gpu.index for gpu in hardware.gpus)[:tp]

        max_model_len = model.context_length
        estimated_max = estimate.max_context_length_estimate.estimated_value
        if estimated_max is not None and model.context_length is not None:
            max_model_len = min(model.context_length, max(estimated_max, 1))

        dtype = model.dtype or "auto"

        return {
            "runtime": "vllm",
            "model": model.identifier,
            "devices": devices,
            "tensor_parallel_size": tp,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": self.overhead_profile().default_gpu_memory_utilization,
            "quantization": model.quantization.method,
        }

    # -- introspection helpers -------------------------------------------------

    def _import_engine_args(self):
        try:
            module = importlib.import_module("vllm.engine.arg_utils")
            return getattr(module, "EngineArgs", None)
        except Exception:  # pragma: no cover - depends on vllm internals
            logger.debug("Could not import vllm.engine.arg_utils.EngineArgs", exc_info=True)
            return None

    def _accepts_param(self, cls, param_name: str) -> bool:
        if cls is None:
            return False
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):  # pragma: no cover
            return False
        return param_name in signature.parameters

    def _quantization_methods(self) -> list[str]:
        try:
            module = importlib.import_module("vllm.model_executor.layers.quantization")
            registry = getattr(module, "QUANTIZATION_METHODS", None)
            if registry is None:
                return []
            return sorted(registry) if not isinstance(registry, dict) else sorted(registry.keys())
        except Exception:  # pragma: no cover
            logger.debug("Could not introspect vLLM quantization registry", exc_info=True)
            return []

    def _supported_architectures(self) -> tuple[str, ...]:
        """Best-effort introspection of the installed build's model registry.

        vLLM has moved/renamed this registry across versions, so several
        historical locations/attribute names are tried; an empty result
        means "couldn't introspect", not "supports nothing" (callers must
        treat it that way — see ArchitectureSupportedByRuntimeRule).
        """
        registry = None
        for module_path in (
            "vllm.model_executor.models.registry",
            "vllm.model_executor.models",
        ):
            try:
                module = importlib.import_module(module_path)
                registry = getattr(module, "ModelRegistry", None)
                if registry is not None:
                    break
            except Exception:  # pragma: no cover
                continue
        if registry is None:
            return ()

        for method_name in ("get_supported_archs", "get_supported_architectures"):
            method = getattr(registry, method_name, None)
            if callable(method):
                try:
                    return tuple(method())
                except Exception:  # pragma: no cover
                    logger.debug("ModelRegistry.%s() failed", method_name, exc_info=True)

        for attr_name in ("models", "_models"):
            models = getattr(registry, attr_name, None)
            if isinstance(models, dict):
                return tuple(models.keys())

        return ()

    def _declared_requirements(self) -> tuple[str | None, str | None]:
        """Python/PyTorch version constraints vLLM's own package metadata declares."""
        python_specifier: str | None = None
        torch_specifier: str | None = None
        try:
            metadata = importlib.metadata.metadata("vllm")
            python_specifier = metadata.get("Requires-Python") or None
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            return None, None

        try:
            from packaging.requirements import InvalidRequirement, Requirement

            for raw_requirement in importlib.metadata.requires("vllm") or []:
                try:
                    requirement = Requirement(raw_requirement)
                except InvalidRequirement:  # pragma: no cover
                    continue
                if requirement.name.lower() != "torch":
                    continue
                if requirement.marker is not None and "extra" in str(requirement.marker):
                    continue  # an optional-extra dependency, not the base requirement
                specifier = str(requirement.specifier)
                torch_specifier = specifier or None
                break
        except Exception:  # pragma: no cover
            logger.debug("Could not parse vLLM's declared torch requirement", exc_info=True)

        return python_specifier, torch_specifier

    def _installed_torch_info(self) -> tuple[str | None, str | None]:
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None

        torch_build_cuda_version = None
        try:
            torch_module = importlib.import_module("torch")
            torch_build_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        except Exception:  # pragma: no cover
            logger.debug("Could not import torch to read its build CUDA version", exc_info=True)

        return torch_version, torch_build_cuda_version


def _pypi_version_exists(package: str, version: str) -> bool | None:
    """Whether ``version`` appears in ``package``'s PyPI release list.

    Returns ``None`` (not ``False``) on any network/lookup failure, so
    callers don't mistake "couldn't check" for "confirmed absent".
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=_PYPI_LOOKUP_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    return version in data.get("releases", {})
