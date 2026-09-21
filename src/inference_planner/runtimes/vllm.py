"""vLLM runtime adapter.

Everything version-specific is discovered by introspecting the installed
``vllm`` package (its ``__version__``, its ``EngineArgs``/``LLM``
constructor signatures, its quantization-method registry) rather than
hardcoded per-release feature tables. If vLLM isn't installed at all, the
adapter degrades to reporting ``installed=False`` with empty capabilities;
it never pretends to know what an absent engine can do.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import logging

from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.estimator import ResourceEstimator
from inference_planner.resources.types import OverheadProfile, ResourceEstimate
from inference_planner.runtimes.base import RuntimeAdapter, RuntimeCapabilities, RuntimeIdentity

logger = logging.getLogger(__name__)

# Generic dtype spellings vLLM has accepted across its history. Not tied to
# any model or GPU; refined below by inspecting the actual installation.
_BASELINE_DTYPES = ("auto", "float16", "bfloat16", "float32")


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

        dtypes = list(_BASELINE_DTYPES)
        if "fp8" in quant_methods and "fp8" not in dtypes:
            dtypes.append("fp8")

        return RuntimeCapabilities(
            supported_dtypes=tuple(dtypes),
            supported_quantization_methods=tuple(quant_methods),
            supports_tensor_parallel=supports_tp,
            supports_pipeline_parallel=supports_pp,
            supports_chunked_prefill=supports_chunked_prefill,
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
        self, model: ModelInfo, hardware: HardwareInfo, *, tensor_parallel_size: int | None = None
    ) -> ResourceEstimate:
        return self._estimator.estimate(
            model, hardware, self.overhead_profile(), tensor_parallel_size=tensor_parallel_size
        )

    def generate_config(
        self, model: ModelInfo, hardware: HardwareInfo, *, tensor_parallel_size: int | None = None
    ) -> dict:
        estimate = self.estimate_resources(
            model, hardware, tensor_parallel_size=tensor_parallel_size
        )
        tp = tensor_parallel_size or estimate.recommended_tensor_parallel_size
        devices = list(range(tp)) if hardware.gpus else []

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
