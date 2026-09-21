"""Generic resource estimation.

Every formula here operates on model/hardware *facts* (parameter count,
layer/head dims, dtype, free VRAM) plus an engine-supplied
:class:`~inference_planner.resources.types.OverheadProfile`. There is no
per-model or per-GPU branching: a new model family or a new GPU just
changes the numbers fed in, not the formulas themselves.

All outputs are wrapped in :class:`MemoryEstimate`/:class:`CountEstimate`
so callers can see the confidence and the basis of every figure.
"""

from __future__ import annotations

from inference_planner.core.dtype import bytes_per_element, bytes_per_element_for_quantization
from inference_planner.core.enums import Confidence
from inference_planner.hardware.base import GPUInfo, HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.tensor_parallel import tp_compatible_with_heads
from inference_planner.resources.types import (
    CountEstimate,
    MemoryEstimate,
    OverheadProfile,
    ResourceEstimate,
)

_BYTES_PER_GB = 1024**3
_UNKNOWN_MEMORY = MemoryEstimate(estimated_gb=None, confidence=Confidence.UNKNOWN)
_UNKNOWN_COUNT = CountEstimate(estimated_value=None, confidence=Confidence.UNKNOWN)


class ResourceEstimator:
    """Estimates memory/concurrency needs for serving a model on given hardware."""

    def estimate(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        overhead_profile: OverheadProfile,
        *,
        tensor_parallel_size: int | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> ResourceEstimate:
        weight_bytes_per_param = self._weight_bytes_per_param(model)
        available_vram_gb = self._min_gpu_free_gb(hardware, device_ids)

        if tensor_parallel_size is None:
            tensor_parallel_size = self._recommend_tensor_parallel_size(
                model, hardware, overhead_profile, weight_bytes_per_param, device_ids
            )

        weight_memory = self._weight_memory(model, weight_bytes_per_param, tensor_parallel_size)
        runtime_overhead = self._runtime_overhead(weight_memory, overhead_profile)
        total_vram_required = self._sum_memory(
            weight_memory, runtime_overhead, basis=("weight memory", "runtime overhead")
        )

        kv_cache_per_1k = self._kv_cache_per_1k_tokens(model, overhead_profile, tensor_parallel_size)

        fits: bool | None = None
        if total_vram_required.estimated_gb is not None:
            fits = total_vram_required.estimated_gb <= available_vram_gb

        usable_gb = available_vram_gb * overhead_profile.default_gpu_memory_utilization
        max_context = self._max_context_length(
            model, total_vram_required, kv_cache_per_1k, usable_gb
        )
        max_concurrency = self._max_concurrency(
            model, total_vram_required, kv_cache_per_1k, usable_gb
        )

        return ResourceEstimate(
            weight_memory=weight_memory,
            kv_cache_memory_per_1k_tokens=kv_cache_per_1k,
            runtime_overhead_memory=runtime_overhead,
            total_vram_required=total_vram_required,
            available_vram_gb=available_vram_gb,
            fits_in_available_vram=fits,
            cpu_ram_required=self._cpu_ram_required(weight_memory),
            max_context_length_estimate=max_context,
            recommended_tensor_parallel_size=tensor_parallel_size,
            approximate_max_concurrency=max_concurrency,
        )

    # -- component estimates --------------------------------------------------

    def _weight_bytes_per_param(self, model: ModelInfo) -> float | None:
        if model.quantization.is_quantized and model.quantization.bits:
            return bytes_per_element_for_quantization(model.quantization.bits)
        return bytes_per_element(model.dtype)

    def _weight_memory(
        self, model: ModelInfo, bytes_per_param: float | None, tensor_parallel_size: int
    ) -> MemoryEstimate:
        count = model.parameter_count.count
        if count is None or bytes_per_param is None:
            return _UNKNOWN_MEMORY
        total_gb = (count * bytes_per_param) / _BYTES_PER_GB
        per_gpu_gb = total_gb / max(tensor_parallel_size, 1)
        basis = (*model.parameter_count.basis, f"dtype/quant bytes-per-param={bytes_per_param}")
        if tensor_parallel_size > 1:
            basis = (*basis, f"sharded across tensor_parallel_size={tensor_parallel_size}")
        # Weight memory can only ever be as confident as the parameter count it's derived from.
        confidence = model.parameter_count.confidence
        return MemoryEstimate(estimated_gb=per_gpu_gb, confidence=confidence, basis=basis)

    def _runtime_overhead(
        self, weight_memory: MemoryEstimate, profile: OverheadProfile
    ) -> MemoryEstimate:
        base = profile.fixed_overhead_gb + profile.per_gpu_fixed_overhead_gb
        basis = ["engine fixed overhead"]
        if weight_memory.estimated_gb is not None and profile.activation_overhead_fraction:
            base += weight_memory.estimated_gb * profile.activation_overhead_fraction
            basis.append("activation/workspace overhead fraction of weight memory")
        return MemoryEstimate(estimated_gb=base, confidence=Confidence.HEURISTIC, basis=tuple(basis))

    def _kv_cache_per_1k_tokens(
        self, model: ModelInfo, profile: OverheadProfile, tensor_parallel_size: int
    ) -> MemoryEstimate:
        num_layers = model.num_layers
        num_kv_heads = model.attention.num_key_value_heads
        head_dim = model.attention.head_dim
        if not (num_layers and num_kv_heads and head_dim):
            return _UNKNOWN_MEMORY
        kv_dtype_bytes = profile.kv_cache_dtype_bytes or bytes_per_element(model.dtype) or 2.0
        # 2x for K and V, one cache slot per layer per KV head per head_dim element.
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
        bytes_per_token /= max(tensor_parallel_size, 1)  # KV heads are also sharded across TP
        gb_per_1k = (bytes_per_token * 1000) / _BYTES_PER_GB
        return MemoryEstimate(
            estimated_gb=gb_per_1k,
            confidence=Confidence.ESTIMATED,
            basis=("num_layers", "num_key_value_heads", "head_dim", f"kv_dtype_bytes={kv_dtype_bytes}"),
        )

    def _cpu_ram_required(self, weight_memory: MemoryEstimate) -> MemoryEstimate:
        if weight_memory.estimated_gb is None:
            return _UNKNOWN_MEMORY
        # Heuristic: loading/staging weights on CPU before transfer to GPU
        # typically needs on the order of one full copy of the weights.
        return MemoryEstimate(
            estimated_gb=weight_memory.estimated_gb,
            confidence=Confidence.HEURISTIC,
            basis=("assumes one CPU-side staging copy of model weights during load",),
        )

    def _max_context_length(
        self,
        model: ModelInfo,
        total_vram_required: MemoryEstimate,
        kv_cache_per_1k: MemoryEstimate,
        usable_gb: float,
    ) -> CountEstimate:
        if total_vram_required.estimated_gb is None or kv_cache_per_1k.estimated_gb is None:
            return _UNKNOWN_COUNT
        if kv_cache_per_1k.estimated_gb <= 0:
            return _UNKNOWN_COUNT
        remaining_gb = usable_gb - total_vram_required.estimated_gb
        if remaining_gb <= 0:
            return CountEstimate(estimated_value=0, confidence=Confidence.HEURISTIC,
                                  basis=("no VRAM remains after weights + overhead",))
        tokens = int((remaining_gb / kv_cache_per_1k.estimated_gb) * 1000)
        basis = ["available VRAM minus weights/overhead", "KV cache bytes per token"]
        if model.context_length is not None:
            tokens = min(tokens, model.context_length)
            basis.append("capped at model's trained context_length")
        return CountEstimate(estimated_value=tokens, confidence=Confidence.HEURISTIC, basis=tuple(basis))

    def _max_concurrency(
        self,
        model: ModelInfo,
        total_vram_required: MemoryEstimate,
        kv_cache_per_1k: MemoryEstimate,
        usable_gb: float,
    ) -> CountEstimate:
        if (
            total_vram_required.estimated_gb is None
            or kv_cache_per_1k.estimated_gb is None
            or model.context_length is None
            or kv_cache_per_1k.estimated_gb <= 0
        ):
            return _UNKNOWN_COUNT
        remaining_gb = usable_gb - total_vram_required.estimated_gb
        if remaining_gb <= 0:
            return CountEstimate(estimated_value=0, confidence=Confidence.HEURISTIC,
                                  basis=("no VRAM remains after weights + overhead",))
        per_sequence_gb = kv_cache_per_1k.estimated_gb * (model.context_length / 1000)
        if per_sequence_gb <= 0:
            return _UNKNOWN_COUNT
        concurrency = max(int(remaining_gb / per_sequence_gb), 0)
        return CountEstimate(
            estimated_value=concurrency,
            confidence=Confidence.HEURISTIC,
            basis=("assumes every concurrent sequence uses the full trained context_length",),
        )

    def _recommend_tensor_parallel_size(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        profile: OverheadProfile,
        bytes_per_param: float | None,
        device_ids: tuple[int, ...] | None = None,
    ) -> int:
        usable_gpu_count = len(device_ids) if device_ids is not None else hardware.gpu_count
        if usable_gpu_count <= 1:
            return max(usable_gpu_count, 1)

        # Only ever recommend a TP degree that could structurally work — searching
        # divisors of the *selected* GPU count, not the whole node's, and skipping
        # any degree that can't evenly shard this model's attention heads.
        candidates = [
            tp for tp in _divisors_ascending(usable_gpu_count)
            if tp_compatible_with_heads(model.attention, tp)
        ]
        if not candidates:
            return 1  # no divisor of the selection is structurally valid; don't guess

        count = model.parameter_count.count
        if count is None or bytes_per_param is None:
            return 1

        available_gb = self._min_gpu_free_gb(hardware, device_ids)
        usable_gb = available_gb * profile.default_gpu_memory_utilization
        for tp in candidates:
            weight_gb = (count * bytes_per_param) / _BYTES_PER_GB / tp
            overhead_gb = profile.fixed_overhead_gb + profile.per_gpu_fixed_overhead_gb
            overhead_gb += weight_gb * profile.activation_overhead_fraction
            if weight_gb + overhead_gb <= usable_gb:
                return tp
        return candidates[-1]  # nothing fits; report the largest structurally-valid degree

    def _selected_gpus(self, hardware: HardwareInfo, device_ids: tuple[int, ...] | None) -> list[GPUInfo]:
        if device_ids is None:
            return list(hardware.gpus)
        by_index = {gpu.index: gpu for gpu in hardware.gpus}
        return [by_index[i] for i in device_ids if i in by_index]

    def _min_gpu_free_gb(self, hardware: HardwareInfo, device_ids: tuple[int, ...] | None = None) -> float:
        gpus = self._selected_gpus(hardware, device_ids)
        if not gpus:
            return 0.0
        free_values = [gpu.memory_free_mb for gpu in gpus if gpu.memory_free_mb is not None]
        if not free_values:
            return 0.0
        return min(free_values) / 1024

    def _sum_memory(self, *estimates: MemoryEstimate, basis: tuple[str, ...]) -> MemoryEstimate:
        if any(e.estimated_gb is None for e in estimates):
            return _UNKNOWN_MEMORY
        total = sum(e.estimated_gb for e in estimates)
        confidences = [e.confidence for e in estimates]
        # The combined figure is only as confident as its weakest input.
        confidence = _weakest_confidence(confidences)
        return MemoryEstimate(estimated_gb=total, confidence=confidence, basis=basis)


_CONFIDENCE_RANK = {
    Confidence.MEASURED: 3,
    Confidence.ESTIMATED: 2,
    Confidence.HEURISTIC: 1,
    Confidence.UNKNOWN: 0,
}


def _weakest_confidence(confidences: list[Confidence]) -> Confidence:
    return min(confidences, key=lambda c: _CONFIDENCE_RANK[c])


def _divisors_ascending(n: int) -> list[int]:
    return sorted(d for d in range(1, n + 1) if n % d == 0)
