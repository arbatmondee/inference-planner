"""Data types shared between the resource estimator and runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from inference_planner.core.enums import Confidence


@dataclass(frozen=True)
class MemoryEstimate:
    """A memory figure that is explicitly never claimed to be exact.

    ``estimated_gb`` is ``None`` when there wasn't enough information to
    even guess, in which case ``confidence`` is ``Confidence.UNKNOWN``.
    """

    estimated_gb: float | None
    confidence: Confidence
    basis: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CountEstimate:
    estimated_value: int | None
    confidence: Confidence
    basis: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OverheadProfile:
    """Engine-specific assumptions fed into the shared estimation formulas.

    Every field here is something a runtime adapter derives from the engine
    it wraps (e.g. vLLM's PagedAttention block-table overhead), not a
    constant baked into the estimator itself. This is what lets one
    estimator serve every engine without branching on engine name.
    """

    fixed_overhead_gb: float
    """Fixed per-process overhead: CUDA context, engine bookkeeping, etc."""
    per_gpu_fixed_overhead_gb: float = 0.0
    activation_overhead_fraction: float = 0.0
    """Extra fraction of weight memory reserved for activations/workspace."""
    kv_cache_dtype_bytes: float | None = None
    """Bytes per element for the KV cache; falls back to the model dtype if unset."""
    default_gpu_memory_utilization: float = 0.90


@dataclass(frozen=True)
class ResourceEstimate:
    """Full resource-estimation result for deploying a model on given hardware."""

    weight_memory: MemoryEstimate
    kv_cache_memory_per_1k_tokens: MemoryEstimate
    """KV cache memory per 1000 tokens of context, per active sequence."""
    runtime_overhead_memory: MemoryEstimate
    total_vram_required: MemoryEstimate
    """Weights + overhead, at zero context (baseline load cost)."""
    available_vram_gb: float
    fits_in_available_vram: bool | None
    cpu_ram_required: MemoryEstimate
    max_context_length_estimate: CountEstimate
    """Approx. max context length that fits in remaining VRAM for one sequence."""
    recommended_tensor_parallel_size: int
    approximate_max_concurrency: CountEstimate
    """Approx. number of concurrent sequences at the model's full context length."""
