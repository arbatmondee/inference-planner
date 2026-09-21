"""Compatibility rules.

Each rule inspects the same generic :class:`CompatibilityContext` — it never
switches on a model or GPU name. New rules are added by registering another
:class:`Rule` instance (see :mod:`inference_planner.compatibility.analyzer`),
not by editing a big conditional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from inference_planner.core.enums import Severity
from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.types import ResourceEstimate
from inference_planner.runtimes.base import RuntimeCapabilities, RuntimeIdentity


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str
    severity: Severity


@dataclass(frozen=True)
class CompatibilityContext:
    model: ModelInfo
    hardware: HardwareInfo
    runtime_identity: RuntimeIdentity
    runtime_capabilities: RuntimeCapabilities
    resource_estimate: ResourceEstimate
    tensor_parallel_size: int


class Rule(ABC):
    """A single compatibility check.

    Returning ``None`` means the rule passed; :meth:`success_message` (if
    overridden) supplies the positive reason to surface in that case.
    """

    @abstractmethod
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None: ...

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        return None


class RuntimeInstalledRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        if not ctx.runtime_identity.installed:
            return CompatibilityIssue(
                code="RUNTIME_NOT_INSTALLED",
                message=f"Runtime '{ctx.runtime_identity.name}' is not installed in this environment.",
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        return f"Runtime '{ctx.runtime_identity.name}' {ctx.runtime_identity.version} is installed"


class DtypeSupportedRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        dtype = ctx.model.dtype
        supported = ctx.runtime_capabilities.supported_dtypes
        if dtype and supported and dtype not in supported:
            return CompatibilityIssue(
                code="UNSUPPORTED_DTYPE",
                message=(
                    f"Model dtype '{dtype}' is not among the runtime's supported "
                    f"dtypes {sorted(supported)}."
                ),
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        # Only claim verified support when the runtime was actually introspected
        # (non-empty supported_dtypes); an uninstalled runtime reports no
        # dtypes at all, which must read as "unknown", not "supported".
        if ctx.model.dtype and ctx.runtime_capabilities.supported_dtypes:
            return f"Required precision '{ctx.model.dtype}' is supported by the runtime"
        return None


class QuantizationSupportedRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        quant = ctx.model.quantization
        supported = ctx.runtime_capabilities.supported_quantization_methods
        if quant.is_quantized and supported and quant.method not in supported:
            return CompatibilityIssue(
                code="UNSUPPORTED_QUANTIZATION",
                message=(
                    f"Model quantization method '{quant.method}' is not supported by "
                    f"this runtime (supports: {sorted(supported)})."
                ),
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.model.quantization.is_quantized and ctx.runtime_capabilities.supported_quantization_methods:
            return f"Quantization method '{ctx.model.quantization.method}' is supported"
        return None


class PrecisionSupportedByHardwareRule(Rule):
    """Checks the requested dtype against each GPU's *own reported* precision support.

    Relies entirely on ``GPUInfo.supported_precisions``, which providers derive
    from device capability (e.g. compute capability), never from a GPU name.
    """

    _DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        dtype = ctx.model.dtype
        if not dtype or not ctx.hardware.gpus:
            return None
        normalized = self._DTYPE_ALIASES.get(dtype, dtype)
        unsupported = [
            gpu for gpu in ctx.hardware.gpus
            if gpu.supported_precisions and normalized not in gpu.supported_precisions
        ]
        if unsupported and len(unsupported) == len(ctx.hardware.gpus):
            names = sorted({g.name for g in unsupported})
            return CompatibilityIssue(
                code="UNSUPPORTED_PRECISION_FOR_HARDWARE",
                message=f"None of the available GPUs ({names}) report support for '{normalized}'.",
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.model.dtype and ctx.hardware.gpus:
            return "GPU compute capability supports the required precision"
        return None


class VRAMFitRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        fits = ctx.resource_estimate.fits_in_available_vram
        if fits is False:
            required = ctx.resource_estimate.total_vram_required.estimated_gb
            available = ctx.resource_estimate.available_vram_gb
            return CompatibilityIssue(
                code="INSUFFICIENT_VRAM",
                message=(
                    f"Estimated required VRAM per GPU ({required:.1f} GB) exceeds "
                    f"available free VRAM ({available:.1f} GB)."
                ),
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.resource_estimate.fits_in_available_vram is True:
            return "Model fits in available VRAM"
        return None


class GPUAvailabilityRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        if ctx.hardware.gpu_count == 0:
            return CompatibilityIssue(
                code="NO_GPU_DETECTED",
                message="No GPU was detected; most inference runtimes require at least one accelerator.",
                severity=Severity.WARNING,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.hardware.gpu_count > 0:
            return f"{ctx.hardware.gpu_count} GPU(s) available"
        return None


class TensorParallelDivisibilityRule(Rule):
    """Attention-head sharding requires the head count to divide evenly by TP size.

    This is a structural constraint of tensor-parallel attention, not a
    per-model special case.
    """

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        tp = ctx.tensor_parallel_size
        heads = ctx.model.attention.num_attention_heads
        if tp > 1 and heads and heads % tp != 0:
            return CompatibilityIssue(
                code="TENSOR_PARALLEL_HEAD_MISMATCH",
                message=(
                    f"tensor_parallel_size={tp} does not evenly divide "
                    f"num_attention_heads={heads}."
                ),
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.tensor_parallel_size > 1:
            return f"tensor_parallel_size={ctx.tensor_parallel_size} evenly divides attention heads"
        return None


class ArchitectureKnownRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        if ctx.model.architecture is None:
            return CompatibilityIssue(
                code="UNKNOWN_ARCHITECTURE",
                message="Could not determine the model's architecture from its metadata.",
                severity=Severity.WARNING,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.model.architecture is not None:
            return f"Model architecture '{ctx.model.architecture}' is declared and recognized"
        return None


DEFAULT_RULES: tuple[Rule, ...] = (
    RuntimeInstalledRule(),
    ArchitectureKnownRule(),
    GPUAvailabilityRule(),
    DtypeSupportedRule(),
    QuantizationSupportedRule(),
    PrecisionSupportedByHardwareRule(),
    TensorParallelDivisibilityRule(),
    VRAMFitRule(),
)
