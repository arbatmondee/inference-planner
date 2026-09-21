"""Compatibility rules.

Each rule inspects the same generic :class:`CompatibilityContext` — it never
switches on a model or GPU name. New rules are added by registering another
:class:`Rule` instance (see :mod:`inference_planner.compatibility.analyzer`),
not by editing a big conditional.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from inference_planner.core.enums import Severity
from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.tensor_parallel import tp_compatible_with_heads
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
    device_ids: tuple[int, ...] | None = field(default=None)
    """Explicit GPU indices selected for this deployment, if any. ``None``
    means "the planner picked GPUs automatically" (currently: the first
    ``tensor_parallel_size`` detected GPUs)."""


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
    """Attention-head sharding requires the head/KV-head counts to divide evenly by TP size.

    This is a structural constraint of tensor-parallel attention (including
    the GQA/MQA replication case where KV heads < TP size), not a per-model
    special case — see :func:`inference_planner.resources.tensor_parallel.tp_compatible_with_heads`.
    """

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        tp = ctx.tensor_parallel_size
        if tp > 1 and not tp_compatible_with_heads(ctx.model.attention, tp):
            heads = ctx.model.attention.num_attention_heads
            kv_heads = ctx.model.attention.num_key_value_heads
            return CompatibilityIssue(
                code="TENSOR_PARALLEL_HEAD_MISMATCH",
                message=(
                    f"tensor_parallel_size={tp} is not compatible with "
                    f"num_attention_heads={heads}, num_key_value_heads={kv_heads}."
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


class DeviceSelectionValidRule(Rule):
    """When specific GPU indices are requested, they must actually exist and
    match the tensor-parallel degree — rather than the planner silently
    assuming devices ``0..tensor_parallel_size-1``.
    """

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        if ctx.device_ids is None:
            return None
        known_indices = {gpu.index for gpu in ctx.hardware.gpus}
        missing = [i for i in ctx.device_ids if i not in known_indices]
        if missing:
            return CompatibilityIssue(
                code="INVALID_DEVICE_ID",
                message=f"Requested device ID(s) {missing} were not found among detected GPUs {sorted(known_indices)}.",
                severity=Severity.ERROR,
            )
        if len(ctx.device_ids) != ctx.tensor_parallel_size:
            return CompatibilityIssue(
                code="DEVICE_COUNT_TP_MISMATCH",
                message=(
                    f"{len(ctx.device_ids)} device(s) were selected "
                    f"({sorted(ctx.device_ids)}) but tensor_parallel_size={ctx.tensor_parallel_size}."
                ),
                severity=Severity.ERROR,
            )
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.device_ids is not None:
            return f"Selected device(s) {sorted(ctx.device_ids)} are valid and match tensor_parallel_size"
        return None


class MIGTensorParallelRule(Rule):
    """Tensor parallelism needs full peer-to-peer GPUs; an NVIDIA MIG slice isn't one.

    Single-instance (TP=1) inference on a MIG slice is fine and not flagged.
    """

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        if ctx.tensor_parallel_size <= 1:
            return None
        selected = self._selected_gpus(ctx)
        mig_gpus = [gpu for gpu in selected if gpu.mig_enabled]
        if mig_gpus:
            return CompatibilityIssue(
                code="MIG_TENSOR_PARALLEL_UNSUPPORTED",
                message=(
                    f"tensor_parallel_size={ctx.tensor_parallel_size} was requested, but "
                    f"GPU(s) {[g.index for g in mig_gpus]} are MIG-partitioned and are not "
                    "full peer-to-peer devices."
                ),
                severity=Severity.ERROR,
            )
        return None

    @staticmethod
    def _selected_gpus(ctx: CompatibilityContext):
        if ctx.device_ids is None:
            return ctx.hardware.gpus
        return [gpu for gpu in ctx.hardware.gpus if gpu.index in ctx.device_ids]


class PythonVersionSupportedRule(Rule):
    """Checks the running Python interpreter against the runtime's declared support range."""

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        specifier_str = ctx.runtime_capabilities.required_python_specifier
        if not specifier_str:
            return None
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        try:
            if Version(current) not in SpecifierSet(specifier_str):
                return CompatibilityIssue(
                    code="UNSUPPORTED_PYTHON_VERSION",
                    message=(
                        f"Running Python {current}, but the runtime requires "
                        f"'{specifier_str}'."
                    ),
                    severity=Severity.ERROR,
                )
        except (InvalidSpecifier, InvalidVersion):
            return None  # can't confidently evaluate; don't false-positive
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.runtime_capabilities.required_python_specifier:
            return "Python version satisfies the runtime's requirement"
        return None


class TorchBuildCudaCompatibleRule(Rule):
    """The installed PyTorch build's compiled-against CUDA version can't exceed
    what the GPU driver supports, or every CUDA call will fail at runtime.
    """

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        torch_cuda = ctx.runtime_capabilities.torch_build_cuda_version
        driver_cuda = ctx.hardware.runtime_stack.cuda_version
        if not torch_cuda or not driver_cuda:
            return None
        try:
            if Version(torch_cuda) > Version(driver_cuda):
                return CompatibilityIssue(
                    code="CUDA_VERSION_MISMATCH",
                    message=(
                        f"Installed PyTorch was built for CUDA {torch_cuda}, but the "
                        f"GPU driver only supports up to CUDA {driver_cuda}."
                    ),
                    severity=Severity.ERROR,
                )
        except InvalidVersion:
            return None
        return None

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.runtime_capabilities.torch_build_cuda_version and ctx.hardware.runtime_stack.cuda_version:
            return "PyTorch's build CUDA version is supported by the GPU driver"
        return None


class ArchitectureSupportedByRuntimeRule(Rule):
    """Checks whether the *installed runtime build* actually recognizes this
    model architecture, rather than treating any declared architecture as
    automatically servable.

    A model with custom modeling code (``auto_map`` in its config) can often
    still be served via the runtime's remote/trust-code path even when its
    architecture isn't in the native registry, so that case is downgraded to
    a warning instead of an error.
    """

    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        architecture = ctx.model.architecture
        supported = ctx.runtime_capabilities.supported_architectures
        if not architecture or not supported:
            return None  # nothing declared, or the runtime's registry couldn't be introspected
        if architecture in supported:
            return None
        has_custom_code = bool(ctx.model.raw_config.get("auto_map"))
        if has_custom_code:
            return CompatibilityIssue(
                code="ARCHITECTURE_NOT_IN_REGISTRY_HAS_CUSTOM_CODE",
                message=(
                    f"Architecture '{architecture}' isn't in this runtime build's native "
                    "model registry, but the model declares custom modeling code "
                    "('auto_map') that may still work via a trust-remote-code path."
                ),
                severity=Severity.WARNING,
            )
        return CompatibilityIssue(
            code="ARCHITECTURE_NOT_SUPPORTED_BY_RUNTIME",
            message=(
                f"Architecture '{architecture}' is not recognized by this runtime "
                "build's model registry and the model has no custom modeling code."
            ),
            severity=Severity.ERROR,
        )

    def success_message(self, ctx: CompatibilityContext) -> str | None:
        if ctx.model.architecture and ctx.runtime_capabilities.supported_architectures:
            if ctx.model.architecture in ctx.runtime_capabilities.supported_architectures:
                return f"Architecture '{ctx.model.architecture}' is natively supported by this runtime build"
        return None


DEFAULT_RULES: tuple[Rule, ...] = (
    RuntimeInstalledRule(),
    ArchitectureKnownRule(),
    ArchitectureSupportedByRuntimeRule(),
    GPUAvailabilityRule(),
    DeviceSelectionValidRule(),
    MIGTensorParallelRule(),
    PythonVersionSupportedRule(),
    TorchBuildCudaCompatibleRule(),
    DtypeSupportedRule(),
    QuantizationSupportedRule(),
    PrecisionSupportedByHardwareRule(),
    TensorParallelDivisibilityRule(),
    VRAMFitRule(),
)
