"""The final structured report handed back to the calling inference platform."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from inference_planner.compatibility.analyzer import CompatibilityResult
from inference_planner.core.enums import AnalysisStage
from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.resources.types import ResourceEstimate
from inference_planner.runtimes.base import RuntimeIdentity
from inference_planner.validation.base import RuntimeValidationResult


@dataclass(frozen=True)
class AnalysisReport:
    """Everything the platform needs to decide whether/how to deploy a model."""

    compatible: bool
    stage: AnalysisStage
    hardware: HardwareInfo
    model: ModelInfo
    runtime: RuntimeIdentity
    compatibility: CompatibilityResult
    resource_estimate: ResourceEstimate
    deployment_plan: dict[str, Any] | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    runtime_validation: RuntimeValidationResult | None = None
    """Populated only once a Stage-2 probe has actually run (see :mod:`inference_planner.validation`)."""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:  # human-readable CLI rendering
        return render_human_readable(self)


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def render_human_readable(report: AnalysisReport) -> str:
    lines: list[str] = []
    lines.append("Inference Planner")
    lines.append("-" * 32)

    hw = report.hardware
    lines.append("")
    lines.append("Hardware")
    if hw.gpus:
        for gpu in hw.gpus:
            vram = f"{gpu.memory_total_mb / 1024:.0f} GB" if gpu.memory_total_mb else "unknown VRAM"
            lines.append(f"  GPU {gpu.index}: {gpu.name} ({vram})")
        lines.append(f"  GPUs: {hw.gpu_count}")
        lines.append(f"  Total VRAM: {hw.total_vram_mb / 1024:.1f} GB")
    else:
        lines.append("  GPU: none detected")
    if hw.runtime_stack.cuda_version:
        lines.append(f"  CUDA: {hw.runtime_stack.cuda_version}")

    model = report.model
    lines.append("")
    lines.append("Model")
    lines.append(f"  Name: {model.identifier}")
    if model.parameter_count.count:
        lines.append(f"  Parameters: ~{model.parameter_count.count / 1e9:.1f}B ({model.parameter_count.confidence.value})")
    lines.append(f"  Architecture: {model.architecture or 'unknown'}")
    lines.append(f"  Precision: {model.dtype or 'unknown'}")

    lines.append("")
    lines.append("Runtime")
    lines.append(f"  Engine: {report.runtime.name}")
    lines.append(f"  Installed: {report.runtime.installed} (version: {report.runtime.version or 'n/a'})")

    lines.append("")
    lines.append("Compatibility")
    for reason in report.compatibility.reasons:
        lines.append(f"  ✓ {reason}")
    for warning in report.compatibility.warnings:
        lines.append(f"  ! {warning.message}")
    for error in report.compatibility.errors:
        lines.append(f"  ✗ {error.message}")

    if report.deployment_plan:
        lines.append("")
        lines.append("Deployment Plan")
        for key, value in report.deployment_plan.items():
            lines.append(f"  {key}: {value}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings")
        for warning in report.warnings:
            lines.append(f"  {warning}")

    return "\n".join(lines)
