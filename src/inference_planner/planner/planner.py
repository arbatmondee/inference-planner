"""Top-level orchestration: hardware + model + runtime -> AnalysisReport."""

from __future__ import annotations

from inference_planner.compatibility.analyzer import CompatibilityAnalyzer
from inference_planner.compatibility.rules import CompatibilityContext
from inference_planner.core.enums import AnalysisStage
from inference_planner.hardware.base import HardwareInfo
from inference_planner.hardware.detector import detect_hardware
from inference_planner.models.analyzer import ModelAnalyzer
from inference_planner.runtimes.registry import get_adapter
from inference_planner.schemas.report import AnalysisReport


class InferencePlanner:
    """Ties hardware detection, model inspection, and a runtime adapter together.

    This is the only class most callers need:

        report = InferencePlanner().analyze(model="Qwen/Qwen3-8B", runtime="vllm")
        if report.compatible:
            plan = report.deployment_plan
    """

    def __init__(
        self,
        *,
        model_analyzer: ModelAnalyzer | None = None,
        compatibility_analyzer: CompatibilityAnalyzer | None = None,
    ) -> None:
        self._model_analyzer = model_analyzer or ModelAnalyzer()
        self._compatibility_analyzer = compatibility_analyzer or CompatibilityAnalyzer()

    def analyze(
        self,
        *,
        model: str,
        runtime: str,
        hardware: HardwareInfo | None = None,
        tensor_parallel_size: int | None = None,
        hf_token: str | None = None,
    ) -> AnalysisReport:
        """``hf_token`` is only consulted by model providers that need auth (e.g.
        for gated/private Hugging Face Hub repos); it is ignored by providers
        that don't. If omitted, providers fall back to their own defaults
        (for the Hugging Face provider, that means any locally cached
        ``huggingface-cli login`` token or the ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN``
        env vars).
        """
        hardware = hardware or detect_hardware()
        model_info = self._model_analyzer.inspect(model, token=hf_token)
        adapter = get_adapter(runtime)

        identity = adapter.identify()
        capabilities = adapter.capabilities()
        resource_estimate = adapter.estimate_resources(
            model_info, hardware, tensor_parallel_size=tensor_parallel_size
        )
        tp_used = tensor_parallel_size or resource_estimate.recommended_tensor_parallel_size

        ctx = CompatibilityContext(
            model=model_info,
            hardware=hardware,
            runtime_identity=identity,
            runtime_capabilities=capabilities,
            resource_estimate=resource_estimate,
            tensor_parallel_size=tp_used,
        )
        compatibility_result = self._compatibility_analyzer.analyze(ctx)

        deployment_plan = None
        if compatibility_result.compatible:
            deployment_plan = adapter.generate_config(
                model_info, hardware, tensor_parallel_size=tensor_parallel_size
            )

        warnings = tuple(hardware.detection_warnings) + tuple(
            issue.message for issue in compatibility_result.warnings
        )
        errors = tuple(issue.message for issue in compatibility_result.errors)

        return AnalysisReport(
            compatible=compatibility_result.compatible,
            stage=AnalysisStage.STATIC_ANALYSIS,
            hardware=hardware,
            model=model_info,
            runtime=identity,
            compatibility=compatibility_result,
            resource_estimate=resource_estimate,
            deployment_plan=deployment_plan,
            warnings=warnings,
            errors=errors,
        )


def analyze(
    *,
    model: str,
    runtime: str,
    hardware: HardwareInfo | None = None,
    tensor_parallel_size: int | None = None,
    hf_token: str | None = None,
) -> AnalysisReport:
    """Convenience module-level function: ``planner.analyze(model=..., runtime=...)``."""
    return InferencePlanner().analyze(
        model=model,
        runtime=runtime,
        hardware=hardware,
        tensor_parallel_size=tensor_parallel_size,
        hf_token=hf_token,
    )
