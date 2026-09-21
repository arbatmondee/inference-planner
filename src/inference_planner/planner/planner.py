"""Top-level orchestration: hardware + model + runtime -> AnalysisReport."""

from __future__ import annotations

import logging

from inference_planner.compatibility.analyzer import CompatibilityAnalyzer
from inference_planner.compatibility.rules import CompatibilityContext
from inference_planner.core.enums import AnalysisStage
from inference_planner.hardware.base import HardwareInfo
from inference_planner.hardware.detector import detect_hardware
from inference_planner.models.analyzer import ModelAnalyzer
from inference_planner.models.base import ModelInfo
from inference_planner.runtimes.base import RuntimeAdapter, RuntimeCandidate, RuntimeCapabilities, RuntimeIdentity
from inference_planner.runtimes.registry import get_adapter
from inference_planner.schemas.report import AnalysisReport
from inference_planner.validation.registry import get_validator

logger = logging.getLogger(__name__)


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
        device_ids: tuple[int, ...] | list[int] | None = None,
        hf_token: str | None = None,
        run_probe: bool = False,
        probe_timeout_seconds: int = 600,
    ) -> AnalysisReport:
        """Analyze the currently-installed ``runtime`` against ``model``/``hardware``.

        ``device_ids``, when given, pins the deployment to those specific GPU
        indices (validated against ``hardware`` by the compatibility layer)
        instead of the planner assuming devices ``0..tensor_parallel_size-1``.

        ``hf_token`` is only consulted by model providers that need auth (e.g.
        for gated/private Hugging Face Hub repos); it is ignored by providers
        that don't. If omitted, providers fall back to their own defaults
        (for the Hugging Face provider, that means any locally cached
        ``huggingface-cli login`` token or the ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN``
        env vars).

        ``run_probe=True`` is a real, opt-in Stage-2 step: if (and only if)
        static compatibility passes, it actually launches the runtime in a
        subprocess, loads the model onto the GPU, and measures load time and
        VRAM use. This downloads/loads real model weights, consumes GPU
        memory, and can take minutes — it never runs unless explicitly
        requested. See :mod:`inference_planner.validation`.
        """
        hardware = hardware or detect_hardware()
        device_ids_tuple = tuple(device_ids) if device_ids is not None else None
        model_info = self._model_analyzer.inspect(model, token=hf_token)
        adapter = get_adapter(runtime)

        identity = adapter.identify()
        capabilities = adapter.capabilities()

        return self._build_report(
            model_info=model_info,
            hardware=hardware,
            adapter=adapter,
            identity=identity,
            capabilities=capabilities,
            tensor_parallel_size=tensor_parallel_size,
            device_ids=device_ids_tuple,
            run_probe=run_probe,
            probe_timeout_seconds=probe_timeout_seconds,
        )

    def evaluate_candidates(
        self,
        *,
        model: str,
        runtime: str,
        candidate_versions: list[str],
        hardware: HardwareInfo | None = None,
        tensor_parallel_size: int | None = None,
        device_ids: tuple[int, ...] | list[int] | None = None,
        hf_token: str | None = None,
    ) -> list[AnalysisReport]:
        """Evaluate several candidate runtime *versions*, not just whatever's installed.

        Unlike :meth:`analyze`, this never introspects real code for a
        version other than the one already installed — it can only confirm
        each candidate version exists (e.g. via PyPI) and apply generic,
        conservative capabilities unless the caller pre-supplies its own via
        ``RuntimeCandidate.capabilities_override``. No runtime probe is
        offered here, since a candidate version generally isn't installed
        and so can't actually be launched.

        Returns one :class:`AnalysisReport` per candidate, in the same order
        as ``candidate_versions``.
        """
        hardware = hardware or detect_hardware()
        device_ids_tuple = tuple(device_ids) if device_ids is not None else None
        model_info = self._model_analyzer.inspect(model, token=hf_token)
        adapter = get_adapter(runtime)

        reports = []
        for version in candidate_versions:
            candidate = RuntimeCandidate(name=runtime, version=version)
            try:
                identity = adapter.identify_candidate(candidate)
                capabilities = adapter.capabilities_for_candidate(candidate)
            except NotImplementedError:
                logger.warning("Runtime adapter '%s' does not support candidate evaluation", runtime)
                identity = RuntimeIdentity(
                    name=runtime, installed=False, version=version,
                    detection_notes=(f"Adapter '{runtime}' does not support candidate evaluation.",),
                )
                capabilities = RuntimeCapabilities()

            reports.append(
                self._build_report(
                    model_info=model_info,
                    hardware=hardware,
                    adapter=adapter,
                    identity=identity,
                    capabilities=capabilities,
                    tensor_parallel_size=tensor_parallel_size,
                    device_ids=device_ids_tuple,
                    run_probe=False,
                    probe_timeout_seconds=0,
                )
            )
        return reports

    def _build_report(
        self,
        *,
        model_info: ModelInfo,
        hardware: HardwareInfo,
        adapter: RuntimeAdapter,
        identity: RuntimeIdentity,
        capabilities: RuntimeCapabilities,
        tensor_parallel_size: int | None,
        device_ids: tuple[int, ...] | None,
        run_probe: bool,
        probe_timeout_seconds: int,
    ) -> AnalysisReport:
        # If the caller pinned devices but not a TP degree, defaulting TP to the
        # device count is the sensible reading of "use exactly these GPUs". If
        # they gave *both* and they disagree, we deliberately do NOT silently
        # override one -- DeviceSelectionValidRule surfaces that as a real error.
        effective_tp_hint = tensor_parallel_size
        if effective_tp_hint is None and device_ids is not None:
            effective_tp_hint = len(device_ids)

        resource_estimate = adapter.estimate_resources(
            model_info, hardware, tensor_parallel_size=effective_tp_hint, device_ids=device_ids
        )
        tp_used = effective_tp_hint or resource_estimate.recommended_tensor_parallel_size

        ctx = CompatibilityContext(
            model=model_info,
            hardware=hardware,
            runtime_identity=identity,
            runtime_capabilities=capabilities,
            resource_estimate=resource_estimate,
            tensor_parallel_size=tp_used,
            device_ids=device_ids,
        )
        compatibility_result = self._compatibility_analyzer.analyze(ctx)

        deployment_plan = None
        if compatibility_result.compatible:
            # Pass the exact TP degree that was just validated, so the plan
            # can't disagree with the compatibility verdict it's attached to.
            deployment_plan = adapter.generate_config(
                model_info, hardware, tensor_parallel_size=tp_used, device_ids=device_ids
            )

        runtime_validation = None
        if run_probe:
            if not compatibility_result.compatible:
                logger.info("Skipping runtime probe: static compatibility failed.")
            else:
                validator = get_validator(identity.name)
                if validator is None:
                    logger.warning("No runtime validator registered for '%s'; skipping probe.", identity.name)
                else:
                    runtime_validation = validator.validate(
                        model_info, hardware, deployment_plan, timeout_seconds=probe_timeout_seconds
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
            runtime_validation=runtime_validation,
        )


def analyze(
    *,
    model: str,
    runtime: str,
    hardware: HardwareInfo | None = None,
    tensor_parallel_size: int | None = None,
    device_ids: tuple[int, ...] | list[int] | None = None,
    hf_token: str | None = None,
    run_probe: bool = False,
    probe_timeout_seconds: int = 600,
) -> AnalysisReport:
    """Convenience module-level function: ``planner.analyze(model=..., runtime=...)``."""
    return InferencePlanner().analyze(
        model=model,
        runtime=runtime,
        hardware=hardware,
        tensor_parallel_size=tensor_parallel_size,
        device_ids=device_ids,
        hf_token=hf_token,
        run_probe=run_probe,
        probe_timeout_seconds=probe_timeout_seconds,
    )
