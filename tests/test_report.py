from __future__ import annotations

import json

from inference_planner.compatibility.analyzer import CompatibilityResult
from inference_planner.core.enums import AnalysisStage, Confidence
from inference_planner.resources.estimator import ResourceEstimator
from inference_planner.resources.types import OverheadProfile
from inference_planner.runtimes.base import RuntimeIdentity
from inference_planner.schemas.report import AnalysisReport
from tests.conftest import make_hardware, make_model

PROFILE = OverheadProfile(fixed_overhead_gb=1.0, per_gpu_fixed_overhead_gb=0.5, activation_overhead_fraction=0.1)


def _build_report(compatible: bool = True) -> AnalysisReport:
    model = make_model()
    hardware = make_hardware(gpu_count=1)
    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)
    return AnalysisReport(
        compatible=compatible,
        stage=AnalysisStage.STATIC_ANALYSIS,
        hardware=hardware,
        model=model,
        runtime=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        compatibility=CompatibilityResult(compatible=compatible, reasons=("Model fits in available VRAM",)),
        resource_estimate=estimate,
        deployment_plan={"runtime": "vllm", "tensor_parallel_size": 1} if compatible else None,
    )


def test_report_serializes_to_valid_json():
    report = _build_report()
    payload = json.loads(report.to_json())

    assert payload["compatible"] is True
    assert payload["stage"] == "static_analysis"
    assert payload["model"]["identifier"] == "mock/model-7b"
    assert payload["resource_estimate"]["weight_memory"]["confidence"] == "estimated"
    # Enums must serialize to plain strings, not repr()
    assert isinstance(payload["hardware"]["confidence"], str)


def test_report_json_round_trips_with_no_python_objects_left():
    report = _build_report()
    text = report.to_json()
    # A strict json.loads with no error means every enum/tuple/dataclass was flattened.
    reloaded = json.loads(text)
    assert isinstance(reloaded, dict)


def test_human_readable_rendering_includes_key_sections():
    report = _build_report()
    text = str(report)

    assert "Inference Planner" in text
    assert "Hardware" in text
    assert "Model" in text
    assert "Runtime" in text
    assert "Compatibility" in text
    assert "Deployment Plan" in text


def test_human_readable_rendering_omits_plan_when_incompatible():
    report = _build_report(compatible=False)
    text = str(report)

    assert "Deployment Plan" not in text
