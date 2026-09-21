"""inference_planner: hardware + model + runtime analysis and deployment planning.

Typical usage::

    from inference_planner import InferencePlanner

    report = InferencePlanner().analyze(model="Qwen/Qwen3-8B", runtime="vllm")
    if report.compatible:
        plan = report.deployment_plan
"""

from inference_planner.hardware.detector import detect_hardware
from inference_planner.models.analyzer import inspect as inspect_model
from inference_planner.planner.planner import InferencePlanner, analyze
from inference_planner.schemas.report import AnalysisReport

__version__ = "0.1.0"

__all__ = [
    "InferencePlanner",
    "analyze",
    "detect_hardware",
    "inspect_model",
    "AnalysisReport",
    "__version__",
]
