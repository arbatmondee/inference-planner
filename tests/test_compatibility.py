from __future__ import annotations

from inference_planner.compatibility.analyzer import CompatibilityAnalyzer
from inference_planner.compatibility.rules import CompatibilityContext
from inference_planner.core.enums import Confidence, Severity
from inference_planner.resources.estimator import ResourceEstimator
from inference_planner.resources.types import CountEstimate, MemoryEstimate, OverheadProfile, ResourceEstimate
from inference_planner.runtimes.base import RuntimeCapabilities, RuntimeIdentity
from tests.conftest import make_hardware, make_model

PROFILE = OverheadProfile(fixed_overhead_gb=1.0, per_gpu_fixed_overhead_gb=0.5, activation_overhead_fraction=0.1)


def _build_estimate(model, hardware, tp=1):
    return ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=tp)


def test_fully_compatible_scenario_reports_reasons_and_no_errors():
    model = make_model()
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("auto", "bfloat16"),
            supported_quantization_methods=(),
        ),
        resource_estimate=estimate,
        tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is True
    assert result.errors == ()
    assert any("VRAM" in r for r in result.reasons)


def test_runtime_not_installed_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=False),
        runtime_capabilities=RuntimeCapabilities(),
        resource_estimate=estimate,
        tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "RUNTIME_NOT_INSTALLED" for e in result.errors)


def test_insufficient_vram_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=1, gpu_overrides={"memory_free_mb": 1_000})
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate,
        tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "INSUFFICIENT_VRAM" for e in result.errors)


def test_unsupported_dtype_is_an_error():
    model = make_model(dtype="float32")
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16", "float16")),
        resource_estimate=estimate,
        tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "UNSUPPORTED_DTYPE" for e in result.errors)


def test_unsupported_architecture_is_only_a_warning():
    model = make_model(architecture=None)
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate,
        tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is True
    assert any(w.code == "UNKNOWN_ARCHITECTURE" for w in result.warnings)


def test_tensor_parallel_head_mismatch_is_an_error():
    from inference_planner.models.base import AttentionConfig

    model = make_model(attention=AttentionConfig(num_attention_heads=30, num_key_value_heads=10, head_dim=128))
    hardware = make_hardware(gpu_count=4)
    estimate = _build_estimate(model, hardware, tp=4)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate,
        tensor_parallel_size=4,  # 30 heads not divisible by 4
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "TENSOR_PARALLEL_HEAD_MISMATCH" for e in result.errors)


def test_tensor_parallel_divisible_configuration_passes():
    from inference_planner.models.base import AttentionConfig

    model = make_model(attention=AttentionConfig(num_attention_heads=32, num_key_value_heads=8, head_dim=128))
    hardware = make_hardware(gpu_count=4)
    estimate = _build_estimate(model, hardware, tp=4)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate,
        tensor_parallel_size=4,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert not any(e.code == "TENSOR_PARALLEL_HEAD_MISMATCH" for e in result.errors)


def test_no_gpu_is_only_a_warning_not_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=0)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model,
        hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate,
        tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert any(w.code == "NO_GPU_DETECTED" for w in result.warnings)
    assert not any(e.code == "NO_GPU_DETECTED" for e in result.errors)
