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


def test_invalid_device_id_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=2)  # indices 0, 1
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate, tensor_parallel_size=1,
        device_ids=(0, 5),
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "INVALID_DEVICE_ID" for e in result.errors)


def test_device_count_tp_mismatch_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=4)
    estimate = _build_estimate(model, hardware, tp=2)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate, tensor_parallel_size=2,
        device_ids=(0, 1, 2),  # 3 devices selected but tensor_parallel_size=2
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "DEVICE_COUNT_TP_MISMATCH" for e in result.errors)


def test_valid_device_selection_passes():
    model = make_model()
    hardware = make_hardware(gpu_count=4)
    estimate = _build_estimate(model, hardware, tp=2)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate, tensor_parallel_size=2,
        device_ids=(1, 2),
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert not any(e.code in ("INVALID_DEVICE_ID", "DEVICE_COUNT_TP_MISMATCH") for e in result.errors)


def test_mig_gpu_with_tensor_parallelism_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=2, gpu_overrides={"mig_enabled": True})
    estimate = _build_estimate(model, hardware, tp=2)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate, tensor_parallel_size=2,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "MIG_TENSOR_PARALLEL_UNSUPPORTED" for e in result.errors)


def test_mig_gpu_without_tensor_parallelism_is_fine():
    model = make_model()
    hardware = make_hardware(gpu_count=1, gpu_overrides={"mig_enabled": True})
    estimate = _build_estimate(model, hardware, tp=1)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(supported_dtypes=("bfloat16",)),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert not any(e.code == "MIG_TENSOR_PARALLEL_UNSUPPORTED" for e in result.errors)


def test_unsupported_python_version_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",), required_python_specifier=">=3.99",
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "UNSUPPORTED_PYTHON_VERSION" for e in result.errors)


def test_python_version_within_range_passes():
    import sys

    model = make_model()
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)
    current = f">={sys.version_info.major}.{sys.version_info.minor}"

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",), required_python_specifier=current,
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert not any(e.code == "UNSUPPORTED_PYTHON_VERSION" for e in result.errors)


def test_torch_build_cuda_newer_than_driver_is_an_error():
    model = make_model()
    hardware = make_hardware(gpu_count=1)
    hardware = _with_cuda_version(hardware, "11.8")
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",), torch_build_cuda_version="12.4",
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "CUDA_VERSION_MISMATCH" for e in result.errors)


def test_torch_build_cuda_older_than_driver_passes():
    model = make_model()
    hardware = make_hardware(gpu_count=1)
    hardware = _with_cuda_version(hardware, "12.4")
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",), torch_build_cuda_version="11.8",
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert not any(e.code == "CUDA_VERSION_MISMATCH" for e in result.errors)


def _with_cuda_version(hardware, cuda_version):
    from dataclasses import replace
    return replace(hardware, runtime_stack=replace(hardware.runtime_stack, cuda_version=cuda_version))


def test_architecture_not_in_registry_without_custom_code_is_an_error():
    model = make_model(architecture="SomeExoticForCausalLM", raw_config={})
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",),
            supported_architectures=("LlamaForCausalLM", "Qwen2ForCausalLM"),
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is False
    assert any(e.code == "ARCHITECTURE_NOT_SUPPORTED_BY_RUNTIME" for e in result.errors)


def test_architecture_not_in_registry_with_custom_code_is_only_a_warning():
    model = make_model(architecture="SomeExoticForCausalLM", raw_config={"auto_map": {"AutoModel": "modeling.MyModel"}})
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",),
            supported_architectures=("LlamaForCausalLM", "Qwen2ForCausalLM"),
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert result.compatible is True
    assert any(w.code == "ARCHITECTURE_NOT_IN_REGISTRY_HAS_CUSTOM_CODE" for w in result.warnings)


def test_architecture_in_registry_passes_cleanly():
    model = make_model(architecture="LlamaForCausalLM")
    hardware = make_hardware(gpu_count=1)
    estimate = _build_estimate(model, hardware)

    ctx = CompatibilityContext(
        model=model, hardware=hardware,
        runtime_identity=RuntimeIdentity(name="vllm", installed=True, version="0.6.3"),
        runtime_capabilities=RuntimeCapabilities(
            supported_dtypes=("bfloat16",),
            supported_architectures=("LlamaForCausalLM", "Qwen2ForCausalLM"),
        ),
        resource_estimate=estimate, tensor_parallel_size=1,
    )

    result = CompatibilityAnalyzer().analyze(ctx)

    assert not any(e.code == "ARCHITECTURE_NOT_SUPPORTED_BY_RUNTIME" for e in result.errors)
    assert any("natively supported" in r for r in result.reasons)
