from __future__ import annotations

from inference_planner.core.enums import Confidence
from inference_planner.models.base import AttentionConfig, ParameterCountEstimate
from inference_planner.resources.estimator import ResourceEstimator
from inference_planner.resources.types import OverheadProfile
from tests.conftest import make_hardware, make_model

PROFILE = OverheadProfile(
    fixed_overhead_gb=1.0,
    per_gpu_fixed_overhead_gb=0.5,
    activation_overhead_fraction=0.10,
    default_gpu_memory_utilization=0.90,
)


def test_weight_memory_scales_with_param_count_and_dtype():
    model = make_model(
        parameter_count=ParameterCountEstimate(count=7_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
    )
    hardware = make_hardware(gpu_count=1)

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    # 7B params * 2 bytes/param, using binary GB (1024**3) as this codebase's unit throughout.
    assert estimate.weight_memory.estimated_gb == 7_000_000_000 * 2 / (1024**3)
    assert estimate.weight_memory.confidence == Confidence.ESTIMATED


def test_weight_memory_shards_across_tensor_parallel_size():
    model = make_model(
        parameter_count=ParameterCountEstimate(count=70_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
    )
    hardware = make_hardware(gpu_count=4)

    est_tp1 = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)
    est_tp4 = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=4)

    assert est_tp4.weight_memory.estimated_gb == est_tp1.weight_memory.estimated_gb / 4


def test_insufficient_vram_is_reported():
    model = make_model(
        parameter_count=ParameterCountEstimate(count=70_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
    )
    hardware = make_hardware(gpu_count=1, gpu_overrides={"memory_free_mb": 8_000})

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    assert estimate.fits_in_available_vram is False


def test_sufficient_vram_is_reported():
    model = make_model(
        parameter_count=ParameterCountEstimate(count=1_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
    )
    hardware = make_hardware(gpu_count=1, gpu_overrides={"memory_free_mb": 40_000})

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    assert estimate.fits_in_available_vram is True


def test_auto_recommends_tensor_parallel_size_that_fits():
    # 70B bf16 = 140GB, doesn't fit on one 24GB GPU, but does across 8 sharded (17.5GB/GPU).
    model = make_model(
        parameter_count=ParameterCountEstimate(count=70_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
    )
    hardware = make_hardware(gpu_count=8, gpu_overrides={"memory_free_mb": 24_000})

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=None)

    assert estimate.recommended_tensor_parallel_size in (4, 8)
    assert estimate.fits_in_available_vram is True


def test_kv_cache_unknown_when_attention_config_missing():
    model = make_model(attention=AttentionConfig())
    hardware = make_hardware(gpu_count=1)

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    assert estimate.kv_cache_memory_per_1k_tokens.confidence == Confidence.UNKNOWN
    assert estimate.max_context_length_estimate.confidence == Confidence.UNKNOWN


def test_weight_memory_unknown_when_param_count_unknown():
    model = make_model(parameter_count=ParameterCountEstimate(count=None, confidence=Confidence.UNKNOWN))
    hardware = make_hardware(gpu_count=1)

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    assert estimate.weight_memory.confidence == Confidence.UNKNOWN
    assert estimate.fits_in_available_vram is None


def test_no_gpu_means_zero_available_vram():
    model = make_model()
    hardware = make_hardware(gpu_count=0)

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    assert estimate.available_vram_gb == 0.0
    assert estimate.fits_in_available_vram is False
