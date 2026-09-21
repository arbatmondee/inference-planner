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


def test_tp_recommendation_skips_degrees_that_dont_divide_heads():
    # 30 attention heads: divisible by 2, not by 4. TP=1 doesn't fit (only
    # ~90GB/GPU free vs. ~145GB needed), and TP=4 would fit best memory-wise
    # but can't shard 30 heads evenly -- so TP=2 (which does fit) must win.
    model = make_model(
        parameter_count=ParameterCountEstimate(count=70_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
        attention=AttentionConfig(num_attention_heads=30, num_key_value_heads=30, head_dim=128),
    )
    hardware = make_hardware(gpu_count=4, gpu_overrides={"memory_free_mb": 90_000})

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=None)

    assert estimate.recommended_tensor_parallel_size == 2


def test_tp_recommendation_restricted_to_selected_device_count():
    # 8 GPUs on the node, but only 2 are selected -> must only ever consider
    # divisors of 2 (1 or 2), never e.g. TP=4 or TP=8.
    model = make_model(
        parameter_count=ParameterCountEstimate(count=1_000_000_000, confidence=Confidence.ESTIMATED),
        dtype="bfloat16",
    )
    hardware = make_hardware(gpu_count=8, gpu_overrides={"memory_free_mb": 40_000})

    estimate = ResourceEstimator().estimate(
        model, hardware, PROFILE, tensor_parallel_size=None, device_ids=(0, 1)
    )

    assert estimate.recommended_tensor_parallel_size in (1, 2)


def test_device_ids_restrict_available_vram_to_selected_gpus():
    hardware = make_hardware(gpu_count=1)
    # Build a 3-GPU hardware snapshot with distinct free memory per GPU.
    from tests.conftest import make_gpu
    from inference_planner.hardware.base import HardwareInfo

    hardware = HardwareInfo(
        cpu=hardware.cpu,
        gpus=(
            make_gpu(index=0, memory_free_mb=10_000),
            make_gpu(index=1, memory_free_mb=40_000),
            make_gpu(index=2, memory_free_mb=80_000),
        ),
        runtime_stack=hardware.runtime_stack,
    )
    model = make_model()

    estimate_all = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)
    estimate_subset = ResourceEstimator().estimate(
        model, hardware, PROFILE, tensor_parallel_size=1, device_ids=(1, 2)
    )

    assert estimate_all.available_vram_gb == 10_000 / 1024   # min across all 3
    assert estimate_subset.available_vram_gb == 40_000 / 1024  # min across selected {1, 2}


def test_device_ids_referencing_nonexistent_gpu_yields_zero_available_vram():
    model = make_model()
    hardware = make_hardware(gpu_count=2)  # indices 0, 1

    estimate = ResourceEstimator().estimate(
        model, hardware, PROFILE, tensor_parallel_size=1, device_ids=(5,)
    )

    assert estimate.available_vram_gb == 0.0
    assert estimate.fits_in_available_vram is False


def test_no_gpu_means_zero_available_vram():
    model = make_model()
    hardware = make_hardware(gpu_count=0)

    estimate = ResourceEstimator().estimate(model, hardware, PROFILE, tensor_parallel_size=1)

    assert estimate.available_vram_gb == 0.0
    assert estimate.fits_in_available_vram is False
