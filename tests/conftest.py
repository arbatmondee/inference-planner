from __future__ import annotations

import pytest

from inference_planner.core.enums import Confidence
from inference_planner.hardware.base import (
    CPUInfo,
    GPUInfo,
    GPUVendor,
    HardwareInfo,
    RuntimeStackInfo,
)
from inference_planner.models.base import (
    AttentionConfig,
    ModelFormat,
    ModelInfo,
    ParameterCountEstimate,
    QuantizationInfo,
    TokenizerInfo,
)


def make_cpu(**overrides) -> CPUInfo:
    defaults = dict(
        model="Test CPU",
        physical_cores=8,
        logical_cores=16,
        total_memory_mb=64_000.0,
        available_memory_mb=48_000.0,
    )
    defaults.update(overrides)
    return CPUInfo(**defaults)


def make_gpu(index: int = 0, **overrides) -> GPUInfo:
    defaults = dict(
        index=index,
        name="Mock GPU",
        vendor=GPUVendor.NVIDIA,
        memory_total_mb=24_000.0,
        memory_free_mb=20_000.0,
        compute_capability="8.6",
        driver_version="550.00",
        supported_precisions=("fp32", "fp16", "bf16"),
    )
    defaults.update(overrides)
    return GPUInfo(**defaults)


def make_hardware(gpu_count: int = 1, cpu: CPUInfo | None = None, gpu_overrides: dict | None = None, **overrides) -> HardwareInfo:
    gpus = tuple(make_gpu(index=i, **(gpu_overrides or {})) for i in range(gpu_count))
    defaults = dict(
        cpu=cpu or make_cpu(),
        gpus=gpus,
        runtime_stack=RuntimeStackInfo(cuda_version="12.4", cuda_driver_version="550.00"),
        confidence=Confidence.MEASURED,
        detection_warnings=(),
    )
    defaults.update(overrides)
    return HardwareInfo(**defaults)


def make_model(**overrides) -> ModelInfo:
    defaults = dict(
        identifier="mock/model-7b",
        architecture="MockForCausalLM",
        model_type="mock",
        task="text-generation",
        parameter_count=ParameterCountEstimate(
            count=7_000_000_000, confidence=Confidence.ESTIMATED, basis=("safetensors index",)
        ),
        format=ModelFormat.SAFETENSORS,
        quantization=QuantizationInfo(),
        dtype="bfloat16",
        context_length=8192,
        num_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        vocab_size=32000,
        attention=AttentionConfig(num_attention_heads=32, num_key_value_heads=8, head_dim=128),
        tokenizer=TokenizerInfo(vocab_size=32000, model_max_length=8192, has_chat_template=True),
        raw_config={},
    )
    defaults.update(overrides)
    return ModelInfo(**defaults)


@pytest.fixture
def hardware_1gpu():
    return make_hardware(gpu_count=1)


@pytest.fixture
def hardware_2gpu():
    return make_hardware(gpu_count=2)


@pytest.fixture
def hardware_no_gpu():
    return make_hardware(gpu_count=0)


@pytest.fixture
def model_7b():
    return make_model()
