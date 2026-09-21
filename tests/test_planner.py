from __future__ import annotations

import json

import pytest

from inference_planner.planner.planner import InferencePlanner
from inference_planner.resources.types import OverheadProfile, ResourceEstimate, MemoryEstimate, CountEstimate
from inference_planner.runtimes.base import RuntimeAdapter, RuntimeCapabilities, RuntimeIdentity
from inference_planner.runtimes.registry import register_adapter
from inference_planner.core.enums import Confidence
from tests.conftest import make_hardware


def _write_local_model(tmp_path, **config_overrides):
    config = {
        "architectures": ["MockForCausalLM"],
        "model_type": "mock",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "vocab_size": 32000,
        "torch_dtype": "bfloat16",
    }
    config.update(config_overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").touch()
    return str(tmp_path)


class _ControllableAdapter(RuntimeAdapter):
    """Fake adapter so planner tests don't depend on vLLM being installed."""

    def __init__(self, *, installed=True, fits=True):
        self._installed = installed
        self._fits = fits

    def identify(self):
        return RuntimeIdentity(name="faketest", installed=self._installed, version="1.0" if self._installed else None)

    def capabilities(self):
        return RuntimeCapabilities(supported_dtypes=("bfloat16", "auto"))

    def overhead_profile(self):
        return OverheadProfile(fixed_overhead_gb=1.0)

    def estimate_resources(self, model, hardware, *, tensor_parallel_size=None):
        return ResourceEstimate(
            weight_memory=MemoryEstimate(estimated_gb=14.0, confidence=Confidence.ESTIMATED),
            kv_cache_memory_per_1k_tokens=MemoryEstimate(estimated_gb=0.1, confidence=Confidence.ESTIMATED),
            runtime_overhead_memory=MemoryEstimate(estimated_gb=1.5, confidence=Confidence.HEURISTIC),
            total_vram_required=MemoryEstimate(estimated_gb=15.5, confidence=Confidence.ESTIMATED),
            available_vram_gb=hardware.total_free_vram_mb / 1024 if hardware.gpus else 0.0,
            fits_in_available_vram=self._fits,
            cpu_ram_required=MemoryEstimate(estimated_gb=14.0, confidence=Confidence.HEURISTIC),
            max_context_length_estimate=CountEstimate(estimated_value=8192, confidence=Confidence.HEURISTIC),
            recommended_tensor_parallel_size=tensor_parallel_size or 1,
            approximate_max_concurrency=CountEstimate(estimated_value=4, confidence=Confidence.HEURISTIC),
        )

    def generate_config(self, model, hardware, *, tensor_parallel_size=None):
        return {"runtime": "faketest", "tensor_parallel_size": tensor_parallel_size or 1}


@pytest.fixture(autouse=True)
def _register_fake_adapters():
    register_adapter("faketest-compatible", _ControllableAdapter(installed=True, fits=True))
    register_adapter("faketest-incompatible", _ControllableAdapter(installed=True, fits=False))
    register_adapter("faketest-not-installed", _ControllableAdapter(installed=False))


def test_analyze_returns_deployment_plan_when_compatible(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)

    report = InferencePlanner().analyze(model=model_path, runtime="faketest-compatible", hardware=hardware)

    assert report.compatible is True
    assert report.deployment_plan is not None
    assert report.deployment_plan["runtime"] == "faketest"
    assert report.errors == ()


def test_analyze_withholds_deployment_plan_when_incompatible(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)

    report = InferencePlanner().analyze(model=model_path, runtime="faketest-incompatible", hardware=hardware)

    assert report.compatible is False
    assert report.deployment_plan is None
    assert any("VRAM" in e for e in report.errors)


def test_analyze_forwards_hf_token_to_model_analyzer(tmp_path, monkeypatch):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)
    seen_tokens = []

    from inference_planner.models.analyzer import ModelAnalyzer

    original_inspect = ModelAnalyzer.inspect

    def spy_inspect(self, identifier, *, token=None):
        seen_tokens.append(token)
        return original_inspect(self, identifier, token=token)

    monkeypatch.setattr(ModelAnalyzer, "inspect", spy_inspect)

    InferencePlanner().analyze(
        model=model_path, runtime="faketest-compatible", hardware=hardware, hf_token="my-secret-token"
    )

    assert seen_tokens == ["my-secret-token"]


def test_analyze_reports_runtime_not_installed(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)

    report = InferencePlanner().analyze(model=model_path, runtime="faketest-not-installed", hardware=hardware)

    assert report.compatible is False
    assert report.runtime.installed is False
