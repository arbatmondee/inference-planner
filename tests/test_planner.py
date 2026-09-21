from __future__ import annotations

import json

import pytest

from inference_planner.planner.planner import InferencePlanner
from inference_planner.resources.types import OverheadProfile, ResourceEstimate, MemoryEstimate, CountEstimate
from inference_planner.runtimes.base import RuntimeAdapter, RuntimeCandidate, RuntimeCapabilities, RuntimeIdentity
from inference_planner.runtimes.registry import register_adapter
from inference_planner.validation.base import RuntimeValidationResult, RuntimeValidator
from inference_planner.validation.registry import register_validator
from inference_planner.core.enums import AnalysisStage, Confidence
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

    def estimate_resources(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
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

    def generate_config(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
        return {
            "runtime": "faketest",
            "tensor_parallel_size": tensor_parallel_size or 1,
            "devices": list(device_ids) if device_ids is not None else None,
        }


class _RecordingAdapter(_ControllableAdapter):
    """Records the device_ids it was called with, for plumbing tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_estimate_device_ids = []
        self.seen_config_device_ids = []

    def estimate_resources(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
        self.seen_estimate_device_ids.append(device_ids)
        return super().estimate_resources(model, hardware, tensor_parallel_size=tensor_parallel_size, device_ids=device_ids)

    def generate_config(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
        self.seen_config_device_ids.append(device_ids)
        return super().generate_config(model, hardware, tensor_parallel_size=tensor_parallel_size, device_ids=device_ids)


class _CandidateAwareAdapter(_ControllableAdapter):
    def identify_candidate(self, candidate: RuntimeCandidate) -> RuntimeIdentity:
        return RuntimeIdentity(name=candidate.name, installed=True, version=candidate.version)

    def capabilities_for_candidate(self, candidate: RuntimeCandidate) -> RuntimeCapabilities:
        return RuntimeCapabilities(supported_dtypes=("bfloat16", "auto"))


class _FakeValidator(RuntimeValidator):
    def __init__(self):
        self.calls = []

    def validate(self, model, hardware, config, *, timeout_seconds=600):
        self.calls.append((model, hardware, config, timeout_seconds))
        return RuntimeValidationResult(
            stage=AnalysisStage.RUNTIME_VALIDATION,
            confidence=Confidence.MEASURED,
            model_load_time_seconds=12.3,
        )


@pytest.fixture(autouse=True)
def _register_fake_adapters():
    register_adapter("faketest-compatible", _ControllableAdapter(installed=True, fits=True))
    register_adapter("faketest-incompatible", _ControllableAdapter(installed=True, fits=False))
    register_adapter("faketest-not-installed", _ControllableAdapter(installed=False))
    register_adapter("faketest-recording", _RecordingAdapter(installed=True, fits=True))
    register_adapter("faketest-candidates", _CandidateAwareAdapter(installed=True, fits=True))
    register_adapter("faketest-no-candidate-support", _ControllableAdapter(installed=True, fits=True))


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


def test_device_ids_are_threaded_through_to_the_adapter(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=4)
    from inference_planner.runtimes.registry import get_adapter

    report = InferencePlanner().analyze(
        model=model_path, runtime="faketest-recording", hardware=hardware, device_ids=(1, 2),
    )

    adapter = get_adapter("faketest-recording")
    assert (1, 2) in adapter.seen_estimate_device_ids
    assert (1, 2) in adapter.seen_config_device_ids
    assert report.compatibility.compatible or not report.compatibility.compatible  # doesn't crash either way


def test_device_ids_default_tp_to_device_count_when_tp_not_given(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=4)

    # The default model fixture has 32 attention heads, so TP must divide 32;
    # 2 devices (TP=2) is valid, 3 would trip TENSOR_PARALLEL_HEAD_MISMATCH.
    report = InferencePlanner().analyze(
        model=model_path, runtime="faketest-recording", hardware=hardware, device_ids=(0, 1),
    )

    assert report.deployment_plan["tensor_parallel_size"] == 2


def test_evaluate_candidates_returns_one_report_per_version(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)

    reports = InferencePlanner().evaluate_candidates(
        model=model_path,
        runtime="faketest-candidates",
        candidate_versions=["1.0.0", "2.0.0"],
        hardware=hardware,
    )

    assert len(reports) == 2
    assert reports[0].runtime.version == "1.0.0"
    assert reports[1].runtime.version == "2.0.0"
    assert all(r.stage == AnalysisStage.STATIC_ANALYSIS for r in reports)


def test_evaluate_candidates_degrades_gracefully_for_adapter_without_support(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)

    reports = InferencePlanner().evaluate_candidates(
        model=model_path,
        runtime="faketest-no-candidate-support",
        candidate_versions=["9.9.9"],
        hardware=hardware,
    )

    assert len(reports) == 1
    assert reports[0].runtime.installed is False
    assert reports[0].compatible is False


def test_run_probe_invokes_validator_when_compatible(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)
    validator = _FakeValidator()
    register_validator("faketest", validator)

    report = InferencePlanner().analyze(
        model=model_path, runtime="faketest-compatible", hardware=hardware,
        run_probe=True, probe_timeout_seconds=42,
    )

    assert report.runtime_validation is not None
    assert report.runtime_validation.model_load_time_seconds == 12.3
    assert len(validator.calls) == 1
    assert validator.calls[0][3] == 42  # timeout_seconds forwarded


def test_run_probe_skipped_when_incompatible(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)
    validator = _FakeValidator()
    register_validator("faketest", validator)

    report = InferencePlanner().analyze(
        model=model_path, runtime="faketest-incompatible", hardware=hardware, run_probe=True,
    )

    assert report.runtime_validation is None
    assert len(validator.calls) == 0


def test_probe_not_run_by_default(tmp_path):
    model_path = _write_local_model(tmp_path)
    hardware = make_hardware(gpu_count=1)
    validator = _FakeValidator()
    register_validator("faketest", validator)

    report = InferencePlanner().analyze(model=model_path, runtime="faketest-compatible", hardware=hardware)

    assert report.runtime_validation is None
    assert len(validator.calls) == 0
