from __future__ import annotations

import json

import pytest

from inference_planner import cli
from inference_planner.planner.planner import InferencePlanner
from inference_planner.resources.types import CountEstimate, MemoryEstimate, OverheadProfile, ResourceEstimate
from inference_planner.core.enums import Confidence
from inference_planner.runtimes.base import RuntimeAdapter, RuntimeCapabilities, RuntimeIdentity
from inference_planner.runtimes.registry import register_adapter
from tests.conftest import make_hardware


class _CliFakeAdapter(RuntimeAdapter):
    def identify(self):
        return RuntimeIdentity(name="clitest", installed=True, version="1.0")

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
            available_vram_gb=40.0,
            fits_in_available_vram=True,
            cpu_ram_required=MemoryEstimate(estimated_gb=14.0, confidence=Confidence.HEURISTIC),
            max_context_length_estimate=CountEstimate(estimated_value=8192, confidence=Confidence.HEURISTIC),
            recommended_tensor_parallel_size=tensor_parallel_size or 1,
            approximate_max_concurrency=CountEstimate(estimated_value=4, confidence=Confidence.HEURISTIC),
        )

    def generate_config(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
        return {"runtime": "clitest", "tensor_parallel_size": tensor_parallel_size or 1}


@pytest.fixture(autouse=True)
def _register_cli_fake_adapter():
    register_adapter("clitest", _CliFakeAdapter())


def _write_local_model(tmp_path):
    config = {
        "architectures": ["MockForCausalLM"],
        "model_type": "mock",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "vocab_size": 32000,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").touch()
    return str(tmp_path)


def test_hardware_command_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_hardware", lambda: make_hardware(gpu_count=1))

    exit_code = cli.main(["hardware", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpus"][0]["name"] == "Mock GPU"


def test_hardware_command_prints_text(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_hardware", lambda: make_hardware(gpu_count=1))

    exit_code = cli.main(["hardware"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "GPU 0: Mock GPU" in out


def test_analyze_command_exit_code_reflects_compatibility(tmp_path, capsys, monkeypatch):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=1),
    )

    exit_code = cli.main(["analyze", "--model", model_path, "--runtime", "clitest"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Inference Planner" in out
    assert "Deployment Plan" in out


def test_analyze_command_forwards_hf_token(tmp_path, monkeypatch):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=1),
    )
    seen_tokens = []
    original_analyze = InferencePlanner.analyze

    def spy_analyze(self, **kwargs):
        seen_tokens.append(kwargs.get("hf_token"))
        return original_analyze(self, **kwargs)

    monkeypatch.setattr(InferencePlanner, "analyze", spy_analyze)

    cli.main(["analyze", "--model", model_path, "--runtime", "clitest", "--hf-token", "cli-secret"])

    assert seen_tokens == ["cli-secret"]


def test_analyze_command_forwards_device_ids(tmp_path, monkeypatch):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=4),
    )
    seen = []
    original_analyze = InferencePlanner.analyze

    def spy_analyze(self, **kwargs):
        seen.append(kwargs.get("device_ids"))
        return original_analyze(self, **kwargs)

    monkeypatch.setattr(InferencePlanner, "analyze", spy_analyze)

    cli.main(["analyze", "--model", model_path, "--runtime", "clitest", "--device-ids", "0,1"])

    assert seen == [(0, 1)]


def test_analyze_command_rejects_malformed_device_ids(capsys):
    with pytest.raises(SystemExit):
        cli.main(["analyze", "--model", "x", "--runtime", "clitest", "--device-ids", "not,numbers"])


def test_analyze_command_probe_flag_forwards_to_planner(tmp_path, monkeypatch):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=1),
    )
    seen = []
    original_analyze = InferencePlanner.analyze

    def spy_analyze(self, **kwargs):
        seen.append((kwargs.get("run_probe"), kwargs.get("probe_timeout_seconds")))
        return original_analyze(self, **kwargs)

    monkeypatch.setattr(InferencePlanner, "analyze", spy_analyze)

    cli.main(["analyze", "--model", model_path, "--runtime", "clitest", "--probe", "--probe-timeout", "30"])

    assert seen == [(True, 30)]


def test_analyze_command_probe_not_forwarded_by_default(tmp_path, monkeypatch):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=1),
    )
    seen = []
    original_analyze = InferencePlanner.analyze

    def spy_analyze(self, **kwargs):
        seen.append(kwargs.get("run_probe"))
        return original_analyze(self, **kwargs)

    monkeypatch.setattr(InferencePlanner, "analyze", spy_analyze)

    cli.main(["analyze", "--model", model_path, "--runtime", "clitest"])

    assert seen == [False]


def test_candidate_versions_flag_prints_summary_per_candidate(tmp_path, monkeypatch, capsys):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=1),
    )

    exit_code = cli.main([
        "analyze", "--model", model_path, "--runtime", "clitest",
        "--candidate-versions", "1.0.0,2.0.0",
    ])

    out = capsys.readouterr().out
    assert "Candidate evaluation for runtime 'clitest'" in out
    assert "1.0.0" in out
    assert "2.0.0" in out
    # _CliFakeAdapter doesn't implement identify_candidate, so both degrade
    # to "not installed" -- and thus incompatible -- without crashing.
    assert exit_code == 2


def test_candidate_versions_json_output_is_a_list(tmp_path, monkeypatch, capsys):
    model_path = _write_local_model(tmp_path)
    monkeypatch.setattr(
        "inference_planner.planner.planner.detect_hardware",
        lambda: make_hardware(gpu_count=1),
    )

    cli.main([
        "analyze", "--model", model_path, "--runtime", "clitest",
        "--candidate-versions", "1.0.0,2.0.0", "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_analyze_command_reports_errors_cleanly(capsys):
    exit_code = cli.main(["analyze", "--model", "/nonexistent/path/xyz", "--runtime", "clitest"])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Error:" in err
