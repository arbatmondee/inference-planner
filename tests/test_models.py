from __future__ import annotations

import json

import pytest

from inference_planner.core.enums import Confidence
from inference_planner.core.exceptions import ModelInspectionError, NoCompatibleAdapterError
from inference_planner.models.analyzer import ModelAnalyzer
from inference_planner.models.base import ModelFormat
from inference_planner.models.huggingface import HuggingFaceModelProvider


def _write_config(path, config: dict) -> None:
    (path / "config.json").write_text(json.dumps(config))


def test_inspect_local_model_reads_generic_config_fields(tmp_path):
    _write_config(tmp_path, {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "vocab_size": 152064,
        "torch_dtype": "bfloat16",
    })
    (tmp_path / "model.safetensors").touch()

    info = ModelAnalyzer().inspect(str(tmp_path))

    assert info.architecture == "Qwen3ForCausalLM"
    assert info.model_type == "qwen3"
    assert info.task == "text-generation"
    assert info.format == ModelFormat.SAFETENSORS
    assert info.context_length == 32768
    assert info.num_layers == 32
    assert info.attention.num_key_value_heads == 8
    assert info.attention.uses_grouped_query_attention is True
    assert info.parameter_count.confidence == Confidence.HEURISTIC
    assert info.parameter_count.count is not None and info.parameter_count.count > 0


def test_parameter_count_uses_safetensors_index_when_available(tmp_path):
    _write_config(tmp_path, {
        "architectures": ["MockForCausalLM"],
        "model_type": "mock",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "torch_dtype": "bfloat16",
    })
    total_bytes = 14_000_000_000  # 7B params * 2 bytes/param
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}})
    )

    info = ModelAnalyzer().inspect(str(tmp_path))

    assert info.parameter_count.confidence == Confidence.ESTIMATED
    assert info.parameter_count.count == 7_000_000_000


def test_quantization_config_is_parsed_generically(tmp_path):
    _write_config(tmp_path, {
        "architectures": ["MockForCausalLM"],
        "model_type": "mock",
        "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
    })

    info = ModelAnalyzer().inspect(str(tmp_path))

    assert info.quantization.is_quantized
    assert info.quantization.method == "awq"
    assert info.quantization.bits == 4
    assert info.quantization.group_size == 128


class _RecordingHfApi:
    """Stand-in for huggingface_hub.HfApi that just records the token it was built with."""

    def __init__(self, token, sink):
        sink.append(token)

    def model_info(self, identifier, files_metadata=False):
        return _FakeModelInfo()


class _FakeModelInfo:
    siblings: list = []
    pipeline_tag = None


def _patch_hub(monkeypatch, tmp_path, seen_tokens):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["MockForCausalLM"], "model_type": "mock"})
    )
    config_path = str(tmp_path / "config.json")

    def fake_hub_download(*, repo_id, filename, token):
        seen_tokens.append(token)
        return config_path if filename == "config.json" else None

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hub_download)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: _RecordingHfApi(token, seen_tokens))


def test_token_is_threaded_through_to_hub_calls(tmp_path, monkeypatch):
    seen_tokens: list = []
    _patch_hub(monkeypatch, tmp_path, seen_tokens)
    provider = HuggingFaceModelProvider()

    provider.inspect("org/some-model", token="secret-token-123")

    assert "secret-token-123" in seen_tokens
    assert all(t in (None, "secret-token-123") for t in seen_tokens)


def test_instance_level_token_used_when_no_per_call_token_given(tmp_path, monkeypatch):
    seen_tokens: list = []
    _patch_hub(monkeypatch, tmp_path, seen_tokens)
    provider = HuggingFaceModelProvider(token="instance-default-token")

    provider.inspect("org/some-model")

    assert "instance-default-token" in seen_tokens


def test_per_call_token_overrides_instance_default(tmp_path, monkeypatch):
    seen_tokens: list = []
    _patch_hub(monkeypatch, tmp_path, seen_tokens)
    provider = HuggingFaceModelProvider(token="instance-default-token")

    provider.inspect("org/some-model", token="override-token")

    assert "override-token" in seen_tokens
    assert "instance-default-token" not in seen_tokens


def test_missing_config_raises_clear_error(tmp_path):
    provider = HuggingFaceModelProvider()
    with pytest.raises(ModelInspectionError):
        provider.inspect(str(tmp_path))


def test_no_provider_can_handle_raises(monkeypatch):
    from inference_planner.core.registry import Registry
    from inference_planner.models.base import ModelProvider

    empty_registry: Registry[ModelProvider] = Registry()
    analyzer = ModelAnalyzer(providers=empty_registry)

    with pytest.raises(NoCompatibleAdapterError):
        analyzer.inspect("anything")


def test_unknown_format_when_no_recognized_weight_files(tmp_path):
    _write_config(tmp_path, {"architectures": ["MockForCausalLM"], "model_type": "mock"})

    info = ModelAnalyzer().inspect(str(tmp_path))

    assert info.format == ModelFormat.UNKNOWN
