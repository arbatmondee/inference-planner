from __future__ import annotations

import importlib.metadata
from unittest.mock import patch

import pytest

from inference_planner.core.exceptions import NoCompatibleAdapterError
from inference_planner.runtimes.base import (
    RuntimeAdapter,
    RuntimeCandidate,
    RuntimeCapabilities,
    RuntimeIdentity,
)
from inference_planner.runtimes.registry import available_adapters, get_adapter, register_adapter
from inference_planner.runtimes.vllm import VLLMAdapter, _pypi_version_exists
from tests.conftest import make_hardware, make_model


def test_identify_reports_not_installed_when_package_missing():
    adapter = VLLMAdapter()
    with patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        identity = adapter.identify()
    assert identity.installed is False
    assert identity.version is None


def test_identify_reports_installed_version():
    adapter = VLLMAdapter()
    with patch("importlib.metadata.version", return_value="0.6.3"):
        identity = adapter.identify()
    assert identity.installed is True
    assert identity.version == "0.6.3"


def test_capabilities_empty_when_not_installed():
    adapter = VLLMAdapter()
    with patch.object(VLLMAdapter, "identify", return_value=RuntimeIdentity(name="vllm", installed=False)):
        caps = adapter.capabilities()
    assert caps == RuntimeCapabilities()


def test_capabilities_introspects_installed_engine_signature():
    adapter = VLLMAdapter()

    class _FakeEngineArgs:
        def __init__(self, tensor_parallel_size=1, pipeline_parallel_size=1, enable_chunked_prefill=False):
            pass

    with patch.object(VLLMAdapter, "identify", return_value=RuntimeIdentity(name="vllm", installed=True, version="0.6.3")), \
         patch.object(VLLMAdapter, "_import_engine_args", return_value=_FakeEngineArgs), \
         patch.object(VLLMAdapter, "_quantization_methods", return_value=["awq", "gptq"]):
        caps = adapter.capabilities()

    assert caps.supports_tensor_parallel is True
    assert caps.supports_pipeline_parallel is True
    assert caps.supports_chunked_prefill is True
    assert "awq" in caps.supported_quantization_methods


def test_generate_config_produces_engine_native_dict():
    adapter = VLLMAdapter()
    model = make_model()
    hardware = make_hardware(gpu_count=1)

    config = adapter.generate_config(model, hardware)

    assert config["runtime"] == "vllm"
    assert config["tensor_parallel_size"] >= 1
    assert config["dtype"] == "bfloat16"
    assert isinstance(config["max_model_len"], int)
    assert 0 < config["gpu_memory_utilization"] <= 1


def test_supported_architectures_introspected_from_model_registry():
    adapter = VLLMAdapter()

    class _FakeModelRegistry:
        @staticmethod
        def get_supported_archs():
            return ["LlamaForCausalLM", "Qwen2ForCausalLM"]

    fake_module = type("FakeModule", (), {"ModelRegistry": _FakeModelRegistry})

    with patch("importlib.import_module", return_value=fake_module):
        archs = adapter._supported_architectures()

    assert archs == ("LlamaForCausalLM", "Qwen2ForCausalLM")


def test_supported_architectures_empty_when_registry_not_introspectable():
    adapter = VLLMAdapter()
    with patch("importlib.import_module", side_effect=ImportError):
        archs = adapter._supported_architectures()
    assert archs == ()


def test_declared_requirements_parses_python_and_torch_specifiers():
    adapter = VLLMAdapter()

    class _FakeMetadata(dict):
        def get(self, key, default=None):
            return {"Requires-Python": ">=3.9"}.get(key, default)

    with patch("importlib.metadata.metadata", return_value=_FakeMetadata()), \
         patch("importlib.metadata.requires", return_value=["torch>=2.0,<2.5", "numpy"]):
        python_spec, torch_spec = adapter._declared_requirements()

    assert python_spec == ">=3.9"
    assert torch_spec == "<2.5,>=2.0"


def test_declared_requirements_ignores_extras_only_torch_dependency():
    adapter = VLLMAdapter()

    class _FakeMetadata(dict):
        def get(self, key, default=None):
            return {"Requires-Python": ">=3.9"}.get(key, default)

    with patch("importlib.metadata.metadata", return_value=_FakeMetadata()), \
         patch("importlib.metadata.requires", return_value=['torch>=2.0; extra == "vision"']):
        _python_spec, torch_spec = adapter._declared_requirements()

    assert torch_spec is None


def test_identify_candidate_returns_installed_identity_for_exact_version_match():
    adapter = VLLMAdapter()
    installed = RuntimeIdentity(name="vllm", installed=True, version="0.6.3")
    with patch.object(VLLMAdapter, "identify", return_value=installed):
        identity = adapter.identify_candidate(RuntimeCandidate(name="vllm", version="0.6.3"))
    assert identity is installed


def test_identify_candidate_checks_pypi_for_other_versions():
    adapter = VLLMAdapter()
    not_installed = RuntimeIdentity(name="vllm", installed=False)
    with patch.object(VLLMAdapter, "identify", return_value=not_installed), \
         patch("inference_planner.runtimes.vllm._pypi_version_exists", return_value=True):
        identity = adapter.identify_candidate(RuntimeCandidate(name="vllm", version="0.5.0"))
    assert identity.installed is True
    assert identity.version == "0.5.0"


def test_identify_candidate_reports_unavailable_when_version_not_on_pypi():
    adapter = VLLMAdapter()
    not_installed = RuntimeIdentity(name="vllm", installed=False)
    with patch.object(VLLMAdapter, "identify", return_value=not_installed), \
         patch("inference_planner.runtimes.vllm._pypi_version_exists", return_value=False):
        identity = adapter.identify_candidate(RuntimeCandidate(name="vllm", version="999.999.999"))
    assert identity.installed is False


def test_identify_candidate_does_not_guess_when_lookup_fails():
    adapter = VLLMAdapter()
    not_installed = RuntimeIdentity(name="vllm", installed=False)
    with patch.object(VLLMAdapter, "identify", return_value=not_installed), \
         patch("inference_planner.runtimes.vllm._pypi_version_exists", return_value=None):
        identity = adapter.identify_candidate(RuntimeCandidate(name="vllm", version="0.5.0"))
    assert identity.installed is False
    assert "Could not verify" in identity.detection_notes[0]


def test_capabilities_for_candidate_uses_override_when_supplied():
    adapter = VLLMAdapter()
    override = RuntimeCapabilities(supported_dtypes=("fp16",))
    candidate = RuntimeCandidate(name="vllm", version="0.1.0", capabilities_override=override)
    assert adapter.capabilities_for_candidate(candidate) is override


def test_capabilities_for_candidate_falls_back_to_conservative_baseline():
    adapter = VLLMAdapter()
    not_installed = RuntimeIdentity(name="vllm", installed=False)
    with patch.object(VLLMAdapter, "identify", return_value=not_installed):
        caps = adapter.capabilities_for_candidate(RuntimeCandidate(name="vllm", version="0.5.0"))
    assert "bfloat16" in caps.supported_dtypes
    assert caps.supported_architectures == ()  # never claimed for an un-introspected version


def test_pypi_version_exists_returns_none_on_network_failure():
    with patch("urllib.request.urlopen", side_effect=OSError("no network")):
        assert _pypi_version_exists("vllm", "0.6.3") is None


def test_registry_lookup_and_registration():
    assert "vllm" in available_adapters()
    assert isinstance(get_adapter("vllm"), VLLMAdapter)

    with pytest.raises(NoCompatibleAdapterError):
        get_adapter("does-not-exist")

    class _NoopAdapter(RuntimeAdapter):
        def identify(self):
            return RuntimeIdentity(name="noop", installed=True, version="1.0")

        def capabilities(self):
            return RuntimeCapabilities()

        def overhead_profile(self):
            from inference_planner.resources.types import OverheadProfile
            return OverheadProfile(fixed_overhead_gb=0.0)

        def estimate_resources(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
            raise NotImplementedError

        def generate_config(self, model, hardware, *, tensor_parallel_size=None, device_ids=None):
            return {}

    register_adapter("noop", _NoopAdapter())
    assert isinstance(get_adapter("noop"), _NoopAdapter)
