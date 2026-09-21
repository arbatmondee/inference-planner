from __future__ import annotations

import importlib.metadata
from unittest.mock import patch

import pytest

from inference_planner.core.exceptions import NoCompatibleAdapterError
from inference_planner.runtimes.base import RuntimeAdapter, RuntimeCapabilities, RuntimeIdentity
from inference_planner.runtimes.registry import available_adapters, get_adapter, register_adapter
from inference_planner.runtimes.vllm import VLLMAdapter
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

        def estimate_resources(self, model, hardware, *, tensor_parallel_size=None):
            raise NotImplementedError

        def generate_config(self, model, hardware, *, tensor_parallel_size=None):
            return {}

    register_adapter("noop", _NoopAdapter())
    assert isinstance(get_adapter("noop"), _NoopAdapter)
