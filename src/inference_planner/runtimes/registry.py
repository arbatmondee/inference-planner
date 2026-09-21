"""Runtime adapter lookup by name."""

from __future__ import annotations

from inference_planner.core.exceptions import NoCompatibleAdapterError
from inference_planner.core.registry import Registry
from inference_planner.runtimes.base import RuntimeAdapter
from inference_planner.runtimes.vllm import VLLMAdapter

_registry: Registry[tuple[str, RuntimeAdapter]] = Registry()
_registry.register(("vllm", VLLMAdapter()))


def register_adapter(name: str, adapter: RuntimeAdapter, *, priority: bool = False) -> None:
    """Register a runtime adapter under a name (e.g. "sglang", "tensorrt_llm")."""
    _registry.register((name, adapter), priority=priority)


def get_adapter(name: str) -> RuntimeAdapter:
    for registered_name, adapter in _registry:
        if registered_name == name:
            return adapter
    raise NoCompatibleAdapterError(f"No registered runtime adapter named '{name}'")


def available_adapters() -> list[str]:
    return [name for name, _ in _registry]
