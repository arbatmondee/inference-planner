"""Model analysis entry point: dispatches to the first provider that claims an identifier."""

from __future__ import annotations

from inference_planner.core.exceptions import NoCompatibleAdapterError
from inference_planner.core.registry import Registry
from inference_planner.models.base import ModelInfo, ModelProvider
from inference_planner.models.huggingface import HuggingFaceModelProvider

_registry: Registry[ModelProvider] = Registry()
_registry.register(HuggingFaceModelProvider())


def register_provider(provider: ModelProvider, *, priority: bool = False) -> None:
    """Register a model provider (e.g. a future GGUF-native or ONNX provider)."""
    _registry.register(provider, priority=priority)


class ModelAnalyzer:
    """Inspects a model identifier using whichever registered provider claims it."""

    def __init__(self, providers: Registry[ModelProvider] | None = None) -> None:
        self._providers = providers if providers is not None else _registry

    def inspect(self, identifier: str, *, token: str | None = None) -> ModelInfo:
        for provider in self._providers:
            if provider.can_handle(identifier):
                return provider.inspect(identifier, token=token)
        raise NoCompatibleAdapterError(f"No registered model provider can handle '{identifier}'")


def inspect(identifier: str, *, token: str | None = None) -> ModelInfo:
    """Convenience module-level function: ``model_analyzer.inspect(...)``."""
    return ModelAnalyzer().inspect(identifier, token=token)
