"""Model data model and provider protocol.

Fields are populated from whatever metadata a model actually publishes
(config.json, tokenizer config, safetensors headers, ...). Nothing here is
keyed off a specific model family name; ``model_type``/``architectures``
are read verbatim from the model's own config and used as generic lookup
keys elsewhere (e.g. compatibility rules), not branched on by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from inference_planner.core.enums import Confidence


class ModelFormat(str, Enum):
    SAFETENSORS = "safetensors"
    PYTORCH_BIN = "pytorch_bin"
    GGUF = "gguf"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QuantizationInfo:
    method: str | None = None
    """Quantization method as reported by the model, e.g. "gptq", "awq", "bitsandbytes"."""
    bits: int | None = None
    group_size: int | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_quantized(self) -> bool:
        return self.method is not None


@dataclass(frozen=True)
class AttentionConfig:
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    """Distinct from num_attention_heads when the model uses GQA/MQA."""
    head_dim: int | None = None

    @property
    def uses_grouped_query_attention(self) -> bool | None:
        if self.num_attention_heads is None or self.num_key_value_heads is None:
            return None
        return self.num_key_value_heads < self.num_attention_heads


@dataclass(frozen=True)
class TokenizerInfo:
    vocab_size: int | None = None
    model_max_length: int | None = None
    has_chat_template: bool = False


@dataclass(frozen=True)
class ParameterCountEstimate:
    count: int | None
    confidence: Confidence
    basis: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelInfo:
    """Structured metadata about a target model."""

    identifier: str
    architecture: str | None
    """Raw architecture class name, e.g. "Qwen3ForCausalLM" (from config.architectures[0])."""
    model_type: str | None
    """The model's own family tag, e.g. config.json's "model_type" field (verbatim, ungoverned)."""
    task: str | None
    """Inferred task, e.g. "text-generation", "automatic-speech-recognition"."""
    parameter_count: ParameterCountEstimate
    format: ModelFormat
    quantization: QuantizationInfo
    dtype: str | None
    context_length: int | None
    num_layers: int | None
    hidden_size: int | None
    intermediate_size: int | None
    vocab_size: int | None
    attention: AttentionConfig
    tokenizer: TokenizerInfo
    raw_config: dict[str, Any] = field(default_factory=dict)
    """The unmodified source config, kept around for adapters/rules that need
    fields this schema doesn't model yet."""

    @property
    def family(self) -> str | None:
        """A coarse family label derived from model_type, not a name lookup table."""
        return self.model_type


class ModelProvider(ABC):
    """A plugin that knows how to fetch/parse metadata for some class of model sources."""

    @abstractmethod
    def can_handle(self, identifier: str) -> bool:
        """Whether this provider knows how to inspect the given identifier/path."""

    @abstractmethod
    def inspect(self, identifier: str, *, token: str | None = None) -> ModelInfo:
        """Fetch and parse metadata for the given model identifier/path.

        ``token`` is a generic per-call credential (e.g. a Hugging Face Hub
        token for gated/private repos). Providers that don't need auth
        simply ignore it.
        """
