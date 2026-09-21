"""Hugging Face Hub / local-checkout model provider.

Reads ``config.json`` (and, when available, ``*.safetensors.index.json`` and
``tokenizer_config.json``) rather than hard-coding per-model-family logic.
Field names like ``num_hidden_layers`` vary a little across architectures
(e.g. GPT-2 uses ``n_layer``), so this module normalizes a handful of known
*generic* aliases — that is a schema-compatibility shim, not model-specific
business logic, and the alias table only grows with new config schemas, not
new model names.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from inference_planner.core.dtype import bytes_per_element
from inference_planner.core.enums import Confidence
from inference_planner.core.exceptions import ModelInspectionError
from inference_planner.models.base import (
    AttentionConfig,
    ModelFormat,
    ModelInfo,
    ModelProvider,
    ParameterCountEstimate,
    QuantizationInfo,
    TokenizerInfo,
)

logger = logging.getLogger(__name__)

# Generic cross-architecture config key aliases (not per-model-name mappings).
_CONTEXT_LENGTH_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length")
_NUM_LAYERS_KEYS = ("num_hidden_layers", "n_layer", "num_layers")
_HIDDEN_SIZE_KEYS = ("hidden_size", "n_embd", "d_model")
_NUM_HEADS_KEYS = ("num_attention_heads", "n_head")
_NUM_KV_HEADS_KEYS = ("num_key_value_heads", "num_kv_heads", "n_kv_heads")
_INTERMEDIATE_SIZE_KEYS = ("intermediate_size", "n_inner", "ffn_dim")
_VOCAB_SIZE_KEYS = ("vocab_size",)


class HuggingFaceModelProvider(ModelProvider):
    """Inspects a Hugging Face Hub repo id or a local model directory."""

    def __init__(self, *, token: str | None = None) -> None:
        self._token = token

    def can_handle(self, identifier: str) -> bool:
        # Local directory containing config.json, or anything shaped like a
        # (possibly namespaced) HF repo id. This provider is intentionally the
        # generic fallback; more specific providers should register with
        # priority=True to shadow it.
        if os.path.isdir(identifier):
            return os.path.isfile(os.path.join(identifier, "config.json"))
        return True

    def inspect(self, identifier: str, *, token: str | None = None) -> ModelInfo:
        token = token if token is not None else self._token
        config, source_files = self._load_config(identifier, token)
        tokenizer_config = self._load_tokenizer_config(identifier, token)
        weight_index = self._load_safetensors_index(identifier, token)
        model_format = self._detect_format(identifier, source_files)
        task = self._infer_task(identifier, config, token)

        dtype = config.get("torch_dtype")
        num_layers = _first_present(config, _NUM_LAYERS_KEYS)
        hidden_size = _first_present(config, _HIDDEN_SIZE_KEYS)
        num_heads = _first_present(config, _NUM_HEADS_KEYS)
        num_kv_heads = _first_present(config, _NUM_KV_HEADS_KEYS) or num_heads
        head_dim = config.get("head_dim") or (
            hidden_size // num_heads if hidden_size and num_heads else None
        )

        attention = AttentionConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )

        quant_cfg = config.get("quantization_config") or {}
        quantization = QuantizationInfo(
            method=quant_cfg.get("quant_method"),
            bits=quant_cfg.get("bits") or quant_cfg.get("weight_bits"),
            group_size=quant_cfg.get("group_size"),
            raw_config=quant_cfg,
        )

        parameter_count = _estimate_parameter_count(
            config=config,
            dtype=dtype,
            weight_index=weight_index,
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=_first_present(config, _INTERMEDIATE_SIZE_KEYS),
            vocab_size=_first_present(config, _VOCAB_SIZE_KEYS),
        )

        tokenizer = TokenizerInfo(
            vocab_size=_first_present(config, _VOCAB_SIZE_KEYS),
            model_max_length=tokenizer_config.get("model_max_length"),
            has_chat_template=bool(tokenizer_config.get("chat_template")),
        )

        architectures = config.get("architectures") or []

        return ModelInfo(
            identifier=identifier,
            architecture=architectures[0] if architectures else None,
            model_type=config.get("model_type"),
            task=task,
            parameter_count=parameter_count,
            format=model_format,
            quantization=quantization,
            dtype=dtype,
            context_length=_first_present(config, _CONTEXT_LENGTH_KEYS),
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=_first_present(config, _INTERMEDIATE_SIZE_KEYS),
            vocab_size=_first_present(config, _VOCAB_SIZE_KEYS),
            attention=attention,
            tokenizer=tokenizer,
            raw_config=config,
        )

    # -- loading helpers -----------------------------------------------------

    def _load_config(self, identifier: str, token: str | None) -> tuple[dict[str, Any], list[str]]:
        if os.path.isdir(identifier):
            config_path = Path(identifier) / "config.json"
            if not config_path.is_file():
                raise ModelInspectionError(f"No config.json found in local path: {identifier}")
            files = [p.name for p in Path(identifier).iterdir() if p.is_file()]
            return json.loads(config_path.read_text()), files

        path = self._hub_download(identifier, "config.json", token)
        if path is None:
            raise ModelInspectionError(
                f"Could not fetch config.json for '{identifier}'. Install the "
                "'huggingface' extra (pip install inference-planner[huggingface]), "
                "verify the identifier/network access, and if this is a gated or "
                "private repo, pass a valid Hugging Face token (or set HF_TOKEN)."
            )
        files = self._list_hub_files(identifier, token)
        return json.loads(Path(path).read_text()), files

    def _load_tokenizer_config(self, identifier: str, token: str | None) -> dict[str, Any]:
        if os.path.isdir(identifier):
            path = Path(identifier) / "tokenizer_config.json"
            if path.is_file():
                return json.loads(path.read_text())
            return {}
        path = self._hub_download(identifier, "tokenizer_config.json", token)
        if path is None:
            return {}
        try:
            return json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_safetensors_index(self, identifier: str, token: str | None) -> dict[str, Any] | None:
        filename = "model.safetensors.index.json"
        if os.path.isdir(identifier):
            path = Path(identifier) / filename
            if path.is_file():
                return json.loads(path.read_text())
            return None
        path = self._hub_download(identifier, filename, token)
        if path is None:
            return None
        try:
            return json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _detect_format(self, identifier: str, files: list[str]) -> ModelFormat:
        lowered = [f.lower() for f in files]
        if any(f.endswith(".safetensors") or f.endswith(".safetensors.index.json") for f in lowered):
            return ModelFormat.SAFETENSORS
        if any(f.endswith(".gguf") for f in lowered):
            return ModelFormat.GGUF
        if any(f.endswith(".bin") for f in lowered):
            return ModelFormat.PYTORCH_BIN
        return ModelFormat.UNKNOWN

    def _infer_task(self, identifier: str, config: dict[str, Any], token: str | None) -> str | None:
        pipeline_tag = self._hub_pipeline_tag(identifier, token)
        if pipeline_tag:
            return pipeline_tag
        architectures = config.get("architectures") or []
        if architectures:
            name = architectures[0]
            if name.endswith("ForCausalLM"):
                return "text-generation"
            if name.endswith("ForConditionalGeneration"):
                return "text2text-generation"
            if name.endswith("ForSequenceClassification"):
                return "text-classification"
        return None

    # -- Hub access (isolated so it degrades gracefully without huggingface_hub) --

    def _hub_download(self, identifier: str, filename: str, token: str | None) -> str | None:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        except ImportError:
            return None
        try:
            return hf_hub_download(repo_id=identifier, filename=filename, token=token)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception:
            logger.debug("hf_hub_download(%s, %s) failed", identifier, filename, exc_info=True)
            return None

    def _list_hub_files(self, identifier: str, token: str | None) -> list[str]:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return []
        try:
            info = HfApi(token=token).model_info(identifier, files_metadata=False)
            return [s.rfilename for s in info.siblings]
        except Exception:
            logger.debug("HfApi.model_info(%s) failed", identifier, exc_info=True)
            return []

    def _hub_pipeline_tag(self, identifier: str, token: str | None) -> str | None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return None
        try:
            info = HfApi(token=token).model_info(identifier)
            return info.pipeline_tag
        except Exception:
            return None


def _first_present(config: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return int(value)
    return None


def _estimate_parameter_count(
    *,
    config: dict[str, Any],
    dtype: str | None,
    weight_index: dict[str, Any] | None,
    hidden_size: int | None,
    num_layers: int | None,
    intermediate_size: int | None,
    vocab_size: int | None,
) -> ParameterCountEstimate:
    """Prefer measured on-disk weight bytes; fall back to a formula heuristic."""
    if weight_index is not None:
        total_bytes = (weight_index.get("metadata") or {}).get("total_size")
        bytes_per_param = bytes_per_element(dtype)
        if total_bytes and bytes_per_param:
            count = int(total_bytes / bytes_per_param)
            return ParameterCountEstimate(
                count=count,
                confidence=Confidence.ESTIMATED,
                basis=("safetensors index total_size", f"dtype={dtype}"),
            )

    if hidden_size and num_layers:
        intermediate = intermediate_size or hidden_size * 4
        vocab = vocab_size or 0
        # Generic dense-transformer parameter formula (attention + MLP per layer
        # + embeddings). Deliberately architecture-agnostic; it will overshoot
        # for MoE models, which is why this is HEURISTIC, not ESTIMATED.
        per_layer = 4 * hidden_size * hidden_size + 2 * hidden_size * intermediate
        count = num_layers * per_layer + 2 * vocab * hidden_size
        return ParameterCountEstimate(
            count=count,
            confidence=Confidence.HEURISTIC,
            basis=(
                "hidden_size", "num_layers", "intermediate_size", "vocab_size",
                "generic dense-transformer formula (inaccurate for MoE/non-standard architectures)",
            ),
        )

    return ParameterCountEstimate(count=None, confidence=Confidence.UNKNOWN, basis=())
