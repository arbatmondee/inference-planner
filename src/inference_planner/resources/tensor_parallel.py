"""Shared tensor-parallel head-divisibility logic.

Used both by the resource estimator (to only ever *recommend* a TP degree
that could actually work) and by the compatibility rules (to reject an
explicitly-requested TP degree that can't). Centralized so the two can't
drift apart.
"""

from __future__ import annotations

from inference_planner.models.base import AttentionConfig


def tp_compatible_with_heads(attention: AttentionConfig, tensor_parallel_size: int) -> bool:
    """Whether attention head sharding across ``tensor_parallel_size`` ranks is structurally valid.

    Unknown head counts are treated as "not disqualifying" (``True``) — this
    is a necessary-condition check, not a substitute for actually knowing.
    """
    if tensor_parallel_size <= 1:
        return True

    heads = attention.num_attention_heads
    if heads is not None and heads % tensor_parallel_size != 0:
        return False

    kv_heads = attention.num_key_value_heads
    if kv_heads is not None and kv_heads > 0:
        # Either the KV heads split evenly across ranks, or each rank
        # replicates the (fewer) KV heads it doesn't have a full share of.
        if kv_heads >= tensor_parallel_size:
            if kv_heads % tensor_parallel_size != 0:
                return False
        elif tensor_parallel_size % kv_heads != 0:
            return False

    return True
