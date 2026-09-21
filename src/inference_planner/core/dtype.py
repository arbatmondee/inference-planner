"""Shared dtype -> bytes-per-element table.

Used both when inferring parameter counts from on-disk weight bytes
(:mod:`inference_planner.models.huggingface`) and when estimating memory
footprints (:mod:`inference_planner.resources.estimator`). Centralized so
the two stay consistent instead of drifting.
"""

from __future__ import annotations

_BYTES_PER_ELEMENT = {
    "float32": 4.0, "fp32": 4.0, "float": 4.0,
    "float16": 2.0, "fp16": 2.0, "half": 2.0,
    "bfloat16": 2.0, "bf16": 2.0,
    "int8": 1.0, "uint8": 1.0,
    "float8": 1.0, "fp8": 1.0, "float8_e4m3fn": 1.0, "float8_e5m2": 1.0,
    "int4": 0.5, "fp4": 0.5, "nf4": 0.5,
}


def bytes_per_element(dtype: str | None) -> float | None:
    if not dtype:
        return None
    key = dtype.lower().replace("torch.", "")
    return _BYTES_PER_ELEMENT.get(key)


def bytes_per_element_for_quantization(bits: int | None) -> float | None:
    if bits is None:
        return None
    return bits / 8.0
