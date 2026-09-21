"""A real, opt-in vLLM runtime probe.

Actually starts vLLM, loads the model, and measures load time / VRAM use /
a rough generation throughput number. This is genuinely expensive (real
model download+load, real GPU memory, potentially minutes of wall time) —
it must never run unless the caller explicitly asked for it (see
``run_probe`` on :meth:`inference_planner.planner.planner.InferencePlanner.analyze`).

Runs vLLM in a **subprocess**, not in this process: a CUDA OOM, a crash, or
vLLM's own multiprocessing setup must never be able to take down the
caller's process. The child receives its config via a temp JSON file (never
via interpolated shell/Python source) and reports back a single JSON line
on stdout.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from inference_planner.core.enums import AnalysisStage, Confidence
from inference_planner.hardware.base import HardwareInfo
from inference_planner.models.base import ModelInfo
from inference_planner.validation.base import RuntimeValidationResult, RuntimeValidator

logger = logging.getLogger(__name__)

# Reads its config from the JSON file named in argv[1] and writes exactly one
# JSON result line to stdout. Deliberately minimal: this is a load+smoke-test
# probe, not a benchmarking harness (see PROBE_LIMITATIONS below).
_PROBE_SCRIPT = r"""
import json, sys, time

def fail(message):
    print(json.dumps({"success": False, "error": message}))
    sys.exit(1)

try:
    with open(sys.argv[1]) as f:
        config = json.load(f)
except Exception as exc:
    fail(f"could not read probe config: {exc}")

try:
    from vllm import LLM, SamplingParams
except Exception as exc:
    fail(f"could not import vllm: {exc}")

devices = config.get("devices") or []
if devices:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)

try:
    start = time.time()
    llm = LLM(
        model=config["model"],
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        dtype=config.get("dtype", "auto"),
        max_model_len=config.get("max_model_len"),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        trust_remote_code=True,
    )
    load_time_seconds = time.time() - start
except Exception as exc:
    fail(f"failed to load model: {exc}")

vram_gb = None
try:
    import torch
    if torch.cuda.is_available():
        vram_bytes = sum(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        )
        vram_gb = vram_bytes / (1024 ** 3)
except Exception:
    pass  # best-effort only

tokens_per_second = None
try:
    gen_start = time.time()
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=16))
    gen_seconds = time.time() - gen_start
    generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    if gen_seconds > 0:
        tokens_per_second = generated_tokens / gen_seconds
except Exception as exc:
    # Load succeeded even if generation didn't -- still real, useful signal.
    print(json.dumps({
        "success": True,
        "load_time_seconds": load_time_seconds,
        "vram_gb": vram_gb,
        "tokens_per_second": None,
        "generation_error": str(exc),
    }))
    sys.exit(0)

print(json.dumps({
    "success": True,
    "load_time_seconds": load_time_seconds,
    "vram_gb": vram_gb,
    "tokens_per_second": tokens_per_second,
}))
"""

PROBE_LIMITATIONS = (
    "This probe measures model load time, approximate peak VRAM, and a "
    "rough single-request generation rate. It does NOT measure "
    "time-to-first-token (that requires the streaming/async API) or "
    "steady-state concurrent throughput.",
)


class VLLMSubprocessValidator(RuntimeValidator):
    """Runs the vLLM probe script in a subprocess and parses its JSON result."""

    def validate(
        self,
        model: ModelInfo,
        hardware: HardwareInfo,
        config: dict,
        *,
        timeout_seconds: int = 600,
    ) -> RuntimeValidationResult:
        if config is None:
            return _unknown_result(("No deployment config was available to probe with.",))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "probe_config.json"
            config_path.write_text(json.dumps(config))

            try:
                result = subprocess.run(
                    [sys.executable, "-c", _PROBE_SCRIPT, str(config_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return _unknown_result((f"Probe timed out after {timeout_seconds}s.",))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to launch vLLM probe subprocess", exc_info=True)
                return _unknown_result((f"Failed to launch probe subprocess: {exc}",))

        payload = _parse_last_json_line(result.stdout)
        if payload is None:
            stderr_tail = "\n".join(result.stderr.strip().splitlines()[-20:])
            return _unknown_result((
                f"Probe process exited with code {result.returncode} and produced no "
                f"parseable output. stderr tail:\n{stderr_tail}",
            ))

        if not payload.get("success"):
            return _unknown_result((f"Probe failed: {payload.get('error', 'unknown error')}",))

        notes = list(PROBE_LIMITATIONS)
        if payload.get("generation_error"):
            notes.append(f"Model loaded successfully, but generation failed: {payload['generation_error']}")

        return RuntimeValidationResult(
            stage=AnalysisStage.RUNTIME_VALIDATION,
            confidence=Confidence.MEASURED,
            model_load_time_seconds=payload.get("load_time_seconds"),
            actual_vram_usage_gb=payload.get("vram_gb"),
            time_to_first_token_ms=None,
            tokens_per_second=payload.get("tokens_per_second"),
            max_observed_concurrency=None,
            notes=tuple(notes),
        )


def _unknown_result(notes: tuple[str, ...]) -> RuntimeValidationResult:
    return RuntimeValidationResult(
        stage=AnalysisStage.RUNTIME_VALIDATION,
        confidence=Confidence.UNKNOWN,
        notes=notes,
    )


def _parse_last_json_line(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None
