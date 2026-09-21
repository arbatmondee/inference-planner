# inference-planner

A hardware + model + inference-runtime **analysis and planning** library for
LLM inference platforms. It answers one question:

> Given this model, this machine, and this inference engine — can I run it,
> where, and with what configuration?

It is a standalone library, not a scheduler or an inference server. Feed its
`AnalysisReport` into whatever deploys your models.

## What it does

```
Inference Platform
        |
        v
inference_planner.InferencePlanner.analyze(model=..., runtime=...)
        |
        +---- hardware/       vendor-agnostic hardware discovery
        +---- models/         model metadata inspection (HF config-driven)
        +---- runtimes/       engine adapters (vLLM today; pluggable)
        +---- compatibility/  rule-based compatibility verdicts with reasons
        +---- resources/      memory/concurrency estimation, confidence-tagged
        +---- planner/        orchestrates the above
        |
        v
AnalysisReport (JSON-serializable)
        |
        v
Your scheduler decides whether/how to deploy
```

## Design principles

- **No hardcoded model/GPU tables.** Nothing branches on `if model == "llama"`
  or `if gpu == "A100"`. Model facts come from the model's own `config.json`;
  GPU facts (including precision support) come from compute capability, not
  a name lookup.
- **Everything is a plugin.** Hardware providers, model providers, runtime
  adapters, and compatibility rules are all registered in a small generic
  `Registry`. Add AMD/SGLang/TensorRT-LLM support by registering a new
  adapter — the planner, compatibility engine, and CLI don't change.
- **Estimates are never presented as measurements.** Every numeric result
  carries a `Confidence` (`measured` / `estimated` / `heuristic` / `unknown`)
  and a `basis` explaining how it was derived.
- **Two stages, clearly separated.** `STATIC_ANALYSIS` only reads metadata.
  A `RUNTIME_VALIDATION` stage (actually loading the engine and measuring
  TTFT/throughput/VRAM) is architected via `inference_planner.validation`
  but not implemented yet.

## Install

```bash
pip install -e ".[dev,huggingface]"
# add [nvidia] for pynvml-based GPU detection, [vllm] to actually run vLLM
```

## Usage

```python
from inference_planner import InferencePlanner

report = InferencePlanner().analyze(model="Qwen/Qwen3-8B", runtime="vllm")

print(report)                 # human-readable
report.to_json()              # JSON for your platform's scheduler

if report.compatible:
    plan = report.deployment_plan   # e.g. {"tensor_parallel_size": 2, "dtype": "bfloat16", ...}
else:
    for error in report.compatibility.errors:
        print(error.code, error.message)
```

## CLI

```bash
inference-planner hardware
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm --json
```

Exit codes: `0` compatible, `2` incompatible, `1` on error (e.g. model not found).

## Extending

**A new hardware vendor:**

```python
from inference_planner.hardware import register_provider, HardwareProvider

class AMDHardwareProvider(HardwareProvider):
    def is_available(self) -> bool: ...
    def detect_gpus(self) -> list[GPUInfo]: ...

register_provider(AMDHardwareProvider())
```

**A new runtime engine:** implement `RuntimeAdapter` (`identify`,
`capabilities`, `overhead_profile`, `estimate_resources`, `generate_config`)
and `register_adapter("sglang", SGLangAdapter())`.

**A new compatibility rule:** implement `Rule.check(ctx) -> CompatibilityIssue | None`
and `register_rule(MyRule())`.

## What can and can't be determined statically

Discoverable automatically today: GPU name/VRAM/compute-capability/driver
(via `pynvml` or `nvidia-smi`), CPU/RAM, and — from a model's HF
`config.json`/safetensors index — architecture, layer/head dimensions,
context length, quantization, and dtype.

Not reliable without actually running the model: real VRAM usage under load,
time-to-first-token, tokens/sec, true concurrency limits. These belong to the
future runtime-validation stage (`inference_planner.validation`), and static
estimates are always confidence-tagged so callers don't mistake one for the
other.

## Testing

```bash
pytest                      # unit tests, fully mocked, no GPU/vLLM required
```

Tests for hardware/runtime detection use mocked providers/adapters so the
suite runs on any machine. A small number of integration tests are gated to
run only when real GPU hardware or vLLM is present (none are currently
required to pass CI).
