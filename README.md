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
- **Two stages, clearly separated.** `STATIC_ANALYSIS` only reads metadata
  and never runs by default. An opt-in `RUNTIME_VALIDATION` stage
  (`run_probe=True` / `--probe`) actually loads the model in a subprocess
  and measures load time/VRAM/a rough generation rate — see
  [DOCUMENTATION.md](DOCUMENTATION.md#real-runtime-validation-opt-in-probe).

## Install

```bash
pip install -e ".[dev,huggingface]"
# add [nvidia] for pynvml-based GPU detection, [vllm] to actually run vLLM
```

## Installing the CLI on another machine

You don't need to clone the repo just to use the CLI. Pick whichever fits:

**With `pip`, straight from GitHub:**

```bash
pip install "inference-planner[huggingface,nvidia] @ git+https://github.com/arbatmondee/inference-planner.git"
```

**With `uv` (installs it as an isolated CLI tool on your PATH — recommended):**

```bash
uv tool install "inference-planner[huggingface,nvidia] @ git+https://github.com/arbatmondee/inference-planner.git"
uv tool update-shell   # first time only, if `inference-planner` isn't found after install
```

**From a copied wheel file** (no network access needed on the target machine):

```bash
python -m build   # run once, on a machine with the source, to produce dist/*.whl
# copy dist/inference_planner-0.1.0-py3-none-any.whl to the target machine, then:
pip install "inference_planner-0.1.0-py3-none-any.whl[huggingface,nvidia]"
# or with uv:
uv tool install "./inference_planner-0.1.0-py3-none-any.whl[huggingface,nvidia]"
```

**Editing the source on the target machine instead of just using the CLI:**

```bash
git clone https://github.com/arbatmondee/inference-planner.git
cd inference-planner
pip install -e ".[dev,huggingface]"     # or: uv pip install -e ".[dev,huggingface]"
```

Verify any of the above with:

```bash
inference-planner hardware
```

If the GitHub repo is private, the target machine needs access to clone/fetch
it (an SSH key with repo access, or an HTTPS URL with a token embedded:
`git+https://<token>@github.com/arbatmondee/inference-planner.git`) — except
for the wheel-file option, which needs no repo access at all.

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
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm --device-ids 2,3
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm --candidate-versions 0.6.3,0.6.2
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm --probe   # expensive, opt-in: actually loads the model
```

Exit codes: `0` compatible, `2` incompatible, `1` on error (e.g. model not found). Full flag reference in [DOCUMENTATION.md](DOCUMENTATION.md#cli-usage).

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

Not reliable from static analysis alone: real VRAM usage under load,
time-to-first-token, tokens/sec, true concurrency limits. An opt-in runtime
probe (`run_probe=True` / `--probe`) can measure real load time, approximate
peak VRAM, and a rough generation rate by actually loading the model in a
subprocess — but it still doesn't measure TTFT or steady-state concurrent
throughput. Static estimates are always confidence-tagged so callers don't
mistake one for the other.

## Testing

```bash
pytest                      # unit tests, fully mocked, no GPU/vLLM required
```

Tests for hardware/runtime detection use mocked providers/adapters so the
suite runs on any machine. A small number of integration tests are gated to
run only when real GPU hardware or vLLM is present (none are currently
required to pass CI).
