# inference-planner — Documentation

`inference-planner` is a standalone Python library that analyzes a **model**,
a **hardware** environment, and an **inference runtime** (e.g. vLLM) and
tells you whether they're compatible, how much resource they'll need, and
what deployment configuration to use. It doesn't run or serve models itself
— it's the decision-support layer an inference platform's scheduler would
call before deploying.

This document covers day-to-day usage. See [README.md](README.md) for a
shorter overview and design principles.

---

## Table of contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [CLI usage](#cli-usage)
4. [Python API](#python-api)
5. [Understanding the `AnalysisReport`](#understanding-the-analysisreport)
6. [Confidence levels — how to read estimates](#confidence-levels--how-to-read-estimates)
7. [Compatibility rules and error codes](#compatibility-rules-and-error-codes)
8. [Authenticating to Hugging Face Hub](#authenticating-to-hugging-face-hub)
9. [Inspecting local models](#inspecting-local-models)
10. [Package architecture](#package-architecture)
11. [Extending the library](#extending-the-library)
12. [Testing](#testing)
13. [Building & distributing](#building--distributing)
14. [Known limitations](#known-limitations)

---

## Install

```bash
# Core install (hardware detection only; no model/runtime introspection extras)
pip install -e .

# With Hugging Face model inspection (recommended for most uses)
pip install -e ".[huggingface]"

# With pynvml-based GPU detection (falls back to `nvidia-smi` CLI without it)
pip install -e ".[nvidia]"

# With vLLM itself installed, so the vllm adapter reports real capabilities
pip install -e ".[vllm]"

# Everything, plus test dependencies
pip install -e ".[all,dev]"
```

Requires Python 3.10+. The only hard dependency is `psutil` (for CPU/RAM
detection) — everything else degrades gracefully if not installed (see
[Known limitations](#known-limitations)).

---

## Quickstart

```python
from inference_planner import InferencePlanner

report = InferencePlanner().analyze(model="Qwen/Qwen3-8B", runtime="vllm")

print(report)  # human-readable summary

if report.compatible:
    plan = report.deployment_plan
    # {"runtime": "vllm", "devices": [0], "tensor_parallel_size": 1,
    #  "dtype": "bfloat16", "max_model_len": 32768, "gpu_memory_utilization": 0.9, ...}
else:
    for error in report.compatibility.errors:
        print(f"{error.code}: {error.message}")
```

Or from the command line:

```bash
inference-planner hardware
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm
```

---

## CLI usage

The CLI is installed as `inference-planner` (via the `pyproject.toml`
`[project.scripts]` entry point).

### `inference-planner hardware`

Detects and prints the local machine's hardware.

```bash
inference-planner hardware
inference-planner hardware --json
```

Text output:

```text
CPU: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz (4 cores, 14.3 GB RAM)
GPU 0: NVIDIA A100-SXM4-80GB [nvidia] - 80.0 GB VRAM
CUDA: 12.4
```

Any non-fatal detection issues (e.g. "no GPU vendor tooling found") are
printed to stderr as `Warning: ...` lines, not mixed into stdout — so
`--json` output is always valid, parseable JSON.

### `inference-planner analyze`

```bash
inference-planner analyze --model <identifier-or-path> --runtime <name> [options]
```

| Flag | Required | Description |
|---|---|---|
| `--model` | yes | A Hugging Face Hub repo id (e.g. `Qwen/Qwen3-8B`) or a local directory containing `config.json`. |
| `--runtime` | yes | Registered runtime adapter name. Currently: `vllm`. |
| `--tensor-parallel-size` | no | Force a specific TP degree. If omitted, one is recommended automatically based on model size and available VRAM. |
| `--hf-token` | no | Hugging Face Hub token for gated/private repos. See [Authenticating](#authenticating-to-hugging-face-hub). |
| `--json` | no | Print the full `AnalysisReport` as JSON instead of the human-readable summary. |

`-v`/`--verbose` (debug logging to stderr) is a **global** flag and must
come *before* the subcommand, not after: `inference-planner -v analyze ...`,
not `inference-planner analyze -v ...`.

**Exit codes:** `0` if compatible, `2` if incompatible, `1` on a hard error
(e.g. model not found, network failure). This makes it safe to gate a
deploy script on: `inference-planner analyze ... && deploy.sh`.

```bash
# Human-readable
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm

# Machine-readable, for piping into another tool
inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm --json | jq .deployment_plan

# Force TP=2, and pass a token for a gated model
inference-planner analyze --model meta-llama/Llama-3-70B --runtime vllm \
    --tensor-parallel-size 2 --hf-token "$HF_TOKEN"
```

---

## Python API

### The one entry point most callers need

```python
from inference_planner import InferencePlanner

planner = InferencePlanner()
report = planner.analyze(
    model="Qwen/Qwen3-8B",      # required: HF repo id or local path
    runtime="vllm",              # required: registered runtime adapter name
    hardware=None,                # optional: pass a pre-detected HardwareInfo; auto-detects if omitted
    tensor_parallel_size=None,    # optional: force a TP degree; auto-recommended if omitted
    hf_token=None,                # optional: Hugging Face token for gated/private models
)
```

There's also a module-level convenience function if you don't need to reuse
a planner instance: `inference_planner.analyze(model=..., runtime=...)`.

### Individual pieces, used standalone

Each analysis axis also works on its own — useful if you only need one
piece (e.g. just hardware detection for a health check):

```python
from inference_planner import detect_hardware, inspect_model

hardware = detect_hardware()
model = inspect_model("Qwen/Qwen3-8B")

print(hardware.gpu_count, hardware.total_vram_mb)
print(model.architecture, model.parameter_count.count, model.parameter_count.confidence)
```

```python
from inference_planner.runtimes.registry import get_adapter

adapter = get_adapter("vllm")
identity = adapter.identify()               # is vLLM installed? what version?
caps = adapter.capabilities()               # what dtypes/quantization does it support?
estimate = adapter.estimate_resources(model, hardware)
config = adapter.generate_config(model, hardware)
```

```python
from inference_planner.compatibility.analyzer import CompatibilityAnalyzer
from inference_planner.compatibility.rules import CompatibilityContext

ctx = CompatibilityContext(
    model=model, hardware=hardware,
    runtime_identity=identity, runtime_capabilities=caps,
    resource_estimate=estimate, tensor_parallel_size=1,
)
result = CompatibilityAnalyzer().analyze(ctx)
print(result.compatible, result.reasons, result.warnings, result.errors)
```

### Passing hardware explicitly (e.g. for a remote/other machine)

`InferencePlanner.analyze()` calls `detect_hardware()` for you if you don't
pass `hardware=`. If your scheduler already knows about a *different*
machine's hardware (not the one running this code), construct or fetch a
`HardwareInfo` yourself and pass it in:

```python
from inference_planner.hardware.base import HardwareInfo, CPUInfo, GPUInfo, GPUVendor, RuntimeStackInfo

remote_hardware = HardwareInfo(
    cpu=CPUInfo(model="AMD EPYC 7763", physical_cores=64, logical_cores=128,
                total_memory_mb=512_000, available_memory_mb=480_000),
    gpus=(GPUInfo(index=0, name="NVIDIA H100", vendor=GPUVendor.NVIDIA,
                   memory_total_mb=81920, memory_free_mb=80000,
                   compute_capability="9.0", supported_precisions=("fp32","fp16","bf16","fp8")),),
    runtime_stack=RuntimeStackInfo(cuda_version="12.4"),
)

report = InferencePlanner().analyze(model="...", runtime="vllm", hardware=remote_hardware)
```

---

## Understanding the `AnalysisReport`

`AnalysisReport` (`inference_planner.schemas.report.AnalysisReport`) is the
single object everything resolves to. It's a frozen dataclass with:

| Field | Type | Meaning |
|---|---|---|
| `compatible` | `bool` | `True` only if there are zero compatibility **errors** (warnings don't block this). |
| `stage` | `AnalysisStage` | Always `STATIC_ANALYSIS` today (see [Known limitations](#known-limitations)). |
| `hardware` | `HardwareInfo` | Full detected/passed-in hardware snapshot. |
| `model` | `ModelInfo` | Full parsed model metadata. |
| `runtime` | `RuntimeIdentity` | Which engine, installed or not, and its version. |
| `compatibility` | `CompatibilityResult` | `reasons` (passed checks), `warnings`, `errors` — each `CompatibilityIssue` has a `code`, `message`, `severity`. |
| `resource_estimate` | `ResourceEstimate` | Memory/concurrency figures — see below. |
| `deployment_plan` | `dict \| None` | Engine-native launch config, **only present when `compatible=True`**. |
| `warnings` / `errors` | `tuple[str, ...]` | Flattened string versions of hardware-detection warnings + compatibility issues, for quick logging without walking nested objects. |
| `runtime_validation` | `RuntimeValidationResult \| None` | Always `None` today — reserved for the future Stage-2 probe. |

It has two convenience methods:

```python
report.to_dict()   # plain dict, all enums -> strings, all tuples -> lists
report.to_json()   # json.dumps(report.to_dict(), indent=2)
str(report)         # human-readable, same rendering the CLI uses
```

### `resource_estimate` fields

`ResourceEstimate` (`inference_planner.resources.types`) breaks the VRAM
budget down explicitly instead of giving one opaque number:

```python
estimate.weight_memory                     # MemoryEstimate: model weights, per-GPU (sharded by TP)
estimate.kv_cache_memory_per_1k_tokens      # MemoryEstimate: KV cache cost per 1000 tokens, per sequence
estimate.runtime_overhead_memory            # MemoryEstimate: engine fixed overhead (CUDA context, etc.)
estimate.total_vram_required                # MemoryEstimate: weights + overhead, at zero context
estimate.available_vram_gb                  # float: min free VRAM across the GPU(s) that would be used
estimate.fits_in_available_vram             # bool | None: None if it couldn't be determined
estimate.cpu_ram_required                   # MemoryEstimate: heuristic staging-copy estimate
estimate.max_context_length_estimate        # CountEstimate: how much context fits in remaining VRAM
estimate.recommended_tensor_parallel_size   # int: smallest TP degree that fits, or 1 on a single GPU
estimate.approximate_max_concurrency        # CountEstimate: concurrent sequences at full context length
```

Every `MemoryEstimate`/`CountEstimate` carries a `confidence` and a `basis`
tuple explaining what it was derived from — see the next section.

---

## Confidence levels — how to read estimates

Every non-trivial number in this library is tagged with one of:

| Confidence | Meaning |
|---|---|
| `MEASURED` | Read directly from a source of truth (e.g. `nvidia-smi`/pynvml hardware readings). |
| `ESTIMATED` | Derived mathematically from measured inputs (e.g. parameter count computed from an exact on-disk safetensors byte size and a known dtype). |
| `HEURISTIC` | A reasonable approximation with real uncertainty (e.g. parameter count guessed from a generic dense-transformer formula when no weight index is available — this overshoots for MoE models). |
| `UNKNOWN` | Not enough information; the value itself is `None`. |

**Never treat a `HEURISTIC` or `ESTIMATED` figure as exact.** Always check
`.confidence` before trusting a number for a hard capacity decision, and
inspect `.basis` (a tuple of strings) to see what assumptions went into it:

```python
pc = model.parameter_count
if pc.confidence.value in ("heuristic",):
    print(f"Parameter count ~{pc.count/1e9:.1f}B is a rough guess, based on: {pc.basis}")
```

---

## Compatibility rules and error codes

`CompatibilityAnalyzer` runs a fixed pipeline of rules
(`inference_planner.compatibility.rules.DEFAULT_RULES`) against a
`CompatibilityContext`. Each rule either passes silently (optionally
contributing a positive `reason`) or returns a `CompatibilityIssue` with a
`code`, human-readable `message`, and `severity` (`error` or `warning`).
**Only `error`-severity issues make `report.compatible = False`.**

| Code | Severity | Meaning |
|---|---|---|
| `RUNTIME_NOT_INSTALLED` | error | The named runtime engine (e.g. `vllm`) isn't importable in this environment. |
| `UNSUPPORTED_DTYPE` | error | The model's dtype isn't in the runtime's supported dtype list. |
| `UNSUPPORTED_QUANTIZATION` | error | The model's quantization method isn't supported by the runtime. |
| `UNSUPPORTED_PRECISION_FOR_HARDWARE` | error | None of the available GPUs report support for the required precision (derived from GPU compute capability, not GPU name). |
| `INSUFFICIENT_VRAM` | error | Estimated required VRAM per GPU exceeds available free VRAM. |
| `TENSOR_PARALLEL_HEAD_MISMATCH` | error | The chosen `tensor_parallel_size` doesn't evenly divide the model's attention head count — a hard structural constraint of TP attention sharding. |
| `NO_GPU_DETECTED` | warning | No GPU found; most runtimes need one, but this alone doesn't fail compatibility (some engines/backends can run CPU-only). |
| `UNKNOWN_ARCHITECTURE` | warning | The model's `config.json` didn't declare an `architectures` field. |

```python
for error in report.compatibility.errors:
    if error.code == "INSUFFICIENT_VRAM":
        # e.g. suggest a smaller model, more GPUs, or a quantized checkpoint
        ...
```

---

## Authenticating to Hugging Face Hub

Model inspection reads `config.json` (and, when available,
`tokenizer_config.json` / `model.safetensors.index.json`) from the Hub. For
**public** models this requires no authentication at all. A token is only
needed for **gated** (license-gated) or **private** repos.

Three ways to provide one, in order of preference:

1. **Environment variable** (recommended — doesn't show up in shell history
   or process listings): `export HF_TOKEN=hf_...` or run `huggingface-cli
   login` once to cache a token locally. Both are picked up automatically
   whenever you don't pass a token explicitly.
2. **Per-call, via the API**:
   ```python
   InferencePlanner().analyze(model="meta-llama/Llama-3-70B", runtime="vllm", hf_token="hf_...")
   ```
3. **Per-call, via the CLI**: `--hf-token hf_...` (the CLI help text
   explicitly recommends the env var over this flag for the reason above).

If a token is required and missing/invalid, you'll get a clear
`ModelInspectionError` (e.g. `Could not fetch config.json for '...'.
Install the 'huggingface' extra ... and if this is a gated or private repo,
pass a valid Hugging Face token (or set HF_TOKEN).`) rather than a generic
network stack trace.

---

## Inspecting local models

Any local directory containing a `config.json` (an on-disk HF-format
checkout) works exactly like a Hub repo id — no network access is used:

```python
from inference_planner import inspect_model

model = inspect_model("/path/to/local/qwen3-8b")
```

```bash
inference-planner analyze --model /path/to/local/qwen3-8b --runtime vllm
```

This is the recommended way to test the library, write unit tests against
it, or analyze models that were already downloaded/converted locally.

---

## Package architecture

```text
inference_planner/
├── core/            shared primitives: Confidence/Severity/AnalysisStage enums,
│                    exception hierarchy, the generic Registry[T] plugin container,
│                    dtype->bytes-per-element table
├── hardware/        vendor-agnostic hardware discovery
│   ├── base.py      HardwareInfo/GPUInfo/CPUInfo data model + HardwareProvider protocol
│   ├── nvidia.py    NVIDIA provider (pynvml, falling back to `nvidia-smi` CLI)
│   ├── cpu.py       CPU/RAM via psutil
│   └── detector.py  HardwareDetector: runs every registered provider, merges output
├── models/          model metadata inspection
│   ├── base.py      ModelInfo data model + ModelProvider protocol
│   ├── huggingface.py  reads config.json/tokenizer_config.json/safetensors index
│   └── analyzer.py  ModelAnalyzer: dispatches to whichever provider claims an identifier
├── runtimes/        inference engine adapters
│   ├── base.py      RuntimeAdapter protocol + RuntimeCapabilities/RuntimeIdentity
│   ├── vllm.py       VLLMAdapter: introspects the installed vllm package
│   └── registry.py  lookup adapters by name ("vllm", ...)
├── compatibility/   rule-based compatibility verdicts
│   ├── rules.py      Rule protocol + DEFAULT_RULES (8 built-in checks)
│   └── analyzer.py  CompatibilityAnalyzer: runs all registered rules, aggregates
├── resources/       memory/concurrency estimation formulas (shared by every runtime adapter)
│   ├── types.py      OverheadProfile, MemoryEstimate, CountEstimate, ResourceEstimate
│   └── estimator.py ResourceEstimator: the actual math
├── planner/         orchestration
│   └── planner.py   InferencePlanner: hardware + model + runtime -> AnalysisReport
├── schemas/         the final report shape
│   └── report.py    AnalysisReport + JSON/human-readable rendering
├── validation/       Stage-2 (runtime-probe) interfaces — not implemented yet
└── cli.py            argparse-based CLI
```

**Key principle**: nothing branches on a model name or GPU name. Model
facts come from the model's own `config.json`; GPU facts (including which
precisions it supports) come from compute capability, read generically.
Adding support for a new model family requires zero code changes — it just
works if the model publishes a standard-shaped `config.json`. Adding a new
GPU vendor or inference engine means writing one new adapter class, not
touching the planner or compatibility logic.

---

## Extending the library

Every extension point is a plugin registered into a small generic
`Registry[T]` (`inference_planner.core.registry.Registry`). Register your
own alongside — or ahead of, with `priority=True` — the built-ins.

### A new hardware vendor (e.g. AMD)

```python
from inference_planner.hardware.base import HardwareProvider, GPUInfo, GPUVendor
from inference_planner.hardware import register_provider

class AMDHardwareProvider(HardwareProvider):
    def is_available(self) -> bool:
        ...  # e.g. check for rocm-smi

    def detect_gpus(self) -> list[GPUInfo]:
        return [GPUInfo(index=0, name="...", vendor=GPUVendor.AMD, ...)]

register_provider(AMDHardwareProvider())
```

### A new model source (e.g. a private model registry)

```python
from inference_planner.models.base import ModelProvider, ModelInfo
from inference_planner.models import register_provider

class MyRegistryModelProvider(ModelProvider):
    def can_handle(self, identifier: str) -> bool:
        return identifier.startswith("myregistry://")

    def inspect(self, identifier: str, *, token: str | None = None) -> ModelInfo:
        ...

register_provider(MyRegistryModelProvider(), priority=True)  # tried before the HF fallback
```

### A new inference engine (e.g. SGLang)

Implement all five `RuntimeAdapter` methods — `identify`, `capabilities`,
`overhead_profile`, `estimate_resources`, `generate_config` — then:

```python
from inference_planner.runtimes.registry import register_adapter

register_adapter("sglang", SGLangAdapter())
```

`estimate_resources` and `generate_config` can (and should) delegate to the
shared `ResourceEstimator`, supplying your engine's own `OverheadProfile`
rather than reimplementing the memory math — see `runtimes/vllm.py` for the
reference implementation.

### A new compatibility rule

```python
from inference_planner.compatibility.rules import Rule, CompatibilityIssue, CompatibilityContext
from inference_planner.compatibility import register_rule
from inference_planner.core.enums import Severity

class MyCustomRule(Rule):
    def check(self, ctx: CompatibilityContext) -> CompatibilityIssue | None:
        if ctx.model.context_length and ctx.model.context_length > 1_000_000:
            return CompatibilityIssue(
                code="CONTEXT_TOO_LONG_FOR_POLICY",
                message="Context length exceeds this platform's policy limit.",
                severity=Severity.ERROR,
            )
        return None

register_rule(MyCustomRule())
```

---

## Testing

```bash
pip install -e ".[dev,huggingface]"
pytest                                        # all 50+ unit tests, no GPU/vLLM required
pytest --cov=inference_planner --cov-report=term-missing
```

All hardware/runtime tests use mocked providers/adapters (see
`tests/conftest.py`'s `make_hardware`/`make_model` fixtures) so the suite
runs identically on a laptop and a GPU box. `HuggingFaceModelProvider`
tests use local temp directories with a synthetic `config.json` — no
network access needed.

---

## Building & distributing

The package uses a standard `src/`-layout `pyproject.toml` (setuptools
backend) and is ready to distribute:

```bash
pip install build twine
python -m build              # produces dist/*.tar.gz and dist/*.whl
twine check dist/*           # validates metadata before upload

# Internal/private index:
twine upload --repository-url <your-index-url> dist/*

# Public PyPI (check the name isn't already taken first):
twine upload dist/*
```

Or install it straight from a Git remote without publishing anywhere:

```bash
pip install git+https://your-git-host/inference-planner.git
```

---

## Known limitations

- **Static analysis only.** Everything is derived from metadata (config
  files, `nvidia-smi`/pynvml readings), never from actually loading or
  running a model. Real VRAM usage, time-to-first-token, and
  tokens/sec can differ from the estimates. The `inference_planner.validation`
  module defines the interface for an eventual "actually start the engine
  and measure it" stage, but nothing implements it yet.
- **One runtime adapter today**: vLLM. SGLang/TensorRT-LLM/ONNX Runtime
  adapters don't exist yet, though the interface is designed for them.
- **NVIDIA-only hardware provider today.** AMD/Intel/Apple providers aren't
  implemented; `GPUVendor` already has enum values for them, and adding a
  provider doesn't require touching any other module.
- **Parameter-count heuristic overshoots for MoE models.** When a model
  doesn't ship a safetensors weight index, parameter count is estimated
  from a generic dense-transformer formula, which doesn't account for
  sparse/MoE routing — this is why it's tagged `HEURISTIC`, not `ESTIMATED`.
- **`--hf-token`/`hf_token` is HF-Hub-specific** even though the underlying
  `ModelProvider.inspect(..., token=...)` parameter is generic; other model
  providers you add can interpret `token` however fits their auth scheme.
