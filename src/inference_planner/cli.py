"""Command-line interface.

    inference-planner hardware
    inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm
    inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm --candidate-versions 0.6.3,0.6.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from inference_planner.hardware.detector import detect_hardware
from inference_planner.planner.planner import InferencePlanner
from inference_planner.schemas.report import AnalysisReport, to_jsonable


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of integers, got '{value}'") from exc


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-planner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware_parser = subparsers.add_parser("hardware", help="Detect and print local hardware")
    hardware_parser.add_argument("--json", action="store_true", help="Print JSON instead of text")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a model + runtime combination")
    analyze_parser.add_argument("--model", required=True, help="Model identifier or local path")
    analyze_parser.add_argument("--runtime", required=True, help="Runtime name, e.g. 'vllm'")
    analyze_parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="Force a specific TP degree. Auto-recommended if omitted.",
    )
    analyze_parser.add_argument(
        "--device-ids", type=_parse_int_list, default=None,
        help=(
            "Comma-separated GPU indices to pin this deployment to (e.g. '0,1'), "
            "instead of letting the planner assume devices 0..tensor_parallel_size-1. "
            "Must match --tensor-parallel-size in count if both are given."
        ),
    )
    analyze_parser.add_argument(
        "--candidate-versions", type=_parse_str_list, default=None,
        help=(
            "Comma-separated runtime version strings to evaluate instead of "
            "(or as well as showing alongside) whatever is installed locally, "
            "e.g. '0.6.3,0.6.2,0.5.4'. Each is checked for existence (e.g. via "
            "PyPI) and evaluated with conservative, generic capabilities -- "
            "its actual code is never introspected unless it happens to match "
            "the installed version. Prints one summary line per candidate."
        ),
    )
    analyze_parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face Hub token, for gated/private model repos. If omitted, "
            "falls back to a locally cached `huggingface-cli login` token or the "
            "HF_TOKEN/HUGGING_FACE_HUB_TOKEN env vars. Prefer the env var over "
            "this flag where possible, since flag values can leak via shell "
            "history or process listings."
        ),
    )
    analyze_parser.add_argument(
        "--probe", action="store_true",
        help=(
            "Real, opt-in Stage-2 validation: if (and only if) static compatibility "
            "passes, actually launch the runtime in a subprocess, download/load the "
            "model, and measure load time and VRAM use. This is expensive -- it "
            "downloads real weights, consumes real GPU memory, and can take minutes. "
            "Never runs unless this flag is passed. Ignored with --candidate-versions."
        ),
    )
    analyze_parser.add_argument(
        "--probe-timeout", type=int, default=600,
        help="Max seconds to wait for --probe before giving up (default: 600).",
    )
    analyze_parser.add_argument("--json", action="store_true", help="Print JSON instead of text")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "hardware":
        return _run_hardware(args)

    if args.command == "analyze":
        if args.candidate_versions:
            return _run_candidate_evaluation(args)
        return _run_analyze(args)

    return 1  # pragma: no cover - argparse enforces a valid subcommand


def _run_hardware(args: argparse.Namespace) -> int:
    hardware = detect_hardware()
    if args.json:
        print(json.dumps(to_jsonable(hardware), indent=2))
    else:
        print(f"CPU: {hardware.cpu.model or 'unknown'} "
              f"({hardware.cpu.physical_cores} cores, "
              f"{hardware.cpu.total_memory_mb / 1024:.1f} GB RAM)")
        if hardware.gpus:
            for gpu in hardware.gpus:
                vram = f"{gpu.memory_total_mb / 1024:.1f} GB" if gpu.memory_total_mb else "unknown"
                print(f"GPU {gpu.index}: {gpu.name} [{gpu.vendor.value}] - {vram} VRAM")
        else:
            print("GPU: none detected")
        if hardware.runtime_stack.cuda_version:
            print(f"CUDA: {hardware.runtime_stack.cuda_version}")
        for warning in hardware.detection_warnings:
            print(f"Warning: {warning}", file=sys.stderr)
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    try:
        report = InferencePlanner().analyze(
            model=args.model,
            runtime=args.runtime,
            tensor_parallel_size=args.tensor_parallel_size,
            device_ids=args.device_ids,
            hf_token=args.hf_token,
            run_probe=args.probe,
            probe_timeout_seconds=args.probe_timeout,
        )
    except Exception as exc:  # surfaced to the user, not a stack trace
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(report.to_json())
    else:
        print(str(report))
    return 0 if report.compatible else 2


def _run_candidate_evaluation(args: argparse.Namespace) -> int:
    if args.probe:
        print("Warning: --probe is ignored with --candidate-versions (candidates aren't installed).", file=sys.stderr)

    try:
        reports = InferencePlanner().evaluate_candidates(
            model=args.model,
            runtime=args.runtime,
            candidate_versions=args.candidate_versions,
            tensor_parallel_size=args.tensor_parallel_size,
            device_ids=args.device_ids,
            hf_token=args.hf_token,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([to_jsonable(r) for r in reports], indent=2))
    else:
        _print_candidate_summary(args.runtime, reports)

    return 0 if any(r.compatible for r in reports) else 2


def _print_candidate_summary(runtime: str, reports: list[AnalysisReport]) -> None:
    print(f"Candidate evaluation for runtime '{runtime}'")
    print("-" * 40)
    for report in reports:
        status = "COMPATIBLE" if report.compatible else "incompatible"
        print(f"{report.runtime.version:>15}  {status}")
        if report.compatible and report.deployment_plan:
            print(f"{'':>15}  tensor_parallel_size={report.deployment_plan.get('tensor_parallel_size')}")
        for error in report.compatibility.errors:
            print(f"{'':>15}  ✗ {error.message}")
        for note in report.runtime.detection_notes:
            print(f"{'':>15}  note: {note}")


if __name__ == "__main__":
    sys.exit(main())
