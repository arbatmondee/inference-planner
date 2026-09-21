"""Command-line interface.

    inference-planner hardware
    inference-planner analyze --model Qwen/Qwen3-8B --runtime vllm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from inference_planner.hardware.detector import detect_hardware
from inference_planner.planner.planner import InferencePlanner
from inference_planner.schemas.report import to_jsonable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-planner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware_parser = subparsers.add_parser("hardware", help="Detect and print local hardware")
    hardware_parser.add_argument("--json", action="store_true", help="Print JSON instead of text")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a model + runtime combination")
    analyze_parser.add_argument("--model", required=True, help="Model identifier or local path")
    analyze_parser.add_argument("--runtime", required=True, help="Runtime name, e.g. 'vllm'")
    analyze_parser.add_argument("--tensor-parallel-size", type=int, default=None)
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

    if args.command == "analyze":
        try:
            report = InferencePlanner().analyze(
                model=args.model,
                runtime=args.runtime,
                tensor_parallel_size=args.tensor_parallel_size,
                hf_token=args.hf_token,
            )
        except Exception as exc:  # surfaced to the user, not a stack trace
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(report.to_json())
        else:
            print(str(report))
        return 0 if report.compatible else 2

    return 1  # pragma: no cover - argparse enforces a valid subcommand


if __name__ == "__main__":
    sys.exit(main())
