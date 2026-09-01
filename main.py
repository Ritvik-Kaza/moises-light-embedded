#!/usr/bin/env python3
"""
Embedded AI-Based Music Source Separation — command-line interface.

Subcommands:
    separate   Split an audio file into vocals / drums / bass / other.
    benchmark  Measure throughput and memory on synthetic audio (no input file).
    info       Print model and system information without running inference.

Examples:
    python main.py info
    python main.py benchmark --duration 30
    python main.py separate data/input/song.wav
    python main.py separate song.wav --backend openunmix
    python main.py benchmark --profile pi4
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from src.config import DEFAULT_OUTPUT_DIR, PROFILES, RunConfig, STEM_NAMES
from src.metrics import model_size_mb, system_info
from src.separator import Separator, configure_threads


def build_config(args: argparse.Namespace) -> RunConfig:
    """Start from a hardware profile, then apply explicit CLI overrides."""
    base = PROFILES.get(args.profile, RunConfig())
    cfg = replace(base)

    if args.backend is not None:
        cfg.backend = args.backend
    if args.preset is not None:
        cfg.preset = args.preset
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.chunk is not None:
        cfg.chunk_seconds = args.chunk
    if args.overlap is not None:
        cfg.overlap_seconds = args.overlap
    if args.threads is not None:
        cfg.threads = args.threads
    if getattr(args, "output", None):
        cfg.output_dir = args.output
    if getattr(args, "stems", None):
        cfg.stems = args.stems

    cfg.validate()
    return cfg


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile",
        default="laptop",
        choices=sorted(PROFILES.keys()),
        help="hardware profile providing default chunk size and thread count",
    )
    p.add_argument(
        "--backend",
        default=None,
        choices=["moises_light", "openunmix"],
        help="moises_light = architecture under study; openunmix = pretrained reference",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="moises_light preset (paper_small, paper_large, fullband_small, ...)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to trained Moises-Light weights (.pth)",
    )
    p.add_argument("--chunk", type=float, default=None, help="chunk length, seconds")
    p.add_argument("--overlap", type=float, default=None, help="crossfade, seconds")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")


def cmd_info(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    info = system_info()

    print("\n--- System ---")
    for k, v in info.items():
        print(f"  {k:18s}: {v}")

    print("\n--- Configuration ---")
    for k, v in cfg.as_dict().items():
        print(f"  {k:18s}: {v}")

    print("\n--- Model ---")
    try:
        sep = Separator(cfg)
        n = sep.metrics.n_params
        print(f"  backend           : {sep.backend.name}")
        print(f"  parameters        : {n:,} ({n/1e6:.2f} M)")
        print(f"  size fp32         : {model_size_mb(n):.1f} MB")
        print(f"  size int8 (est.)  : {model_size_mb(n, 1):.1f} MB")
        print(f"  pretrained        : {sep.backend.is_pretrained}")
        print(f"  load time         : {sep.metrics.model_load_seconds:.2f} s")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not build backend: {exc}")
        return 1
    print()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    verbose = not args.quiet

    sep = Separator(cfg)
    metrics = sep.benchmark(
        duration_seconds=args.duration, warmup=args.warmup, verbose=verbose
    )
    print(metrics.report())

    if args.save:
        metrics.save_json(args.save)
        print(f"[save] metrics -> {args.save}")
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    verbose = not args.quiet

    sep = Separator(cfg)
    metrics = sep.separate_file(args.input, output_dir=cfg.output_dir, verbose=verbose)
    print(metrics.report())

    if not sep.backend.is_pretrained:
        print(
            "\nNOTE: this run used randomly-initialised weights. The timing and\n"
            "memory figures above are real; the audio files are not separated\n"
            "music. Use --backend openunmix for audible stems, or supply\n"
            "trained weights with --checkpoint.\n"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Embedded AI-based music source separation using Moises-Light",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print model and system information")
    add_common_args(p_info)
    p_info.set_defaults(func=cmd_info)

    p_bench = sub.add_parser("benchmark", help="measure throughput and memory")
    add_common_args(p_bench)
    p_bench.add_argument(
        "--duration", type=float, default=30.0, help="synthetic audio seconds"
    )
    p_bench.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    p_bench.add_argument("--save", default=None, help="write metrics JSON here")
    p_bench.set_defaults(func=cmd_benchmark)

    p_sep = sub.add_parser("separate", help="separate an audio file into stems")
    add_common_args(p_sep)
    p_sep.add_argument("input", help="path to input audio file (WAV recommended)")
    p_sep.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_DIR, help="output directory"
    )
    p_sep.add_argument(
        "--stems",
        nargs="+",
        default=None,
        choices=STEM_NAMES,
        help="subset of stems to write (default: all four)",
    )
    p_sep.set_defaults(func=cmd_separate)

    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should not dump a traceback
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
