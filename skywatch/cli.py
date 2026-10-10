"""Command-line interface: simulate, detect, score, report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from skywatch import __version__, run
from skywatch.scoring.metrics import format_scoreboard
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skywatch", description=__doc__)
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="generate a synthetic survey")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--stars", type=int, default=200)
    sim.add_argument("--nights", type=int, default=3)
    sim.add_argument("--cadence", type=int, default=60, help="seconds between samples")
    sim.add_argument("--out", default="runs/")
    sim.add_argument("--run-id", default=None)
    sim.add_argument("--overwrite", action="store_true")

    det = sub.add_parser("detect", help="run a detector over a run")
    det.add_argument("--run", required=True)
    det.add_argument("--detector", required=True)
    det.add_argument("--window", type=int, default=None)
    det.add_argument("--z-threshold", type=float, default=None)

    sc = sub.add_parser("score", help="score detections against truth")
    sc.add_argument("--run", required=True)

    rep = sub.add_parser("report", help="print the scoreboard")
    rep.add_argument("--run", required=True)
    return p


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "simulate":
        cfg = SurveyConfig(
            seed=args.seed,
            n_nights=args.nights,
            cadence_s=args.cadence,
            field=FieldConfig(n_stars=args.stars),
        )
        print(run.simulate_run(cfg, Path(args.out), args.run_id, args.overwrite))
    elif args.command == "detect":
        kwargs = {
            k: v
            for k, v in (("window", args.window), ("z_threshold", args.z_threshold))
            if v is not None
        }
        n = run.detect_run(Path(args.run), args.detector, **kwargs)
        print(f"{n} detections")
    elif args.command == "score":
        print(format_scoreboard(run.score_run(Path(args.run))))
    elif args.command == "report":
        print(run.report_run(Path(args.run)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 2
    try:
        _dispatch(args)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as e:
        print(f"skywatch: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
