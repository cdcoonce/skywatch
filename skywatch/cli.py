"""Command-line interface. Subcommands are stubs until their slices land."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from skywatch import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skywatch", description=__doc__)
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="generate a synthetic survey")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--stars", type=int, default=200)
    sim.add_argument("--nights", type=int, default=3)
    sim.add_argument("--out", default="runs/")

    det = sub.add_parser("detect", help="run a detector over a run")
    det.add_argument("--run", required=True)
    det.add_argument("--detector", required=True)

    sc = sub.add_parser("score", help="score detections against truth")
    sc.add_argument("--run", required=True)

    rep = sub.add_parser("report", help="print the scoreboard")
    rep.add_argument("--run", required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 2
    print("not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
