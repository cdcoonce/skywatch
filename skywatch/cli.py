"""Command-line interface: simulate, detect, score, planets, report, view, site."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from skywatch import __version__, planets, run, site, view
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

    pl = sub.add_parser("planets", help="per-planet events and stacked S/N from a stored run")
    pl.add_argument("--run", required=True)
    pl.add_argument("--min-events", type=int, default=1, help="hide planets with fewer events")
    pl.add_argument("--min-snr", type=float, default=0.0, help="hide planets below this snr_stack")

    rep = sub.add_parser("report", help="print the scoreboard")
    rep.add_argument("--run", required=True)

    vw = sub.add_parser("view", help="draw one star's night with truth and detector spans")
    vw.add_argument("--run", required=True)
    vw.add_argument("--star", type=int, default=None)
    vw.add_argument("--night", type=int, default=None, help="0-based night index")
    vw.add_argument("--pick", action="store_true", help="choose the busiest star-night")
    vw.add_argument("--out", default=None, help=".png or .svg (default runs/<id>/views/...)")
    vw.add_argument("--detectors", default=None, help="comma-separated, default every detector")

    st = sub.add_parser("site", help="write the static page: the views beside the scoreboard")
    st.add_argument("--spec", required=True, help="site spec JSON (runs, views, captions)")
    st.add_argument("--out", default=None, help="page path (default: index.html beside the spec)")
    return p


def _view(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no run at {run_dir} (manifest.json missing)")
    names = None
    if args.detectors is not None:
        names = [n for n in args.detectors.split(",") if n]
        if not names:
            raise ValueError("--detectors is empty; omit it to draw every detector")
    if args.pick:
        if args.star is not None or args.night is not None:
            raise ValueError("--pick chooses the star and night; drop --star and --night")
        pick = view.pick_view(run_dir, names)
        print(pick.describe())
        star, night = pick.star_id, pick.night
    else:
        if args.star is None or args.night is None:
            raise ValueError("give --star and --night, or --pick")
        star, night = args.star, args.night
    data = view.assemble_view(run_dir, star, night, names)
    out = Path(args.out) if args.out else view.default_out_path(run_dir, star, night)
    print(view.render_view(data, out))


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
    elif args.command == "planets":
        rows = planets.planet_rows(Path(args.run))
        print(planets.format_planets(rows, args.min_events, args.min_snr))
    elif args.command == "report":
        print(run.report_run(Path(args.run)))
    elif args.command == "view":
        _view(args)
    elif args.command == "site":
        print(site.generate(Path(args.spec), Path(args.out) if args.out else None))


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
    except (ValueError, TypeError, OSError) as e:
        print(f"skywatch: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
