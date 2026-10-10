"""Orchestration behind the CLI: simulate, detect, score, report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import skywatch.detectors.rolling_z  # noqa: F401  (registers the detector)
from skywatch.detectors import REGISTRY
from skywatch.io import store
from skywatch.scoring.match import match_detections
from skywatch.scoring.metrics import KindScore, confusion, format_scoreboard, score
from skywatch.world.defaults import default_injectors
from skywatch.world.survey import SurveyConfig, simulate


def simulate_run(
    cfg: SurveyConfig, out_root: Path, run_id: str | None = None, overwrite: bool = False
) -> Path:
    # Planets come from the field stream, independent of injectors, so a bare pass finds them.
    science, instrument = default_injectors(simulate(cfg).planets)
    result = simulate(cfg, science, instrument)
    return store.write_run(out_root, result, run_id=run_id, overwrite=overwrite)


def detect_run(run_dir: Path, detector_name: str, **kwargs: Any) -> int:
    if detector_name not in REGISTRY:
        raise ValueError(
            f"unknown detector {detector_name!r}; available: {', '.join(sorted(REGISTRY))}"
        )
    detector = REGISTRY[detector_name](**kwargs)
    curves = store.read_curves(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    detections = detector.detect(curves, cadence_s)
    store.write_detections(run_dir, detections)
    if not detections:  # still leave a file so score_run sees the detector ran
        store.write_empty_detections(run_dir, detector_name)
    return len(detections)


def score_run(run_dir: Path) -> list[KindScore]:
    if not (Path(run_dir) / "detections").is_dir():
        raise FileNotFoundError(f"no detections in {run_dir}; run `skywatch detect` first")
    truth = store.read_truth(run_dir)
    detections = store.read_detections(run_dir)
    cadence_s = store.read_manifest(run_dir)["config"]["cadence_s"]
    names = store.detector_names(run_dir)
    matches = match_detections(detections, truth, cadence_s)
    scores = score(matches, detections, truth, detectors=names)
    conf = confusion(matches, detections, truth, detectors=names)
    store.write_scores(run_dir, scores, matches, conf)
    return scores


def report_run(run_dir: Path) -> str:
    if not (Path(run_dir) / "scores.parquet").exists():
        raise FileNotFoundError(f"no scores in {run_dir}; run `skywatch score` first")
    board = format_scoreboard(store.read_scores(run_dir))
    conf = store.read_confusion(run_dir)
    lines = ["confusion (detector, truth_kind, detected_as, count)"]
    lines += [f"{d}  {t}  {a}  {n}" for (d, t, a), n in sorted(conf.items())]
    return board + "\n\n" + "\n".join(lines)
