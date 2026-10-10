"""Run directories: Parquet tables plus a JSON manifest. See docs/design.md, "Data model"."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from skywatch import __version__
from skywatch.scoring.match import Match
from skywatch.scoring.metrics import KindScore
from skywatch.types import Detection, LightCurve, Planet, Star, TruthLabel
from skywatch.world.survey import SurveyResult


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, sort_keys=True)


def _write(path: Path, schema: pa.Schema, columns: dict[str, Any]) -> None:
    pq.write_table(pa.table(columns, schema=schema), path)


def _read(path: Path) -> dict[str, list]:
    return pq.read_table(path).to_pydict()


_STARS = pa.schema(
    [
        ("star_id", pa.int64()),
        ("flux0", pa.float64()),
        ("noise_floor", pa.float64()),
        ("variability", pa.string()),
        ("params", pa.string()),
        ("ra", pa.float64()),
        ("dec", pa.float64()),
    ]
)
_PLANETS = pa.schema(
    [
        ("planet_id", pa.int64()),
        ("star_id", pa.int64()),
        ("period_s", pa.float64()),
        ("epoch_s", pa.float64()),
        ("depth", pa.float64()),
        ("duration_s", pa.float64()),
    ]
)
_OBS = pa.schema(
    [
        ("star_id", pa.int64()),
        ("t", pa.int64()),
        ("flux", pa.float64()),
        ("flux_err", pa.float64()),
        ("quality", pa.uint8()),
    ]
)
_TRUTH = pa.schema(
    [
        ("label_id", pa.string()),
        ("kind", pa.string()),
        ("origin", pa.string()),
        ("star_id", pa.int64()),
        ("t_start", pa.int64()),
        ("t_end", pa.int64()),
        ("params", pa.string()),
    ]
)
_DETECTIONS = pa.schema(
    [
        ("detector", pa.string()),
        ("det_id", pa.string()),
        ("star_id", pa.int64()),
        ("t_start", pa.int64()),
        ("t_end", pa.int64()),
        ("score", pa.float64()),
        ("kind_guess", pa.string()),
    ]
)
_MATCHES = pa.schema(
    [
        ("detector", pa.string()),
        ("det_id", pa.string()),
        ("label_id", pa.string()),
        ("iou", pa.float64()),
    ]
)
_SCORES = pa.schema(
    [
        ("detector", pa.string()),
        ("kind", pa.string()),
        ("tp", pa.int64()),
        ("fp", pa.int64()),
        ("fn", pa.int64()),
        ("precision", pa.float64()),
        ("recall", pa.float64()),
        ("f1", pa.float64()),
    ]
)
_CONFUSION = pa.schema(
    [
        ("detector", pa.string()),
        ("truth_kind", pa.string()),
        ("detected_as", pa.string()),
        ("count", pa.int64()),
    ]
)


def write_run(
    out_root: Path, result: SurveyResult, run_id: str | None = None, overwrite: bool = False
) -> Path:
    cfg = result.config
    if run_id is None:
        run_id = f"seed{cfg.seed}-{len(result.stars)}s-{cfg.n_nights}n"
    run_dir = Path(out_root) / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    s = result.stars
    _write(
        run_dir / "stars.parquet",
        _STARS,
        {
            "star_id": [x.star_id for x in s],
            "flux0": [x.flux0 for x in s],
            "noise_floor": [x.noise_floor for x in s],
            "variability": [x.variability for x in s],
            "params": [_dumps(x.params) for x in s],
            "ra": [x.ra for x in s],
            "dec": [x.dec for x in s],
        },
    )
    p = result.planets
    _write(
        run_dir / "planets.parquet",
        _PLANETS,
        {
            "planet_id": [x.planet_id for x in p],
            "star_id": [x.star_id for x in p],
            "period_s": [x.period_s for x in p],
            "epoch_s": [x.epoch_s for x in p],
            "depth": [x.depth for x in p],
            "duration_s": [x.duration_s for x in p],
        },
    )
    ids = sorted(result.curves)
    curves = [result.curves[i] for i in ids]
    n_samples = sum(len(c) for c in curves)
    _write(
        run_dir / "observations.parquet",
        _OBS,
        {
            "star_id": np.concatenate([np.full(len(c), c.star_id, dtype=np.int64) for c in curves])
            if curves
            else [],
            "t": np.concatenate([c.t for c in curves]).astype(np.int64) if curves else [],
            "flux": np.concatenate([c.flux for c in curves]) if curves else [],
            "flux_err": np.concatenate([c.flux_err for c in curves]) if curves else [],
            "quality": np.concatenate([c.quality for c in curves]).astype(np.uint8)
            if curves
            else [],
        },
    )
    t = result.truth
    _write(
        run_dir / "truth.parquet",
        _TRUTH,
        {
            "label_id": [x.label_id for x in t],
            "kind": [x.kind for x in t],
            "origin": [x.origin for x in t],
            "star_id": [x.star_id for x in t],
            "t_start": [x.t_start for x in t],
            "t_end": [x.t_end for x in t],
            "params": [_dumps(x.params) for x in t],
        },
    )
    manifest = {
        "seed": cfg.seed,
        "config": asdict(cfg),
        "skywatch_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "n_stars": len(result.stars),
        "n_nights": cfg.n_nights,
        "n_samples": n_samples,
        "n_truth": len(t),
        "truth_counts": dict(sorted(Counter(x.kind for x in t).items())),
    }
    (run_dir / "manifest.json").write_text(_dumps(manifest) + "\n")
    return run_dir


def read_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def read_stars(run_dir: Path) -> list[Star]:
    d = _read(Path(run_dir) / "stars.parquet")
    return [
        Star(
            star_id=d["star_id"][i],
            flux0=d["flux0"][i],
            noise_floor=d["noise_floor"][i],
            variability=d["variability"][i],
            ra=d["ra"][i],
            dec=d["dec"][i],
            params=json.loads(d["params"][i]),
        )
        for i in range(len(d["star_id"]))
    ]


def read_planets(run_dir: Path) -> list[Planet]:
    d = _read(Path(run_dir) / "planets.parquet")
    return [Planet(**{k: d[k][i] for k in _PLANETS.names}) for i in range(len(d["planet_id"]))]


def read_curves(run_dir: Path) -> dict[int, LightCurve]:
    tbl = pq.read_table(Path(run_dir) / "observations.parquet")
    star = tbl["star_id"].to_numpy()
    t = tbl["t"].to_numpy()
    flux = tbl["flux"].to_numpy()
    err = tbl["flux_err"].to_numpy()
    q = tbl["quality"].to_numpy()
    out: dict[int, LightCurve] = {}
    for sid in np.unique(star):
        idx = np.flatnonzero(star == sid)
        idx = idx[np.argsort(t[idx], kind="stable")]
        out[int(sid)] = LightCurve(int(sid), t[idx], flux[idx], err[idx], q[idx])
    return out


def read_truth(run_dir: Path) -> list[TruthLabel]:
    d = _read(Path(run_dir) / "truth.parquet")
    return [
        TruthLabel(
            label_id=d["label_id"][i],
            kind=d["kind"][i],
            origin=d["origin"][i],
            star_id=d["star_id"][i],
            t_start=d["t_start"][i],
            t_end=d["t_end"][i],
            params=json.loads(d["params"][i]),
        )
        for i in range(len(d["label_id"]))
    ]


def write_detections(run_dir: Path, detections: Sequence[Detection]) -> None:
    """Write/replace `detections/<detector>.parquet` for each detector present."""
    out = Path(run_dir) / "detections"
    out.mkdir(exist_ok=True)
    by: dict[str, list[Detection]] = {}
    for d in detections:
        by.setdefault(d.detector, []).append(d)
    for name, ds in by.items():
        _write(
            out / f"{name}.parquet",
            _DETECTIONS,
            {
                "detector": [d.detector for d in ds],
                "det_id": [d.det_id for d in ds],
                "star_id": [d.star_id for d in ds],
                "t_start": [d.t_start for d in ds],
                "t_end": [d.t_end for d in ds],
                "score": [d.score for d in ds],
                "kind_guess": [d.kind_guess for d in ds],
            },
        )


def write_empty_detections(run_dir: Path, detector: str) -> None:
    """Record that `detector` ran and found nothing."""
    out = Path(run_dir) / "detections"
    out.mkdir(exist_ok=True)
    _write(out / f"{detector}.parquet", _DETECTIONS, {k: [] for k in _DETECTIONS.names})


def detector_names(run_dir: Path) -> list[str]:
    """Names of detectors that have a detections file, sorted."""
    return sorted(p.stem for p in (Path(run_dir) / "detections").glob("*.parquet"))


def read_detections(run_dir: Path, detector: str | None = None) -> list[Detection]:
    names = detector_names(run_dir) if detector is None else [detector]
    out: list[Detection] = []
    for name in names:
        d = _read(Path(run_dir) / "detections" / f"{name}.parquet")
        out.extend(
            Detection(**{k: d[k][i] for k in _DETECTIONS.names}) for i in range(len(d["det_id"]))
        )
    return out


def write_scores(
    run_dir: Path,
    scores: Sequence[KindScore],
    matches: Sequence[Match],
    confusion: Mapping[tuple[str, str, str], int],
) -> None:
    run_dir = Path(run_dir)
    _write(
        run_dir / "scores.parquet",
        _SCORES,
        {k: [getattr(s, k) for s in scores] for k in _SCORES.names},
    )
    _write(
        run_dir / "matches.parquet",
        _MATCHES,
        {k: [getattr(m, k) for m in matches] for k in _MATCHES.names},
    )
    keys = sorted(confusion)
    _write(
        run_dir / "confusion.parquet",
        _CONFUSION,
        {
            "detector": [k[0] for k in keys],
            "truth_kind": [k[1] for k in keys],
            "detected_as": [k[2] for k in keys],
            "count": [confusion[k] for k in keys],
        },
    )


def read_scores(run_dir: Path) -> list[KindScore]:
    d = _read(Path(run_dir) / "scores.parquet")
    return [KindScore(**{k: d[k][i] for k in _SCORES.names}) for i in range(len(d["detector"]))]


def read_confusion(run_dir: Path) -> dict[tuple[str, str, str], int]:
    d = _read(Path(run_dir) / "confusion.parquet")
    return {
        (d["detector"][i], d["truth_kind"][i], d["detected_as"][i]): d["count"][i]
        for i in range(len(d["detector"]))
    }
