"""One star, one night: the flux, the injected truth and every detector's spans.

`assemble_view` reads a run directory into a `ViewData` (no drawing, no matplotlib);
`render_view` draws it. The split keeps the data layer unit-testable without a figure,
and keeps ``detect`` and ``score`` free of any matplotlib import: this module imports it
only inside `render_view`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from skywatch.io import store
from skywatch.types import Detection, TruthLabel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

GRADES = ("bounded", "touched", "corroborating", "false_alarm")
"""Grades of a scored detection. ``None`` (not in this tuple) means the run was not scored."""

# One colour per kind in docs/design.md (Okabe-Ito based, distinguishable for colour-blind
# readers). Science kinds are blues, greens and oranges; instrument glitches are the
# purples, reds and browns. Grades use a separate channel (fill style), see `_GRADE_STYLE`.
KIND_COLORS: dict[str, str] = {
    "transit": "#0072B2",
    "flare": "#E69F00",
    "eclipse": "#56B4E9",
    "pulsation": "#009E73",
    "cosmic_ray": "#CC79A7",
    "cloud": "#7570B3",
    "tracking_drift": "#D55E00",
    "hot_pixel": "#8C564B",
    "dropout": "#555555",
}

FALLBACK_KIND_COLOR = "#999999"
"""Fill for a truth kind that is not in `KIND_COLORS` (a kind added after this palette)."""

FLAG_COLOR = "#111111"
"""Instrument-flagged samples: hollow rings in a colour no truth kind uses."""

GRADE_LABELS: dict[str | None, str] = {
    "bounded": "bounded (IoU ≥ 0.3 with truth)",
    "touched": "touched (overlaps truth, IoU < 0.3)",
    "corroborating": "corroborating (repeat of a counted event)",
    "false_alarm": "false alarm (matched no truth label)",
    None: "unscored (no match rows for this detector)",
}

FLUX_PERCENTILES = (0.5, 99.5)
"""The flux axis spans these percentiles of the night's samples, plus `FLUX_MARGIN`."""
FLUX_MARGIN = 0.15
_CLIP_CLUSTER_H = 0.25  # hours; clipped samples closer than this share one annotation

FIG_SIZE = (12.0, 8.0)
DPI = 100
_MIN_BAR_H = 0.04  # hours; one-cadence events would otherwise be sub-pixel slivers


# --------------------------------------------------------------------------- data layer


@dataclass(frozen=True)
class ViewTruth:
    label_id: str
    kind: str
    origin: str
    star_id: int | None
    t_start: int
    t_end: int
    observable: bool

    @property
    def suffix(self) -> str:
        """The label id without its kind prefix, e.g. ``n1-s5-0``."""
        return self.label_id.removeprefix(self.kind + "-")


@dataclass(frozen=True)
class ViewDetection:
    det_id: str
    star_id: int | None
    t_start: int
    t_end: int
    score: float
    kind_guess: str | None
    grade: str | None
    """bounded, touched, corroborating, false_alarm, or None when the run was not scored."""
    label_id: str | None


@dataclass(frozen=True)
class Lane:
    detector: str
    detections: tuple[ViewDetection, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {g: sum(1 for d in self.detections if d.grade == g) for g in GRADES}

    def summary(self, scored: bool) -> str:
        # a detector detected after scoring has no match rows: all its grades are None, and
        # zero counts would read as "zero false alarms"
        if not scored or (self.detections and all(d.grade is None for d in self.detections)):
            return f"unscored {len(self.detections)}"
        c = self.counts
        return (
            f"bounded {c['bounded']} · touched {c['touched']} · "
            f"corr {c['corroborating']} · fp {c['false_alarm']}"
        )


@dataclass(frozen=True, eq=False)
class ViewData:
    run_id: str
    star_id: int
    night: int
    night_start: int
    night_length_s: int
    cadence_s: int
    t: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray
    truth: tuple[ViewTruth, ...]
    lanes: tuple[Lane, ...]
    scored: bool


@dataclass(frozen=True)
class Pick:
    star_id: int
    night: int
    kinds: tuple[str, ...]
    n_detections: int

    def describe(self) -> str:
        return (
            f"picked star {self.star_id}, night {self.night}: "
            f"{len(self.kinds)} truth kinds ({', '.join(self.kinds) or 'none'}), "
            f"{self.n_detections} detections"
        )


def default_out_path(run_dir: Path, star_id: int, night: int) -> Path:
    return Path(run_dir) / "views" / f"star{star_id}-night{night}.png"


def _night_bounds(manifest: dict, night: int) -> tuple[int, int]:
    cfg = manifest["config"]
    if not 0 <= night < cfg["n_nights"]:
        raise ValueError(f"night must be in 0..{cfg['n_nights'] - 1}, got {night}")
    start = night * (cfg["night_length_s"] + cfg["night_gap_s"])
    return start, start + cfg["night_length_s"]


def _intersects(t_start: int, t_end: int, lo: int, hi: int) -> bool:
    """Half-open spans [t_start, t_end) and [lo, hi) share at least one instant."""
    return t_start < hi and t_end > lo


def _select_detectors(run_dir: Path, detectors: Sequence[str] | None) -> list[str]:
    available = store.detector_names(run_dir) if (Path(run_dir) / "detections").is_dir() else []
    if detectors is None:
        return available
    unknown = [d for d in detectors if d not in available]
    if unknown:
        raise ValueError(
            f"no detections for {', '.join(repr(u) for u in unknown)} in {run_dir}; "
            f"available: {', '.join(available) or 'none'}"
        )
    return list(detectors)


def _grades(run_dir: Path) -> dict[tuple[str, str], tuple[str | None, str | None]] | None:
    """(detector, det_id) -> (grade, label_id) from matches.parquet, or None if unscored."""
    if not (Path(run_dir) / "matches.parquet").exists():
        return None
    return {(m.detector, m.det_id): (m.grade, m.label_id) for m in store.read_matches(run_dir)}


def _view_detection(d: Detection, grades: dict | None) -> ViewDetection:
    grade: str | None = None
    label_id: str | None = None
    if grades is not None and (d.detector, d.det_id) in grades:
        g, label_id = grades[(d.detector, d.det_id)]
        grade = g if g is not None else "false_alarm"
    return ViewDetection(
        d.det_id, d.star_id, d.t_start, d.t_end, d.score, d.kind_guess, grade, label_id
    )


def _in_view(d: Detection, star_id: int, lo: int, hi: int) -> bool:
    return (d.star_id is None or d.star_id == star_id) and _intersects(d.t_start, d.t_end, lo, hi)


def _truth_in_view(t: TruthLabel, star_id: int, lo: int, hi: int) -> bool:
    return (t.star_id is None or t.star_id == star_id) and _intersects(t.t_start, t.t_end, lo, hi)


def assemble_view(
    run_dir: Path, star_id: int, night: int, detectors: Sequence[str] | None = None
) -> ViewData:
    """Everything the picture needs for one star on one 0-based night.

    Truth is the star's own labels plus the field-wide ones (``star_id`` null) that
    intersect the night; a label straddling a night edge is included. Each detector's
    lane holds its detections on this star plus its field-wide detections that intersect
    the night, joined to ``matches.parquet`` for grade and label id. A detection with no
    match row in a scored run is a ``false_alarm``; with no match row at all (the run, or
    that detector, was never scored) its grade is ``None``.
    """
    run_dir = Path(run_dir)
    manifest = store.read_manifest(run_dir)
    lo, hi = _night_bounds(manifest, night)
    if star_id not in {s.star_id for s in store.read_stars(run_dir)}:
        raise ValueError(f"star {star_id} is not in {run_dir}")
    names = _select_detectors(run_dir, detectors)

    curve = store.read_curves(run_dir).get(star_id)
    if curve is None:
        empty = np.empty(0)
        t, flux, err, q = empty.astype(np.int64), empty, empty, empty.astype(np.uint8)
    else:
        keep = (curve.t >= lo) & (curve.t < hi)
        t, flux, err, q = curve.t[keep], curve.flux[keep], curve.flux_err[keep], curve.quality[keep]

    truth = sorted(
        (
            ViewTruth(x.label_id, x.kind, x.origin, x.star_id, x.t_start, x.t_end, x.observable)
            for x in store.read_truth(run_dir)
            if _truth_in_view(x, star_id, lo, hi)
        ),
        key=lambda x: (x.t_start, x.t_end, x.label_id),
    )

    grades = _grades(run_dir)
    lanes = []
    for name in names:
        dets = sorted(
            (
                _view_detection(d, grades)
                for d in store.read_detections(run_dir, name)
                if _in_view(d, star_id, lo, hi)
            ),
            key=lambda d: (d.t_start, d.t_end, d.det_id),
        )
        lanes.append(Lane(name, tuple(dets)))

    return ViewData(
        run_id=run_dir.name,
        star_id=star_id,
        night=night,
        night_start=lo,
        night_length_s=hi - lo,
        cadence_s=manifest["config"]["cadence_s"],
        t=t,
        flux=flux,
        flux_err=err,
        quality=q,
        truth=tuple(truth),
        lanes=tuple(lanes),
        scored=grades is not None,
    )


def pick_view(run_dir: Path, detectors: Sequence[str] | None = None) -> Pick:
    """The (star, night) with the most distinct truth kinds; ties by detections, then ids.

    Kinds are those of the star's own labels plus the field-wide ones intersecting the
    night, so a cloud counts for every star. Unobservable labels (the event changed zero
    samples) do not count: the picture should show events that are visible in the flux. Detections are counted across the chosen
    detectors the same way (own plus field-wide). Remaining ties go to the lowest star id,
    then the lowest night.
    """
    run_dir = Path(run_dir)
    manifest = store.read_manifest(run_dir)
    n_nights = manifest["config"]["n_nights"]
    names = _select_detectors(run_dir, detectors)
    truth = [t for t in store.read_truth(run_dir) if t.observable]
    dets = [d for name in names for d in store.read_detections(run_dir, name)]
    stars = sorted(s.star_id for s in store.read_stars(run_dir))
    if not stars:
        raise ValueError(f"no stars in {run_dir}")

    best: tuple[tuple[int, int, int, int], Pick] | None = None
    for night in range(n_nights):
        lo, hi = _night_bounds(manifest, night)
        field_kinds = {
            t.kind for t in truth if t.star_id is None and _intersects(t.t_start, t.t_end, lo, hi)
        }
        field_dets = sum(
            1 for d in dets if d.star_id is None and _intersects(d.t_start, d.t_end, lo, hi)
        )
        own_kinds: dict[int, set[str]] = {}
        for t in truth:
            if t.star_id is not None and _intersects(t.t_start, t.t_end, lo, hi):
                own_kinds.setdefault(t.star_id, set()).add(t.kind)
        own_dets: dict[int, int] = {}
        for d in dets:
            if d.star_id is not None and _intersects(d.t_start, d.t_end, lo, hi):
                own_dets[d.star_id] = own_dets.get(d.star_id, 0) + 1
        for sid in stars:
            kinds = field_kinds | own_kinds.get(sid, set())
            n_det = field_dets + own_dets.get(sid, 0)
            key = (len(kinds), n_det, -sid, -night)
            if best is None or key > best[0]:
                best = (key, Pick(sid, night, tuple(sorted(kinds)), n_det))
    assert best is not None  # stars and n_nights are both non-empty
    return best[1]


# --------------------------------------------------------------------------- drawing


def _pack(extents: Sequence[tuple[float, float]]) -> list[int]:
    """Greedy row assignment: items in the given order, first row where they fit."""
    row_end: list[float] = []
    rows = []
    for lo, hi in extents:
        for r, end in enumerate(row_end):
            if lo >= end:
                row_end[r] = hi
                rows.append(r)
                break
        else:
            row_end.append(hi)
            rows.append(len(row_end) - 1)
    return rows


def _hours(data: ViewData, t) -> np.ndarray:
    return (np.asarray(t, dtype=float) - data.night_start) / 3600.0


def _display_span(data: ViewData, t_start: int, t_end: int) -> tuple[float, float]:
    x0, x1 = (float(_hours(data, t_start)), float(_hours(data, t_end)))
    x0, x1 = max(x0, 0.0), min(x1, data.night_length_s / 3600.0)
    if x1 - x0 < _MIN_BAR_H:
        mid = (x0 + x1) / 2
        x0, x1 = mid - _MIN_BAR_H / 2, mid + _MIN_BAR_H / 2
    return x0, x1


def _text_hours(text: str, fontsize: float, night_hours: float, axes_in: float) -> float:
    """Rough width of `text` on the time axis, for packing; a generous per-character guess."""
    return len(text) * fontsize * 0.62 / 72.0 * night_hours / axes_in + 0.06


def _kind_color(kind: str) -> str:
    return KIND_COLORS.get(kind, FALLBACK_KIND_COLOR)


def _flux_limits(flux: np.ndarray) -> tuple[float, float]:
    """Axis limits from robust percentiles of the night's flux, padded by `FLUX_MARGIN`.

    A single cosmic ray at 20x the baseline would otherwise squash everything else into a
    sliver; at most ``100 - FLUX_PERCENTILES[1]`` percent of the samples fall outside, and
    a plateau of several percent of samples (a hot pixel) stays inside.
    """
    lo, hi = np.percentile(flux, FLUX_PERCENTILES)
    span = hi - lo
    if span <= 0:  # a constant night: any non-empty window will do
        span = max(abs(float(hi)) * 0.02, 1.0)
        lo, hi = lo - span / 2, hi + span / 2
    return float(lo - FLUX_MARGIN * span), float(hi + FLUX_MARGIN * span)


def _annotate_clipped(ax: Axes, t: np.ndarray, flux: np.ndarray, lo: float, hi: float, night_h):
    """Mark samples beyond the limits: a triangle glyph and the true value, pinned to the edge."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for sign, glyph, ypos, va in ((1, "▲", 0.99, "top"), (-1, "▼", 0.01, "bottom")):
        out = np.flatnonzero(flux > hi if sign > 0 else flux < lo)
        if not len(out):
            continue
        clusters = np.split(out, np.flatnonzero(np.diff(t[out]) > _CLIP_CLUSTER_H) + 1)
        for idx in clusters:
            worst = idx[np.argmax(sign * flux[idx])]
            x = float(t[worst])
            ha = "left" if x < 0.4 else "right" if x > night_h - 0.4 else "center"
            ax.text(
                x,
                ypos,
                f"{glyph} {_flux_tick(float(flux[worst]), 0)}",
                transform=trans,
                ha=ha,
                va=va,
                fontsize=8,
                fontweight="bold",
                color=FLAG_COLOR,
                zorder=4,
                clip_on=False,
            )


def _flux_tick(value: float, _pos: int) -> str:
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def _draw_flux(ax: Axes, data: ViewData) -> None:
    from matplotlib.ticker import FuncFormatter

    night_h = data.night_length_s / 3600.0
    for tr in data.truth:
        x0, x1 = _display_span(data, tr.t_start, tr.t_end)
        long = (x1 - x0) > 0.6 * night_h
        ax.axvspan(x0, x1, color=_kind_color(tr.kind), alpha=0.07 if long else 0.2, lw=0, zorder=0)
    t = _hours(data, data.t)
    if len(t):
        gap = np.flatnonzero(np.diff(data.t) > 1.5 * data.cadence_s)
        for i in gap:  # missing samples: hatched grey band between the neighbours
            x0, x1 = float(t[i]), float(t[i + 1])
            ax.axvspan(
                x0,
                x1,
                facecolor="#dddddd",
                edgecolor="#999999",
                hatch="///",
                lw=0,
                alpha=0.6,
                zorder=0.5,
            )
        flux = data.flux.astype(float)
        lo, hi = _flux_limits(flux)
        shown = np.clip(flux, lo, hi)  # a spike is drawn to the edge and annotated instead
        line_t, line_f = np.insert(t, gap + 1, np.nan), np.insert(shown, gap + 1, np.nan)
        ax.plot(line_t, line_f, color="#9aa5b1", lw=0.5, zorder=1)
        inside = (flux >= lo) & (flux <= hi)
        ax.errorbar(
            t[inside],
            flux[inside],
            yerr=data.flux_err[inside],
            fmt="none",
            ecolor="#9aa5b1",
            elinewidth=0.5,
            alpha=0.35,
            zorder=1.5,
        )
        flagged = data.quality != 0
        ok = inside & ~flagged
        ax.scatter(t[ok], flux[ok], s=5, color="#1f3b5c", zorder=2, lw=0)
        if (inside & flagged).any():
            ax.scatter(
                t[inside & flagged],
                flux[inside & flagged],
                s=22,
                facecolors="none",
                edgecolors=FLAG_COLOR,
                linewidths=0.9,
                zorder=3,
            )
        ax.set_ylim(lo, hi)
        _annotate_clipped(ax, t, flux, lo, hi, night_h)
    else:
        ax.text(
            0.5,
            0.5,
            "no observations for this star on this night",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#666666",
        )
    ax.yaxis.set_major_formatter(FuncFormatter(_flux_tick))
    ax.set_ylabel("flux\n(photons / cadence)", fontsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)


def _draw_lane(
    ax: Axes, data: ViewData, items: list[tuple[float, float, str, dict]], axes_in: float
) -> None:
    """Bars with text starting at the bar's left edge; overlapping items stack in rows."""
    night_h = data.night_length_s / 3600.0
    fontsize = 7.5
    # Text runs right from the bar's left edge; a bar too near the night's end flips its
    # text to run left from the bar's right edge, so nothing is clipped at the axis.
    flip = [
        bool(text) and x0 + _text_hours(text, fontsize, night_h, axes_in) > night_h
        for x0, _, text, _ in items
    ]
    extents = []
    for (x0, x1, text, _), f in zip(items, flip, strict=True):
        width = _text_hours(text, fontsize, night_h, axes_in) if text else 0.0
        extents.append((min(x0, x1 - width), x1) if f else (x0, max(x1, x0 + width)))
    rows = _pack(extents)
    n_rows = max(rows, default=0) + 1
    ax.set_ylim(n_rows, 0)
    for (x0, x1, text, style), row, f in zip(items, rows, flip, strict=True):
        ax.barh(row + 0.5, x1 - x0, left=x0, height=0.72, align="center", zorder=2, **style)
        if text:
            ax.text(
                x1 - 0.02 if f else x0 + 0.02,
                row + 0.5,
                text,
                fontsize=fontsize,
                va="center",
                ha="right" if f else "left",
                zorder=3,
                color="#111111",
                clip_on=True,
            )


def _grade_style(grade: str | None) -> dict:
    green, blue, red, grey = "#1b7f3b", "#1f5fbf", "#c62828", "#8a8a8a"
    return {
        "bounded": {"facecolor": green, "edgecolor": green, "lw": 1.0, "alpha": 0.85},
        "touched": {"facecolor": "#bfe3c8", "edgecolor": green, "lw": 1.0, "hatch": "////"},
        "corroborating": {"facecolor": "none", "edgecolor": blue, "lw": 1.6},
        "false_alarm": {"facecolor": "none", "edgecolor": red, "lw": 1.8},
        None: {"facecolor": "#d0d0d0", "edgecolor": grey, "lw": 1.0},
    }[grade]


def _rc():
    import matplotlib

    return matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "hatch.linewidth": 0.8,
            "svg.hashsalt": "skywatch",
            "svg.fonttype": "path",
        }
    )


def render_view(data: ViewData, out_path: Path) -> Path:
    """Draw `data` to `out_path` (PNG, or SVG when the name ends in ``.svg``).

    Headless: the Agg backend is forced, and the figure is built without pyplot so no
    global figure state is touched. Figure size, dpi, fonts and file metadata are fixed so
    the same data gives the same bytes.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)

    out_path = Path(out_path)
    suffix = out_path.suffix.lower()
    if suffix not in (".png", ".svg"):
        raise ValueError(f"--out must end in .png or .svg, got {out_path.name!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _rc():
        fig = _build_figure(data)
        meta = {"Software": None} if suffix == ".png" else {"Date": None}
        fig.savefig(out_path, format=suffix[1:], dpi=DPI, metadata=meta)
    return out_path


def _build_figure(data: ViewData) -> Figure:
    """The whole picture as a matplotlib Figure (tests read its text and limits)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_lanes = len(data.lanes)
    night_h = data.night_length_s / 3600.0
    with _rc():
        fig = Figure(figsize=FIG_SIZE, dpi=DPI)
        FigureCanvasAgg(fig)
        ratios = [5.0, 1.5, *([1.2] * n_lanes)]
        title_lines = 1 + max(1, -(-n_lanes // 3)) if n_lanes else 1
        top = 1.0 - (0.045 + 0.03 * title_lines)
        axes = fig.subplots(
            len(ratios),
            1,
            sharex=True,
            height_ratios=ratios,
            gridspec_kw={"left": 0.17, "right": 0.985, "top": top, "bottom": 0.17, "hspace": 0.12},
        )
        axes = list(np.atleast_1d(axes))
        axes_in = FIG_SIZE[0] * (0.985 - 0.17)

        _draw_flux(axes[0], data)

        truth_items = [
            (
                *_display_span(data, tr.t_start, tr.t_end),
                f"{tr.kind} {tr.suffix}" + ("" if tr.observable else " (unobservable)"),
                {
                    "facecolor": _kind_color(tr.kind),
                    "edgecolor": _kind_color(tr.kind),
                    "alpha": 0.6,
                    "lw": 0.8,
                    "hatch": None if tr.observable else "xx",
                },
            )
            for tr in data.truth
        ]
        _draw_lane(axes[1], data, truth_items, axes_in)
        if not truth_items:
            axes[1].text(
                0.5,
                0.5,
                "no truth events on this star tonight",
                transform=axes[1].transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
        axes[1].set_ylabel("truth\n(injected)", fontsize=9, fontweight="bold")

        for ax, lane in zip(axes[2:], data.lanes, strict=True):
            items = [
                (
                    *_display_span(data, d.t_start, d.t_end),
                    d.kind_guess or "",
                    _grade_style(d.grade),
                )
                for d in lane.detections
            ]
            _draw_lane(ax, data, items, axes_in)
            if not items:
                ax.text(
                    0.5,
                    0.5,
                    "no detections",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                )
            ax.set_ylabel(lane.detector, fontsize=9, fontweight="bold")

        for ax in axes:
            ax.set_xlim(0, night_h)
            ax.set_xticks(np.arange(0, night_h + 0.01, 1.0))
            ax.grid(axis="x", color="#eeeeee", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.yaxis.set_label_coords(-0.075 if ax is axes[0] else -0.012, 0.5)
            ax.yaxis.label.set_ha("right")
            ax.yaxis.label.set_va("center")
            ax.yaxis.label.set_rotation(0)
        for ax in axes[1:]:
            ax.set_yticks([])
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel(
            f"hours since night {data.night} started (t = {data.night_start} s)", fontsize=9
        )

        fig.text(
            0.017,
            0.975,
            f"run {data.run_id} · star {data.star_id} · night {data.night}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        chunks = [data.lanes[i : i + 3] for i in range(0, n_lanes, 3)] or [()]
        for i, chunk in enumerate(chunks):
            line = "      ".join(f"{ln.detector}: {ln.summary(data.scored)}" for ln in chunk)
            fig.text(
                0.017,
                0.975 - 0.035 * (i + 1) - 0.005,
                line or "no detectors in this run",
                fontsize=9,
                va="top",
                color="#333333",
            )

        kinds = sorted(
            {t.kind for t in data.truth},
            key=lambda k: (list(KIND_COLORS).index(k) if k in KIND_COLORS else len(KIND_COLORS), k),
        )
        grade_keys = [*GRADES] if data.scored else [None]
        if data.scored and any(d.grade is None for ln in data.lanes for d in ln.detections):
            grade_keys.append(None)
        kind_handles = [
            Patch(facecolor=_kind_color(k), edgecolor=_kind_color(k), alpha=0.6, label=k)
            for k in kinds
        ]
        kind_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markerfacecolor="none",
                markeredgecolor=FLAG_COLOR,
                markersize=5,
                label="sample flagged by instrument",
            )
        )
        if kinds or len(data.t):
            leg = fig.legend(
                handles=kind_handles,
                title="truth kind (shading on flux and lane)",
                loc="lower left",
                bbox_to_anchor=(0.017, 0.012),
                fontsize=8,
                title_fontsize=8.5,
                ncol=2,
                frameon=False,
                alignment="left",
            )
            fig.add_artist(leg)
        grade_handles = []
        for g in grade_keys:
            st = _grade_style(g)
            grade_handles.append(
                Patch(
                    facecolor=st["facecolor"],
                    edgecolor=st["edgecolor"],
                    lw=st["lw"],
                    hatch=st.get("hatch"),
                    label=GRADE_LABELS[g],
                )
            )
        if n_lanes:
            fig.legend(
                handles=grade_handles,
                title="detector span, graded against truth",
                loc="lower right",
                bbox_to_anchor=(0.985, 0.012),
                fontsize=8,
                title_fontsize=8.5,
                frameon=False,
                alignment="left",
            )
    return fig


__all__ = [
    "GRADES",
    "KIND_COLORS",
    "Lane",
    "Pick",
    "ViewData",
    "ViewDetection",
    "ViewTruth",
    "assemble_view",
    "default_out_path",
    "pick_view",
    "render_view",
]
