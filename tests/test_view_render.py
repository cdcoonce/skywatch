"""Rendering: assert on files written and the backend, never on pixels."""

from dataclasses import replace

import matplotlib
import numpy as np
import pytest

from skywatch import view
from skywatch.io import store
from skywatch.types import Detection
from tests.test_view import DETS, MATCHES, PERIOD, TRUTH, result  # noqa: F401 (fixture)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def scored(tmp_path, result):  # noqa: F811
    run = store.write_run(tmp_path, result, run_id="tiny")
    store.write_detections(run, DETS)
    store.write_scores(run, [], MATCHES, {})
    return run


def test_render_writes_png_and_svg(scored, tmp_path):
    data = view.assemble_view(scored, 1, 0)
    png = view.render_view(data, tmp_path / "out" / "x.png")
    svg = view.render_view(data, tmp_path / "out" / "x.svg")
    assert png.read_bytes().startswith(PNG_MAGIC) and png.stat().st_size > 0
    assert svg.read_text().lstrip().startswith("<?xml") and "<svg" in svg.read_text()


def test_agg_backend_is_forced(scored, tmp_path):
    matplotlib.use("pdf", force=True)  # something else was selected; render must override it
    view.render_view(view.assemble_view(scored, 1, 0), tmp_path / "x.png")
    assert matplotlib.get_backend().lower() == "agg"


def test_output_is_deterministic(scored, tmp_path):
    data = view.assemble_view(scored, 1, 0)
    for ext in ("png", "svg"):
        a = view.render_view(data, tmp_path / f"a.{ext}").read_bytes()
        b = view.render_view(data, tmp_path / f"b.{ext}").read_bytes()
        assert a == b


def test_figure_size_does_not_depend_on_the_data(scored, tmp_path):
    from PIL import Image

    sizes = set()
    for sid, night in ((1, 0), (3, 1), (0, 0)):
        out = view.render_view(view.assemble_view(scored, sid, night), tmp_path / f"{sid}.png")
        with Image.open(out) as im:
            sizes.add(im.size)
    assert sizes == {(1200, 800)}


def test_run_with_zero_detections_renders(tmp_path, result):  # noqa: F811
    run = store.write_run(tmp_path, result, run_id="bare")
    out = view.render_view(view.assemble_view(run, 1, 0), tmp_path / "bare.png")
    assert out.stat().st_size > 0


def test_detected_but_empty_lane_renders(tmp_path, result):  # noqa: F811
    run = store.write_run(tmp_path, result, run_id="empty")
    store.write_empty_detections(run, "quiet")
    out = view.render_view(view.assemble_view(run, 1, 0), tmp_path / "e.png")
    assert out.stat().st_size > 0


def test_lane_with_zero_truth_renders(tmp_path, result):  # noqa: F811
    run = store.write_run(tmp_path, replace(result, truth=[]), run_id="notruth")
    store.write_detections(run, DETS)
    data = view.assemble_view(run, 1, 0)
    assert data.truth == ()
    assert view.render_view(data, tmp_path / "t.png").stat().st_size > 0


def test_unscored_run_renders(tmp_path, result):  # noqa: F811
    run = store.write_run(tmp_path, result, run_id="unscored")
    store.write_detections(run, DETS)
    data = view.assemble_view(run, 1, 0)
    assert not data.scored
    assert view.render_view(data, tmp_path / "u.png").stat().st_size > 0


def test_star_with_no_observations_renders(tmp_path, result):  # noqa: F811
    curves = {k: v for k, v in result.curves.items() if k != 2}
    run = store.write_run(tmp_path, replace(result, curves=curves), run_id="nocurve")
    data = view.assemble_view(run, 2, 0)
    assert len(data.t) == 0
    assert view.render_view(data, tmp_path / "n.png").stat().st_size > 0


def test_crowded_lane_and_unobservable_truth_render(tmp_path, result):  # noqa: F811
    run = store.write_run(tmp_path, result, run_id="crowd")
    many = [
        Detection("busy", f"busy-{i}", 1, 100 * i, 100 * i + 3000, 1.0, "flare") for i in range(40)
    ]
    store.write_detections(run, many)
    data = view.assemble_view(run, 0, 1)  # star 0 has an unobservable label in the gap
    assert view.render_view(data, tmp_path / "c.png").stat().st_size > 0
    data = view.assemble_view(run, 1, 0)
    assert view.render_view(data, tmp_path / "c2.png").stat().st_size > 0


def test_bad_suffix_is_refused(scored, tmp_path):
    with pytest.raises(ValueError, match="png or .svg"):
        view.render_view(view.assemble_view(scored, 1, 0), tmp_path / "x.jpg")


def test_night_two_truth_constants_are_in_range():
    # guard for the fixture shared with test_view: the night-1 labels sit after PERIOD
    assert all(t.t_start >= PERIOD for t in TRUTH if "-n1-" in t.label_id)


def test_flux_ticks_are_short_and_consistent():
    from skywatch.view import _flux_tick

    assert [_flux_tick(v, 0) for v in (0, 500, 20000, 350000, 1e6, 1.2e6)] == [
        "0",
        "500",
        "20k",
        "350k",
        "1.0M",
        "1.2M",
    ]


# ----------------------------------------------------------- legend, summaries, metadata


def _figure(data):
    with view._rc():
        return view._build_figure(data)


def _legend_texts(fig):
    return [t.get_text() for leg in fig.legends for t in leg.get_texts()]


def test_grade_legend_text_is_true_to_the_scorer(scored):
    texts = _legend_texts(_figure(view.assemble_view(scored, 1, 0)))
    for expected in (
        "bounded (IoU ≥ 0.3 with truth)",
        "touched (overlaps truth, IoU < 0.3)",
        "corroborating (repeat of a counted event)",
        "false alarm (matched no truth label)",
    ):
        assert expected in texts
    assert "false alarm (no truth here)" not in texts  # a red box can sit inside a truth span


def test_unscored_lane_is_stated_in_summary_and_legend(scored):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = scored / "matches.parquet"
    table = pq.read_table(path)
    pq.write_table(table.filter(pc.not_equal(table["detector"], "beta")), path)
    fig = _figure(view.assemble_view(scored, 1, 0))
    summaries = [t.get_text() for t in fig.texts]
    line = next(t for t in summaries if "beta:" in t)
    assert "beta: unscored 2" in line
    assert "alpha: bounded 1 · touched 1 · corr 0 · fp 1" in line
    assert "unscored (no match rows for this detector)" in _legend_texts(fig)


def test_png_carries_no_software_metadata(scored, tmp_path):
    from PIL import Image

    out = view.render_view(view.assemble_view(scored, 1, 0), tmp_path / "m.png")
    with Image.open(out) as im:
        assert "Software" not in im.info
        assert not any("matplotlib" in str(v).lower() for v in im.info.values())


def test_flag_colour_is_not_a_kind_colour():
    kinds = {c.lower() for c in view.KIND_COLORS.values()}
    assert view.FLAG_COLOR.lower() not in kinds
    assert view.FALLBACK_KIND_COLOR.lower() not in kinds


def test_unknown_kind_falls_back_to_grey(scored, tmp_path):
    data = view.assemble_view(scored, 1, 0)
    odd = view.ViewTruth("nova-n0-s1-0", "nova", "science", 1, 1000, 2000, True)
    data = replace(data, truth=(*data.truth, odd))
    out = view.render_view(data, tmp_path / "odd.png")
    assert out.stat().st_size > 0
    assert "nova" in _legend_texts(_figure(data)) + [
        t.get_text() for t in _figure(data).axes[1].texts
    ]
    assert view._kind_color("nova") == view.FALLBACK_KIND_COLOR


# ------------------------------------------------------------------ flux axis limits


def _synthetic_night(flux, cadence=60):
    n = len(flux)
    return view.ViewData(
        run_id="syn",
        star_id=0,
        night=0,
        night_start=0,
        night_length_s=n * cadence,
        cadence_s=cadence,
        t=np.arange(n, dtype=np.int64) * cadence,
        flux=np.asarray(flux, dtype=float),
        flux_err=np.full(n, 5.0),
        quality=np.zeros(n, dtype=np.uint8),
        truth=(),
        lanes=(),
        scored=False,
    )


def _baseline(n=480):
    return 1000.0 + np.random.default_rng(0).normal(0, 10.0, n)


def _clip_notes(ax):
    return [t.get_text() for t in ax.texts if "▲" in t.get_text() or "▼" in t.get_text()]


def test_one_spike_does_not_set_the_flux_axis():
    flux = _baseline()
    flux[240] = 20 * 1000.0  # a 20x cosmic ray
    ax = _figure(_synthetic_night(flux)).axes[0]
    lo, hi = ax.get_ylim()
    assert hi < 2000.0 and lo > 800.0  # the baseline fills the panel, not 5% of it
    assert (flux > hi).sum() / len(flux) <= 0.005
    assert _clip_notes(ax) == ["▲ 20k"]  # the spike stays an event, with its true value


def test_negative_spike_is_annotated_at_the_bottom():
    flux = _baseline()
    flux[100] = -5000.0
    ax = _figure(_synthetic_night(flux)).axes[0]
    assert _clip_notes(ax) == ["▼ -5k"]


def test_hot_pixel_plateau_stays_inside_the_limits():
    flux = _baseline()
    flux[100:148] += 300.0  # 10% of the night on a raised plateau
    ax = _figure(_synthetic_night(flux)).axes[0]
    lo, hi = ax.get_ylim()
    assert lo <= flux.min() and hi >= flux.max()
    assert _clip_notes(ax) == []


def test_constant_flux_has_a_finite_axis():
    ax = _figure(_synthetic_night(np.full(480, 101.0))).axes[0]
    lo, hi = ax.get_ylim()
    assert lo < 101.0 < hi


def test_flagged_samples_are_drawn_with_the_flag_colour():
    data = _synthetic_night(_baseline())
    q = data.quality.copy()
    q[10:13] = 1
    ax = _figure(replace(data, quality=q)).axes[0]
    edges = [
        tuple(c.get_edgecolor()[0])
        for c in ax.collections
        if len(c.get_offsets()) == 3 and c.get_facecolor().size == 0
    ]
    from matplotlib.colors import to_rgba

    assert edges == [to_rgba(view.FLAG_COLOR)]
