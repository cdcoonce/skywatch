"""Rendering: assert on files written and the backend, never on pixels."""

from dataclasses import replace

import matplotlib
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
