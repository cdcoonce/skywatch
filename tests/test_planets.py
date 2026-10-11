"""`skywatch planets`: per-planet events and stacked S/N, checked against an independent
recomputation from the stored parquet files (no helper shared with skywatch.planets)."""

import json
import math
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from skywatch import planets, run
from skywatch.cli import main
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig

HEADER = ["planet_id", "star", "period_d", "depth", "dur_h", "events", "snr_stack", "snr_max"]


def _simulate(root, seed=5):
    cfg = SurveyConfig(
        seed=seed,
        n_nights=8,
        cadence_s=300,
        field=FieldConfig(n_stars=40, planet_rate=1.5),
    )
    return run.simulate_run(cfg, root)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return _simulate(tmp_path_factory.mktemp("runs"))


def _rewrite(path, edit):
    """Replace a parquet table with ``edit(pydict)`` applied, keeping its schema."""
    tbl = pq.read_table(path)
    d = tbl.to_pydict()
    edit(d)
    pq.write_table(pa.table(d, schema=tbl.schema), path)


def _clone(run_dir, tmp_path):
    out = tmp_path / "clone"
    out.mkdir()
    for p in run_dir.iterdir():
        if p.is_file():
            (out / p.name).write_bytes(p.read_bytes())
    return out


def _labels(run_dir):
    d = pq.read_table(run_dir / "truth.parquet").to_pydict()
    out = []
    for i in range(len(d["label_id"])):
        if d["kind"][i] == "transit":
            out.append((d["star_id"][i], d["t_start"][i], d["t_end"][i], json.loads(d["params"][i])))
    return out


def _expected(run_dir):
    """planet_id -> (events, snr_stack, snr_max), straight from the formula in issue #28."""
    stars = pq.read_table(run_dir / "stars.parquet").to_pydict()
    sigma = {
        stars["star_id"][i]: math.sqrt(stars["noise_floor"][i] ** 2 + 1.0 / stars["flux0"][i])
        for i in range(len(stars["star_id"]))
    }
    obs = pq.read_table(run_dir / "observations.parquet", columns=["star_id", "t"]).to_pydict()
    star_col = np.array(obs["star_id"])
    t_col = np.array(obs["t"], dtype=float)
    pl = pq.read_table(run_dir / "planets.parquet").to_pydict()
    out = {pl["planet_id"][i]: [0, []] for i in range(len(pl["planet_id"]))}
    for star, t0, t1, prm in _labels(run_dir):
        entry = out[prm["planet_id"]]
        entry[0] += 1
        if prm.get("footprint_samples", 1) == 0:
            entry[1].append(0.0)
            continue
        t = t_col[(star_col == star) & (t_col >= t0) & (t_col <= t1)]
        dur = prm["duration_s"]
        ramp = 0.1 * dur
        start = prm["t_mid"] - dur / 2
        end = prm["t_mid"] + dur / 2
        prof = np.clip(np.minimum((t - start) / ramp, (end - t) / ramp), 0.0, 1.0)
        entry[1].append(prm["depth"] * math.sqrt(float(np.sum(prof**2))) / sigma[star])
    return {
        pid: (n, math.sqrt(sum(s * s for s in snrs)), max(snrs, default=0.0))
        for pid, (n, snrs) in out.items()
    }


def _parse(text):
    """Table rows as dicts of strings, plus the summary line."""
    lines = text.strip().splitlines()
    assert lines[0].split() == HEADER
    rows = [dict(zip(HEADER, ln.split(), strict=True)) for ln in lines[1:-1]]
    return rows, lines[-1]


def test_events_equal_transit_label_count(run_dir):
    expected = _expected(run_dir)
    got = {r.planet_id: r for r in planets.planet_rows(run_dir)}
    assert set(got) == set(expected)
    for pid, (n, _, _) in expected.items():
        assert got[pid].events == n
    assert sum(n for n, _, _ in expected.values()) == len(_labels(run_dir)) > 0


def test_snr_matches_independent_recomputation(run_dir):
    expected = _expected(run_dir)
    got = {r.planet_id: r for r in planets.planet_rows(run_dir)}
    assert max(s for _, s, _ in expected.values()) > 1.0  # the fixture exercises real signal
    for pid, (_, stack, peak) in expected.items():
        assert got[pid].snr_stack == pytest.approx(stack, rel=1e-9, abs=1e-12)
        assert got[pid].snr_max == pytest.approx(peak, rel=1e-9, abs=1e-12)
    rows, _ = _parse(planets.format_planets(planets.planet_rows(run_dir)))
    for r in rows:
        assert r["snr_stack"] == f"{expected[int(r['planet_id'])][1]:.1f}"
        assert r["snr_max"] == f"{expected[int(r['planet_id'])][2]:.1f}"


def test_rows_sorted_by_snr_then_planet_id(run_dir):
    rows, _ = _parse(planets.format_planets(planets.planet_rows(run_dir)))
    key = [(-float(r["snr_stack"]), int(r["planet_id"])) for r in rows]
    assert key == sorted(key)


def test_columns_are_period_depth_duration_in_display_units(run_dir):
    pl = pq.read_table(run_dir / "planets.parquet").to_pydict()
    by_id = {pl["planet_id"][i]: i for i in range(len(pl["planet_id"]))}
    rows, _ = _parse(planets.format_planets(planets.planet_rows(run_dir)))
    for r in rows:
        i = by_id[int(r["planet_id"])]
        assert int(r["star"]) == pl["star_id"][i]
        assert float(r["period_d"]) == pytest.approx(pl["period_s"][i] / 86400, abs=5e-4)
        assert re.fullmatch(r"\d\.\d{5}", r["depth"])
        assert float(r["depth"]) == pytest.approx(pl["depth"][i], abs=5e-6)
        assert float(r["dur_h"]) == pytest.approx(pl["duration_s"][i] / 3600, abs=5e-3)


def test_planet_with_no_in_night_event_prints_zero(run_dir, tmp_path):
    clone = _clone(run_dir, tmp_path)

    def add(d):
        d["planet_id"].append(999)
        d["star_id"].append(d["star_id"][0])
        d["period_s"].append(900.0 * 86400.0)
        d["epoch_s"].append(450.0 * 86400.0)  # first transit long after the survey ends
        d["depth"].append(0.01)
        d["duration_s"].append(7200.0)

    _rewrite(clone / "planets.parquet", add)
    rows, _ = _parse(planets.format_planets(planets.planet_rows(clone)))
    row = next(r for r in rows if r["planet_id"] == "999")
    assert row["events"] == "0"
    assert row["snr_stack"] == "0.0" and row["snr_max"] == "0.0"


def test_unobservable_label_contributes_zero(run_dir, tmp_path):
    clone = _clone(run_dir, tmp_path)
    before = {r.planet_id: r for r in planets.planet_rows(run_dir)}
    # the strongest single event of the strongest multi-event planet
    pid = max((r for r in before.values() if r.events >= 2), key=lambda r: r.snr_stack).planet_id
    victim = {}

    def zero_one(d):
        best = -1.0
        for i in range(len(d["label_id"])):
            prm = json.loads(d["params"][i])
            if d["kind"][i] == "transit" and prm["planet_id"] == pid:
                if prm["footprint_samples"] > 0 and prm["depth"] > best:
                    best, victim["i"] = prm["depth"], i
        prm = json.loads(d["params"][victim["i"]])
        prm["footprint_samples"] = 0
        d["params"][victim["i"]] = json.dumps(prm, sort_keys=True)

    _rewrite(clone / "truth.parquet", zero_one)
    after = {r.planet_id: r for r in planets.planet_rows(clone)}
    assert after[pid].events == before[pid].events  # still counted as an event
    assert after[pid].snr_stack < before[pid].snr_stack  # but it no longer adds signal
    # recompute independently without that label
    assert after[pid].snr_stack == pytest.approx(
        _expected(clone)[pid][1], rel=1e-9, abs=1e-12
    )


def test_a_label_of_zero_footprint_alone_gives_snr_zero(run_dir, tmp_path):
    clone = _clone(run_dir, tmp_path)
    single = next(r for r in planets.planet_rows(run_dir) if r.events == 1 and r.snr_stack > 0)

    def zero(d):
        for i in range(len(d["label_id"])):
            prm = json.loads(d["params"][i])
            if d["kind"][i] == "transit" and prm["planet_id"] == single.planet_id:
                prm["footprint_samples"] = 0
                d["params"][i] = json.dumps(prm, sort_keys=True)

    _rewrite(clone / "truth.parquet", zero)
    row = next(r for r in planets.planet_rows(clone) if r.planet_id == single.planet_id)
    assert row.events == 1 and row.snr_stack == 0.0 and row.snr_max == 0.0


def test_filters_drop_rows_but_not_the_summary(run_dir):
    all_rows = planets.planet_rows(run_dir)
    full = planets.format_planets(all_rows)
    _, summary_full = _parse(full)
    n_two = sum(1 for r in all_rows if r.events >= 2)
    n_strong = sum(1 for r in all_rows if r.events >= 2 and r.snr_stack >= 7)
    assert summary_full == (
        f"{len(all_rows)} planets; {n_two} with >= 2 events; "
        f"{n_strong} with >= 2 events and snr_stack >= 7"
    )
    assert 0 < n_two < len(all_rows)

    cut = planets.format_planets(all_rows, min_events=2, min_snr=1.0)
    rows, summary = _parse(cut)
    assert summary == summary_full
    want = [r.planet_id for r in all_rows if r.events >= 2 and r.snr_stack >= 1.0]
    assert sorted(int(r["planet_id"]) for r in rows) == sorted(want)
    assert 0 < len(rows) < len(all_rows)

    only_events = planets.format_planets(all_rows, min_events=2)
    assert len(_parse(only_events)[0]) == n_two

    none = planets.format_planets(all_rows, min_events=10_000)
    assert _parse(none) == ([], summary_full)


def test_cli_output_is_byte_identical_across_invocations(run_dir, capsys):
    argv = ["planets", "--run", str(run_dir)]
    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    assert capsys.readouterr().out == first
    assert first.splitlines()[0].split() == HEADER


def test_cli_filter_flags(run_dir, capsys):
    assert main(["planets", "--run", str(run_dir), "--min-events", "2", "--min-snr", "1"]) == 0
    rows, summary = _parse(capsys.readouterr().out)
    assert rows and all(int(r["events"]) >= 2 and float(r["snr_stack"]) >= 0.95 for r in rows)
    assert "planets;" in summary


def test_missing_run_exits_2(tmp_path, capsys):
    assert main(["planets", "--run", str(tmp_path / "nope")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("skywatch: error:") and "nope" in err


def test_planets_module_does_not_import_matplotlib():
    with open(planets.__file__) as f:
        assert "matplotlib" not in f.read()
