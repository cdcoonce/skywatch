import pytest

from skywatch import __version__
from skywatch.cli import main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def _sim(tmp_path):
    argv = ["simulate", "--seed", "3", "--stars", "8", "--nights", "1", "--cadence", "120"]
    assert main([*argv, "--out", str(tmp_path)]) == 0
    return tmp_path / "seed3-8s-1n"


def test_full_pipeline(tmp_path, capsys):
    run_dir = _sim(tmp_path)
    assert str(run_dir) in capsys.readouterr().out
    assert main(["detect", "--run", str(run_dir), "--detector", "rolling_z"]) == 0
    assert "detections" in capsys.readouterr().out
    assert main(["score", "--run", str(run_dir)]) == 0
    assert "rolling_z" in capsys.readouterr().out
    assert main(["report", "--run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "rolling_z" in out and "all" in out


def test_detect_flags(tmp_path, capsys):
    run_dir = _sim(tmp_path)
    capsys.readouterr()
    argv = ["detect", "--run", str(run_dir), "--detector", "rolling_z"]
    assert main([*argv, "--window", "10", "--z-threshold", "3"]) == 0


def test_bad_detector_returns_2(tmp_path, capsys):
    run_dir = _sim(tmp_path)
    capsys.readouterr()
    assert main(["detect", "--run", str(run_dir), "--detector", "nope"]) == 2
    assert "rolling_z" in capsys.readouterr().err


def test_overwrite_refused_returns_2(tmp_path):
    _sim(tmp_path)
    assert (
        main(
            [
                "simulate",
                "--seed",
                "3",
                "--stars",
                "8",
                "--nights",
                "1",
                "--cadence",
                "120",
                "--out",
                str(tmp_path),
            ]
        )
        == 2
    )


def test_load_all_registers_every_detector_module():
    from skywatch.detectors import REGISTRY, load_all

    load_all()
    assert "rolling_z" in REGISTRY


def _detected_run(tmp_path):
    run_dir = _sim(tmp_path)
    for det in ("rolling_z", "field_common_mode"):
        assert main(["detect", "--run", str(run_dir), "--detector", det]) == 0
    return run_dir


def test_view_pick_writes_png(tmp_path, capsys):
    run_dir = _detected_run(tmp_path)
    assert main(["score", "--run", str(run_dir)]) == 0
    capsys.readouterr()
    out = tmp_path / "x.png"
    assert main(["view", "--run", str(run_dir), "--pick", "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "picked star" in stdout and "night 0" in stdout and str(out) in stdout
    assert out.read_bytes().startswith(b"\x89PNG")


def test_view_explicit_star_default_out_and_svg(tmp_path, capsys):
    run_dir = _detected_run(tmp_path)  # detected but not scored
    capsys.readouterr()
    argv = ["view", "--run", str(run_dir), "--star", "0", "--night", "0"]
    assert main(argv) == 0
    assert (run_dir / "views" / "star0-night0.png").stat().st_size > 0
    svg = tmp_path / "a.svg"
    assert main([*argv, "--detectors", "rolling_z", "--out", str(svg)]) == 0
    assert svg.read_text().lstrip().startswith("<?xml")


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "--star and --night"),
        (["--star", "0"], "--star and --night"),
        (["--pick", "--star", "0"], "--pick"),
        (["--star", "0", "--night", "9"], "night"),
        (["--star", "999", "--night", "0"], "star 999"),
        (["--star", "0", "--night", "0", "--detectors", "nope"], "nope"),
        (["--star", "0", "--night", "0", "--detectors", ","], "empty"),
    ],
)
def test_view_bad_arguments_return_2(tmp_path, capsys, extra, message):
    run_dir = _detected_run(tmp_path)
    capsys.readouterr()
    assert main(["view", "--run", str(run_dir), *extra]) == 2
    assert message in capsys.readouterr().err


def test_view_missing_run_returns_2(tmp_path, capsys):
    assert main(["view", "--run", str(tmp_path / "nope"), "--pick"]) == 2
    assert "no run" in capsys.readouterr().err
