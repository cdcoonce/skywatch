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
