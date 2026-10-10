import pytest

from skywatch import __version__
from skywatch.cli import main


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


@pytest.mark.parametrize(
    "argv",
    [
        ["simulate", "--seed", "42", "--stars", "200", "--nights", "3", "--out", "runs/"],
        ["detect", "--run", "runs/x", "--detector", "rolling_z"],
        ["score", "--run", "runs/x"],
        ["report", "--run", "runs/x"],
    ],
)
def test_subcommands_not_implemented(argv):
    assert main(argv) == 2
