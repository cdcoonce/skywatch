"""`skywatch site`: the generated static page. Built on a tiny run made under tmp_path."""

import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from skywatch import run, site, view
from skywatch.cli import main
from skywatch.io import store
from skywatch.world.field import FieldConfig
from skywatch.world.survey import SurveyConfig

DETECTORS = ("field_common_mode", "median_residual", "rolling_z")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("site-runs")
    cfg = SurveyConfig(seed=3, n_nights=2, cadence_s=120, field=FieldConfig(n_stars=8))
    run_dir = run.simulate_run(cfg, root, "tiny")
    for name in DETECTORS:
        run.detect_run(run_dir, name)
    run.score_run(run_dir)
    return run_dir


def _stub_linked_files(tmp_path: Path, extra: dict) -> None:
    """Create the files the findings and honesty links name, as a real repo has them."""
    for f in extra.get("findings") or []:
        log = f.get("log") if isinstance(f, dict) else None
        if isinstance(log, str) and log and "/" not in log and log.endswith(".md"):
            (tmp_path / "docs" / "log").mkdir(parents=True, exist_ok=True)
            (tmp_path / "docs" / "log" / log).write_text("a build log\n")
    for h in extra.get("honesty") or []:
        link = h.get("link") if isinstance(h, dict) else None
        path = link.get("path") if isinstance(link, dict) else None
        if isinstance(path, str) and path and ".." not in path and not path.startswith("/"):
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("a linked file\n")


def _write_spec(tmp_path: Path, run_dir: Path, views=None, stub=True, **extra) -> Path:
    spec_dir = tmp_path / "docs"
    spec_dir.mkdir(exist_ok=True)
    if stub:
        try:
            _stub_linked_files(tmp_path, extra)
        except OSError:
            pass
    if views is None:
        views = [
            {"run": str(run_dir), "pick": True, "file": "img/pick.png", "caption": "the pick"},
            {
                "run": str(run_dir),
                "star": 2,
                "night": 1,
                "file": "img/explicit.png",
                "caption": "explicit",
            },
        ]
    spec = {"runs": [str(run_dir)], "views": views, **extra}
    path = spec_dir / "site.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture
def built(tmp_path, tiny):
    spec = _write_spec(tmp_path, tiny)
    out = site.generate(spec)
    return spec, out


class _Tables(HTMLParser):
    """Collect every table as {id: [[cell, ...], ...]} plus img srcs and tag names."""

    def __init__(self):
        super().__init__()
        self.tables: dict[str, list[list[str]]] = {}
        self.tags: list[tuple[str, dict]] = []
        self._table = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if tag == "table":
            self._table = self.tables.setdefault(a.get("id", f"t{len(self.tables)}"), [])
        elif tag == "tr" and self._table is not None:
            self._row = []
            self._table.append(self._row)
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr":
            self._row = None
        elif tag == "table":
            self._table = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _parse(path: Path) -> _Tables:
    p = _Tables()
    p.feed(path.read_text())
    return p


def _report_all_rows(run_dir: Path) -> dict[str, list[str]]:
    rows = {}
    for line in run.report_run(run_dir).splitlines():
        cells = line.split()
        if len(cells) == 12 and cells[1] == "all":
            rows[cells[0]] = [cells[0], *cells[2:]]
    return rows


def test_page_names_run_and_every_detector(built, tiny):
    _, out = built
    text = out.read_text()
    assert out.name == "index.html" and out.parent.name == "docs"
    assert "tiny" in text
    for name in DETECTORS:
        assert name in text
    assert "seed 3" in text and "8 stars" in text and "2 nights" in text


def test_all_rows_match_report_and_scorer(built, tiny):
    _, out = built
    tables = _parse(out).tables
    rows = tables["all-tiny"]
    assert rows[0] == [
        "detector", "tpB", "tpT", "corr", "fp", "fn", "unobs", "P", "R", "F1", "bounded"
    ]  # fmt: skip
    expected = _report_all_rows(tiny)
    assert {r[0] for r in rows[1:]} == set(DETECTORS)
    for row in rows[1:]:
        assert row == expected[row[0]]
    for s in store.read_scores(tiny):
        if s.kind == "all":
            row = next(r for r in rows[1:] if r[0] == s.detector)
            assert row[1:7] == [
                str(s.tp_bounded), str(s.tp_touched), str(s.corroborating),
                str(s.fp), str(s.fn), str(s.unobservable),
            ]  # fmt: skip
            assert row[7:] == [f"{s.precision:.3f}", f"{s.recall:.3f}", f"{s.f1:.3f}",
                               f"{s.bounded_rate:.3f}"]  # fmt: skip


def test_every_kind_table_follows_report_row_order(built, tiny):
    _, out = built
    rows = _parse(out).tables["kinds-tiny"]
    full = [line.split() for line in run.report_run(tiny).splitlines()[2:]]
    full = [c for c in full if len(c) == 12]
    assert rows[1:] == full
    assert "<details>" in out.read_text() and "<summary>Every kind</summary>" in out.read_text()


def test_pick_view_is_the_pick_the_cli_makes(built, tiny, tmp_path, capsys):
    spec, _ = built
    data = site.assemble(spec)
    pick = view.pick_view(tiny)
    head = data.figures[0]
    assert (head.star_id, head.night) == (pick.star_id, pick.night)
    cli_png = tmp_path / "cli-pick.png"
    assert main(["view", "--run", str(tiny), "--pick", "--out", str(cli_png)]) == 0
    assert (spec.parent / "img" / "pick.png").read_bytes() == cli_png.read_bytes()


def test_explicit_star_night_renders_like_the_cli(built, tiny, tmp_path):
    spec, _ = built
    cli_png = tmp_path / "cli-explicit.png"
    args = ["view", "--run", str(tiny), "--star", "2", "--night", "1", "--out", str(cli_png)]
    assert main(args) == 0
    assert (spec.parent / "img" / "explicit.png").read_bytes() == cli_png.read_bytes()


def test_headline_first_then_scoreboard_then_rest_then_regenerate(built):
    _, out = built
    text = out.read_text()
    order = [
        text.index("What you are looking at"),
        text.index("the pick"),
        text.index(">Scoreboard<"),
        text.index("explicit"),
        text.index(">Regenerate<"),
        text.index("Generated by"),
    ]
    assert order == sorted(order)
    assert text.count("<h1>skywatch</h1>") == 1
    assert "https://github.com/cdcoonce/skywatch" in text
    assert "A synthetic sky survey where detectors compete to tell science from glitches." in text


def test_regenerate_block_uses_manifest_config(built, tiny):
    _, out = built
    text = out.read_text()
    pre = re.search(r"<pre[^>]*>(.*?)</pre>", text, re.DOTALL).group(1)
    assert "skywatch simulate --seed 3 --stars 8 --nights 2 --cadence 120 --run-id tiny" in pre
    for name in DETECTORS:
        assert f"skywatch detect --run {tiny} --detector {name}" in pre
    assert f"skywatch score --run {tiny}" in pre
    assert pre.strip().splitlines()[-1] == "skywatch site --spec docs/site.json"


def test_no_script_no_stylesheet_relative_images(built):
    _, out = built
    tags = _parse(out).tags
    names = [t for t, _ in tags]
    assert "script" not in names
    assert not any(t == "link" and a.get("rel") == "stylesheet" for t, a in tags)
    srcs = [a["src"] for t, a in tags if t == "img"]
    assert srcs == ["img/pick.png", "img/explicit.png"]
    for src in srcs:
        assert not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", src) and not src.startswith(("/", "//"))
    assert "<script" not in out.read_text().lower()


def test_no_timestamps_from_the_manifest(built, tiny):
    _, out = built
    created = store.read_manifest(tiny)["created_at"]
    assert created not in out.read_text() and created[:10] not in out.read_text()


def test_two_generations_are_byte_identical(built):
    spec, out = built
    first = out.read_bytes()
    pngs = {p.name: p.read_bytes() for p in (spec.parent / "img").glob("*.png")}
    assert site.generate(spec) == out
    assert out.read_bytes() == first
    assert {p.name: p.read_bytes() for p in (spec.parent / "img").glob("*.png")} == pngs


def test_caption_is_escaped(tmp_path, tiny):
    views = [{"run": str(tiny), "pick": True, "file": "p.png", "caption": "a <b>bold</b> & more"}]
    out = site.generate(_write_spec(tmp_path, tiny, views))
    text = out.read_text()
    assert "a &lt;b&gt;bold&lt;/b&gt; &amp; more" in text
    assert "<b>bold</b>" not in text


def test_out_option_and_relative_src(tmp_path, tiny):
    spec = _write_spec(tmp_path, tiny)
    out = tmp_path / "elsewhere" / "page.html"
    assert main(["site", "--spec", str(spec), "--out", str(out)]) == 0
    srcs = [a["src"] for t, a in _parse(out).tags if t == "img"]
    assert srcs == ["../docs/img/pick.png", "../docs/img/explicit.png"]


def _fails_before_writing(tmp_path, spec, capsys, message):
    before = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert main(["site", "--spec", str(spec)]) == 2
    assert message in capsys.readouterr().err
    assert sorted(p for p in tmp_path.rglob("*") if p.is_file()) == before


def test_missing_run_fails_before_writing(tmp_path, tiny, capsys):
    spec = _write_spec(tmp_path, tiny)
    data = json.loads(spec.read_text())
    data["runs"].append(str(tmp_path / "nope"))
    spec.write_text(json.dumps(data))
    _fails_before_writing(tmp_path, spec, capsys, "nope")


def test_view_of_missing_run_fails_before_writing(tmp_path, tiny, capsys):
    views = [{"run": str(tmp_path / "ghost"), "pick": True, "file": "a.png", "caption": "x"}]
    _fails_before_writing(tmp_path, _write_spec(tmp_path, tiny, views), capsys, "ghost")


@pytest.mark.parametrize(
    "view_args",
    [{}, {"star": 1}, {"night": 0}, {"pick": False}, {"pick": False, "star": 1}],
)
def test_view_needs_pick_or_star_and_night(tmp_path, tiny, capsys, view_args):
    views = [{"run": str(tiny), "file": "a.png", "caption": "x", **view_args}]
    _fails_before_writing(tmp_path, _write_spec(tmp_path, tiny, views), capsys, "pick")


@pytest.mark.parametrize(
    ("file", "message"),
    [
        ("../out.png", "stay inside"),
        ("/abs/out.png", "file"),
        ("img/../../out.png", "stay inside"),
        ("http://x/a.png", "file"),
        ("a\\b.png", "plain relative path"),
        ("C:x.png", "plain relative path"),
        ("/abs.png", "plain relative path"),
        ("a//b.png", "stay inside"),
        ("./a.png", "stay inside"),
        ("a/./b.png", "stay inside"),
        ("x.txt", "must end in .png or .svg"),
    ],
)
def test_file_outside_spec_dir_fails(tmp_path, tiny, capsys, file, message):
    views = [{"run": str(tiny), "pick": True, "file": file, "caption": "x"}]
    _fails_before_writing(tmp_path, _write_spec(tmp_path, tiny, views), capsys, message)


def test_two_views_sharing_a_file_fail_before_writing(tmp_path, tiny, capsys):
    views = [
        {"run": str(tiny), "pick": True, "file": "same.png", "caption": "a"},
        {"run": str(tiny), "star": 2, "night": 1, "file": "same.png", "caption": "b"},
    ]
    _fails_before_writing(
        tmp_path, _write_spec(tmp_path, tiny, views), capsys, "two views write the same file"
    )


def test_empty_runs_fail_before_writing(tmp_path, tiny, capsys):
    spec = _write_spec(tmp_path, tiny, runs=[])
    _fails_before_writing(tmp_path, spec, capsys, "runs")


def test_empty_views_fail_before_writing(tmp_path, tiny, capsys):
    spec = _write_spec(tmp_path, tiny, views=[])
    _fails_before_writing(tmp_path, spec, capsys, "views")


def _pngs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png"))


def test_out_that_is_a_directory_fails_before_drawing(tmp_path, tiny, capsys):
    spec = _write_spec(tmp_path, tiny)
    target = tmp_path / "adir"
    target.mkdir()
    assert main(["site", "--spec", str(spec), "--out", str(target)]) == 2
    assert "adir" in capsys.readouterr().err
    assert _pngs(tmp_path) == [] and not (spec.parent / "index.html").exists()


def test_out_under_a_regular_file_fails_before_drawing(tmp_path, tiny, capsys):
    spec = _write_spec(tmp_path, tiny)
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory")
    out = blocker / "sub" / "page.html"
    assert main(["site", "--spec", str(spec), "--out", str(out)]) == 2
    assert "afile" in capsys.readouterr().err
    assert _pngs(tmp_path) == [] and blocker.read_text() == "not a directory"


def test_pick_is_consulted_and_matches_the_cli(tmp_path, tiny, monkeypatch):
    real = view.pick_view(tiny)
    assert (real.star_id, real.night) == (0, 0)  # the tiny run's own pick is the default cell
    chosen = view.Pick(star_id=3, night=1, kinds=("x",), n_detections=0)
    monkeypatch.setattr(view, "pick_view", lambda run_dir, detectors=None: chosen)
    spec = _write_spec(tmp_path, tiny)
    out = site.generate(spec)
    head = site.assemble(spec).figures[0]
    assert (head.star_id, head.night) == (3, 1)
    assert "tiny \u00b7 star 3 \u00b7 night 1" in out.read_text()
    cli_png = tmp_path / "cli-pick.png"
    assert main(["view", "--run", str(tiny), "--pick", "--out", str(cli_png)]) == 0
    assert (spec.parent / "img" / "pick.png").read_bytes() == cli_png.read_bytes()


def test_page_is_written_after_the_pictures(tmp_path, tiny, monkeypatch):
    spec = _write_spec(tmp_path, tiny)

    def boom(data, out):
        raise RuntimeError("render failed")

    monkeypatch.setattr(view, "render_view", boom)
    with pytest.raises(RuntimeError, match="render failed"):
        site.generate(spec)
    assert not (spec.parent / "index.html").exists()


def _pre(text: str) -> str:
    return re.search(r"<pre[^>]*>(.*?)</pre>", text, re.DOTALL).group(1)


def test_regenerate_adds_out_only_for_runs_outside_a_runs_folder(tmp_path, tiny, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(tiny, tmp_path / "runs" / "under")
    views = [
        {"run": str(tiny), "pick": True, "file": "a.png", "caption": "a"},
        {"run": "runs/under", "pick": True, "file": "b.png", "caption": "b"},
    ]
    spec = _write_spec(tmp_path, tiny, views, runs=[str(tiny), "runs/under"])
    pre = _pre(site.generate(spec).read_text())
    sim = [ln for ln in pre.splitlines() if ln.startswith("skywatch simulate")]
    assert len(sim) == 2
    assert sim[0].endswith(f"--run-id tiny --out {tiny.parent.as_posix()}")
    assert sim[1].endswith("--run-id under") and "--out" not in sim[1]


def test_run_figure_and_regenerate_order_follow_the_spec(tmp_path, tiny):
    runs_dir = tmp_path / "copies"
    zulu, alpha = runs_dir / "zulu", runs_dir / "alpha"
    shutil.copytree(tiny, zulu)
    shutil.copytree(tiny, alpha)
    views = [
        {"run": str(zulu), "star": 1, "night": 0, "file": "one.png", "caption": "first"},
        {"run": str(alpha), "star": 2, "night": 1, "file": "two.png", "caption": "second"},
        {"run": str(zulu), "star": 3, "night": 1, "file": "three.png", "caption": "third"},
    ]
    spec = _write_spec(tmp_path, tiny, views, runs=[str(zulu), str(alpha)])
    text = site.generate(spec).read_text()
    labels = [
        "zulu \u00b7 star 1 \u00b7 night 0",
        "alpha \u00b7 star 2 \u00b7 night 1",
        "zulu \u00b7 star 3 \u00b7 night 1",
    ]
    positions = [text.index(f'<span class="meta">{label}') for label in labels]
    assert positions == sorted(positions)
    assert text.index("<h3>zulu</h3>") < text.index("<h3>alpha</h3>")
    sim = [ln for ln in _pre(text).splitlines() if ln.startswith("skywatch simulate")]
    assert [ln.split("--run-id ")[1].split()[0] for ln in sim] == ["zulu", "alpha"]


def test_meta_line_and_figure_label_are_exact(built):
    _, out = built
    text = out.read_text()
    assert "seed 3 \u00b7 8 stars \u00b7 2 nights \u00b7 120 s cadence" in text
    assert '<span class="meta">tiny \u00b7 star 2 \u00b7 night 1</span>' in text


def test_unscored_run_and_bad_star_fail_before_writing(tmp_path, tiny, capsys):
    views = [{"run": str(tiny), "star": 999, "night": 0, "file": "a.png", "caption": "x"}]
    _fails_before_writing(tmp_path, _write_spec(tmp_path, tiny, views), capsys, "star")


def test_missing_spec_returns_2(tmp_path, capsys):
    assert main(["site", "--spec", str(tmp_path / "none.json")]) == 2
    assert "none.json" in capsys.readouterr().err


def test_site_module_does_not_import_matplotlib(tiny, tmp_path):
    spec = _write_spec(tmp_path, tiny)
    code = (
        "import sys\n"
        "from skywatch import site, cli\n"
        f"data = site.assemble({str(spec)!r})\n"
        "site.render_html(data)\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


# ------------------------------------------------------------------ findings and honesty

LOG_URL = "https://github.com/cdcoonce/skywatch/blob/main/docs/log/"
FINDINGS = [
    {"text": "first finding, 12 of 34", "log": "0002-the-first.md"},
    {"text": "second finding", "log": "0007-the-second.md"},
]
HONESTY = [
    {"text": "truth is written at injection"},
    {
        "text": "seeds are scored once",
        "link": {"label": "the seed ledger", "path": "docs/seeds.md"},
    },
]


def _hrefs(path: Path) -> list[str]:
    return [a["href"] for t, a in _parse(path).tags if t == "a" and "href" in a]


def test_findings_section_sits_between_the_explainer_and_the_scoreboard(tmp_path, tiny):
    out = site.generate(_write_spec(tmp_path, tiny, findings=FINDINGS, honesty=HONESTY))
    text = out.read_text()
    order = [
        text.index("What you are looking at"),
        text.index("the pick"),
        text.index(">What it found<"),
        text.index("first finding"),
        text.index("second finding"),
        text.index(">How it keeps itself honest<"),
        text.index("truth is written at injection"),
        text.index("seeds are scored once"),
        text.index(">Scoreboard<"),
    ]
    assert order == sorted(order)
    assert text.count(">What it found<") == 1 and text.count(">How it keeps itself honest<") == 1
    assert '<ul class="findings">' in text and '<ul class="honest">' in text


def test_each_finding_links_to_its_log_on_github(tmp_path, tiny):
    out = site.generate(_write_spec(tmp_path, tiny, findings=FINDINGS))
    text = out.read_text()
    hrefs = _hrefs(out)
    assert LOG_URL + "0002-the-first.md" in hrefs and LOG_URL + "0007-the-second.md" in hrefs
    assert hrefs.index(LOG_URL + "0002-the-first.md") < hrefs.index(LOG_URL + "0007-the-second.md")
    first = text[text.index("first finding") : text.index("second finding")]
    assert f'<a href="{LOG_URL}0002-the-first.md">Build log 0002</a>' in first
    assert "0007-the-second.md" not in first


def test_honesty_item_links_into_the_repo(tmp_path, tiny):
    out = site.generate(_write_spec(tmp_path, tiny, honesty=HONESTY))
    text = out.read_text()
    assert "https://github.com/cdcoonce/skywatch/blob/main/docs/seeds.md" in _hrefs(out)
    assert ">the seed ledger</a>" in text
    assert "truth is written at injection" in text


def test_findings_and_honesty_text_is_escaped(tmp_path, tiny):
    findings = [{"text": "a <b>bold</b> & more", "log": "0002-x.md"}]
    honesty = [{"text": "<i>no</i> & yes", "link": {"label": "<u>l</u> & m", "path": "docs/s.md"}}]
    text = site.generate(
        _write_spec(tmp_path, tiny, findings=findings, honesty=honesty)
    ).read_text()
    assert "a &lt;b&gt;bold&lt;/b&gt; &amp; more" in text
    assert "&lt;i&gt;no&lt;/i&gt; &amp; yes" in text
    assert "&lt;u&gt;l&lt;/u&gt; &amp; m" in text
    for raw in ("<b>bold</b>", "<i>no</i>", "<u>l</u>"):
        assert raw not in text


def test_link_targets_and_log_numbers_are_escaped(tmp_path, tiny):
    findings = [{"text": "t", "log": '<b>"-x.md'}]
    honesty = [{"text": "t", "link": {"label": "l", "path": 'docs/<i>"s.md'}}]
    out = site.generate(_write_spec(tmp_path, tiny, findings=findings, honesty=honesty))
    text = out.read_text()
    for raw in ("<b>", "<i>"):
        assert raw not in text
    assert [h for h in _hrefs(out) if "x.md" in h or "s.md" in h] == [
        LOG_URL + "%3Cb%3E%22-x.md",
        "https://github.com/cdcoonce/skywatch/blob/main/docs/" + "%3Ci%3E%22s.md",
    ]
    assert ">Build log &lt;b&gt;&quot;</a>" in text


@pytest.mark.parametrize("extra", [{}, {"findings": []}])
def test_no_findings_omits_the_section(tmp_path, tiny, extra):
    text = site.generate(_write_spec(tmp_path, tiny, honesty=HONESTY, **extra)).read_text()
    assert "What it found" not in text
    assert "How it keeps itself honest" in text


@pytest.mark.parametrize("extra", [{}, {"honesty": []}])
def test_no_honesty_omits_the_section(tmp_path, tiny, extra):
    text = site.generate(_write_spec(tmp_path, tiny, findings=FINDINGS, **extra)).read_text()
    assert "How it keeps itself honest" not in text
    assert "What it found" in text


def test_spec_without_either_section_has_neither_heading(built):
    _, out = built
    text = out.read_text()
    assert "What it found" not in text and "How it keeps itself honest" not in text


def test_findings_page_is_byte_identical_across_generations(tmp_path, tiny):
    spec = _write_spec(tmp_path, tiny, findings=FINDINGS, honesty=HONESTY)
    out = site.generate(spec)
    first = out.read_bytes()
    site.generate(spec)
    assert out.read_bytes() == first
    assert "<script" not in out.read_text().lower()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"findings": "nope"}, "findings must be a list"),
        ({"findings": ["nope"]}, "finding 1: must be an object"),
        ({"findings": [{"log": "0002-x.md"}]}, "finding 1: text"),
        ({"findings": [{"text": "", "log": "0002-x.md"}]}, "finding 1: text"),
        ({"findings": [{"text": "t"}]}, "finding 1: log"),
        ({"findings": [{"text": "t", "log": "../x.md"}]}, "finding 1: log"),
        ({"findings": [{"text": "t", "log": "a/b.md"}]}, "finding 1: log"),
        ({"findings": [{"text": "t", "log": "x.txt"}]}, "finding 1: log"),
        ({"findings": [{"text": "t", "log": "http:x.md"}]}, "finding 1: log"),
        ({"honesty": "nope"}, "honesty must be a list"),
        ({"honesty": [{}]}, "honesty 1: text"),
        ({"honesty": [{"text": "t", "link": "docs/seeds.md"}]}, "honesty 1: link"),
        ({"honesty": [{"text": "t", "link": {"label": "", "path": "docs/s.md"}}]}, "label"),
        ({"honesty": [{"text": "t", "link": {"label": "l", "path": "../s.md"}}]}, "path"),
        ({"honesty": [{"text": "t", "link": {"label": "l", "path": "/s.md"}}]}, "path"),
        ({"honesty": [{"text": "t", "link": {"label": "l", "path": "https://x/s.md"}}]}, "path"),
        ({"findings": [{"text": "   ", "log": "0002-x.md"}]}, "finding 1: text"),
        ({"honesty": [{"text": "  "}]}, "honesty 1: text"),
        ({"findings": [{"text": "t", "log": "a\\b.md"}]}, "finding 1: log"),
        ({"honesty": [{"text": "t", "link": {"label": "l", "path": "docs\\s.md"}}]}, "path"),
        ({"honesty": [{"text": "t", "link": {"label": "l", "path": "docs//s.md"}}]}, "path"),
        ({"honesty": [{"text": "t", "link": {"label": "l", "path": "docs/./s.md"}}]}, "path"),
        (
            {"honesty": [{"text": "t", "link": {"label": "l", "path": "javascript:alert(1)"}}]},
            "path",
        ),
        ({"honesty": ["nope"]}, "honesty 1: must be an object"),
        ({"honesty": [{"text": "t", "link": {}}]}, "honesty 1: link: label"),
        ({"honesty": [{"text": "t", "link": ""}]}, "honesty 1: link"),
        ({"findings": [{"text": 7, "log": "0002-x.md"}]}, "finding 1: text"),
        ({"findings": [{"text": "t", "log": 7}]}, "finding 1: log"),
        ({"findings": [{"text": "t", "log": "0002-x.md", "href": "x"}]}, "finding 1: unknown key"),
        ({"honesty": [{"text": "t", "links": {}}]}, "honesty 1: unknown key"),
        (
            {"honesty": [{"text": "t", "link": {"label": "l", "path": "docs/s.md", "x": 1}}]},
            "honesty 1: link: unknown key",
        ),
        (
            {"findings": [{"text": "t", "log": "0002-x.md"}, {"text": "t", "log": "0003-y.md"}]},
            "finding 2: duplicate",
        ),
        ({"honesty": [{"text": "t"}, {"text": "t"}]}, "honesty 2: duplicate"),
    ],
)
def test_bad_findings_or_honesty_fail_before_writing(tmp_path, tiny, capsys, extra, message):
    _fails_before_writing(tmp_path, _write_spec(tmp_path, tiny, **extra), capsys, message)


def test_a_finding_whose_log_does_not_exist_fails_before_writing(tmp_path, tiny, capsys):
    findings = [{"text": "t", "log": "0002-x.md"}, {"text": "u", "log": "9999-nope.md"}]
    spec = _write_spec(tmp_path, tiny, findings=findings, stub=False)
    (tmp_path / "docs" / "log").mkdir()
    (tmp_path / "docs" / "log" / "0002-x.md").write_text("a build log\n")
    _fails_before_writing(tmp_path, spec, capsys, "finding 2: log 'docs/log/9999-nope.md'")


@pytest.mark.parametrize("path", ["docs/nope.md", "docs", "docs/log"])
def test_an_honesty_link_to_a_missing_file_fails_before_writing(tmp_path, tiny, capsys, path):
    honesty = [{"text": "t", "link": {"label": "l", "path": path}}]
    spec = _write_spec(tmp_path, tiny, honesty=honesty, stub=False)
    (tmp_path / "docs" / "log").mkdir()
    _fails_before_writing(tmp_path, spec, capsys, f"honesty 1: link: path {path!r}")


def test_link_paths_are_percent_quoted_in_the_href(tmp_path, tiny):
    honesty = [{"text": "t", "link": {"label": "l", "path": "docs/a b#c?.md"}}]
    out = site.generate(_write_spec(tmp_path, tiny, honesty=honesty))
    assert "https://github.com/cdcoonce/skywatch/blob/main/docs/a%20b%23c%3F.md" in _hrefs(out)


# The published spec: the page's own claims are checked against the logs they link to.

REPO = Path(__file__).resolve().parent.parent
PUBLISHED = json.loads((REPO / "docs" / "site.json").read_text())
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def missing_numbers(text: str, body: str) -> list[str]:
    """Numbers quoted in `text` that do not appear in `body` as whole figures.

    Weak for single-digit figures: they recur in every log, so a wrong one still matches.
    """
    found = (n.rstrip(",") for n in NUMBER.findall(text))
    return [n for n in found if not re.search(rf"(?<![\d.,]){re.escape(n)}(?!\d|\.\d|,\d)", body)]


@pytest.mark.parametrize(
    ("text", "body", "expected"),
    [
        ("14 to 2", "it fell from 14 to 2 here", []),
        ("3 of 35", "we saw 35 of 35", ["3"]),
        ("0.65 of it", "a median of 10.65 here", ["0.65"]),
        ("0.65 of it", "a 0 and a 65", ["0.65"]),
        ("1,234 sites", "1,2345 sites", ["1,234"]),
        ("312 alarms", "no figures at all", ["312"]),
    ],
)
def test_the_number_check_rejects_figures_the_log_does_not_contain(text, body, expected):
    assert missing_numbers(text, body) == expected


def test_published_spec_has_five_to_seven_findings_and_three_or_four_honesty_sentences():
    assert 5 <= len(PUBLISHED["findings"]) <= 7
    assert 3 <= len(PUBLISHED["honesty"]) <= 4


@pytest.mark.parametrize("finding", PUBLISHED["findings"], ids=lambda f: f["log"][:4])
def test_every_quoted_number_appears_verbatim_in_the_linked_log(finding):
    log = REPO / "docs" / "log" / finding["log"]
    assert log.is_file(), f"{finding['log']} is not a build log"
    body = log.read_text()
    numbers = NUMBER.findall(finding["text"])
    assert numbers, "a finding quotes at least one number from its log"
    missing = missing_numbers(finding["text"], body)
    assert not missing, f"{missing} not in {finding['log']}"


def test_published_findings_are_one_sentence_each():
    for finding in PUBLISHED["findings"]:
        text = finding["text"].strip()
        assert text.endswith(".") and len(text) < 420
        assert len(re.findall(r"[.!?](?:\s+[A-Z])", text)) == 0, text


def test_published_honesty_links_point_at_files_in_the_repo():
    for item in PUBLISHED["honesty"]:
        if "link" in item:
            assert (REPO / item["link"]["path"]).is_file(), item["link"]["path"]
