"""`skywatch site`: one self-contained static page from stored runs.

The spec (``docs/site.json``) says which runs and which pictures; this module assembles the
page's data, draws the pictures through the existing view renderer, and renders the HTML.
Assembly and HTML rendering never import matplotlib (only `view.render_view` does), and the
page carries no timestamp, so two generations from the same runs are byte-identical.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

from skywatch import view
from skywatch.io import store
from skywatch.scoring.metrics import ALL, SCOREBOARD_HEADER, scoreboard_cells

REPO_URL = "https://github.com/cdcoonce/skywatch"
DESCRIPTION = "A synthetic sky survey where detectors compete to tell science from glitches."
DEFAULT_CADENCE_S = 60
SITE_COMMAND = "skywatch site --spec docs/site.json"

HOW_TO_READ = (
    "Each picture is one star on one night. The line is the flux the telescope returned, "
    "with instrument-flagged samples drawn as hollow rings. The shaded bands under it are the "
    "truth the simulator injected, one colour per kind, which the detectors never see. Below "
    "the flux is one lane per detector, showing every span it reported on that star and night, "
    "coloured by how the scorer graded it: solid green bounded the event (IoU at least 0.3), "
    "hatched touched it, a blue outline corroborates an event the detector had already found, "
    "and a red outline matched no truth at all."
)
LEGEND = (
    "tpB/tpT = true positives bounded/touched, corr = corroborating, fp = false alarms, "
    "fn = missed, unobs = unobservable, excluded from fn and recall, "
    "P/R/F1 = precision, recall and F1, bounded = share of true positives that bounded the event"
)


class SpecError(ValueError):
    """The spec file is malformed; `main` reports it like any bad argument (exit 2)."""


@dataclass(frozen=True)
class ViewSpec:
    run: Path
    pick: bool
    star: int | None
    night: int | None
    file: str
    caption: str


@dataclass(frozen=True)
class Spec:
    path: Path
    runs: tuple[Path, ...]
    views: tuple[ViewSpec, ...]


@dataclass(frozen=True)
class RunSection:
    run_dir: Path
    run_id: str
    seed: int
    n_stars: int
    n_nights: int
    cadence_s: int
    detectors: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    """Every scoreboard row as `scoreboard_cells` formats it, in `report`'s order."""


@dataclass(frozen=True)
class Figure:
    spec: ViewSpec
    star_id: int
    night: int
    data: view.ViewData


@dataclass(frozen=True)
class SiteData:
    spec: Spec
    runs: tuple[RunSection, ...]
    figures: tuple[Figure, ...]


# ---------------------------------------------------------------------------- the spec


def _check_run(path: Path) -> None:
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"no run at {path} (manifest.json missing)")


def _check_file(n: int, file: object) -> str:
    where = f"view {n}: file"
    if not isinstance(file, str) or not file:
        raise SpecError(f"{where} must be a non-empty relative path")
    if ":" in file or "\\" in file or file.startswith("/"):
        raise SpecError(f"{where} {file!r} must be a plain relative path inside the spec's folder")
    if any(part in ("", ".", "..") for part in file.split("/")):
        raise SpecError(f"{where} {file!r} must stay inside the spec's folder")
    if Path(file).suffix.lower() not in (".png", ".svg"):
        raise SpecError(f"{where} {file!r} must end in .png or .svg")
    return file


def _int(n: int, key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"view {n}: {key} must be an integer, got {value!r}")
    return value


def _parse_view(n: int, raw: object) -> ViewSpec:
    if not isinstance(raw, dict):
        raise SpecError(f"view {n}: must be an object")
    run = raw.get("run")
    if not isinstance(run, str) or not run:
        raise SpecError(f"view {n}: run must be a run directory path")
    pick = raw.get("pick", False)
    if not isinstance(pick, bool):
        raise SpecError(f"view {n}: pick must be true or false")
    star, night = raw.get("star"), raw.get("night")
    if pick:
        if star is not None or night is not None:
            raise SpecError(f"view {n}: pick chooses the star and night; drop star and night")
    elif star is None or night is None:
        raise SpecError(f"view {n}: give pick: true, or both star and night")
    caption = raw.get("caption", "")
    if not isinstance(caption, str):
        raise SpecError(f"view {n}: caption must be text")
    return ViewSpec(
        Path(run),
        pick,
        None if pick else _int(n, "star", star),
        None if pick else _int(n, "night", night),
        _check_file(n, raw.get("file")),
        caption,
    )


def load_spec(path: Path) -> Spec:
    """Read and validate a site spec; raises before anything is written."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no spec at {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SpecError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: the spec must be a JSON object")
    runs_raw, views_raw = raw.get("runs"), raw.get("views")
    if (
        not isinstance(runs_raw, list)
        or not runs_raw
        or not all(isinstance(r, str) for r in runs_raw)
    ):
        raise SpecError(f"{path}: runs must be a non-empty list of run directory paths")
    if not isinstance(views_raw, list) or not views_raw:
        raise SpecError(f"{path}: views must be a non-empty list")
    runs = tuple(Path(r) for r in runs_raw)
    views = tuple(_parse_view(i + 1, v) for i, v in enumerate(views_raw))
    files = [v.file for v in views]
    if len(set(files)) != len(files):
        raise SpecError(f"{path}: two views write the same file")
    for run_dir in dict.fromkeys([*runs, *(v.run for v in views)]):
        _check_run(run_dir)
    return Spec(path, runs, views)


# ---------------------------------------------------------------------------- assembly


def _section(run_dir: Path) -> RunSection:
    if not (run_dir / "scores.parquet").exists():
        raise FileNotFoundError(f"no scores in {run_dir}; run `skywatch score` first")
    manifest = store.read_manifest(run_dir)
    cfg = manifest["config"]
    return RunSection(
        run_dir,
        run_dir.name,
        manifest["seed"],
        manifest["n_stars"],
        manifest["n_nights"],
        cfg["cadence_s"],
        tuple(store.detector_names(run_dir)),
        tuple(scoreboard_cells(s) for s in store.read_scores(run_dir)),
    )


def assemble(spec_path: Path) -> SiteData:
    """Validate the spec and gather everything the page shows. Writes nothing, draws nothing."""
    spec = load_spec(spec_path)
    runs = tuple(_section(r) for r in spec.runs)
    figures = []
    for v in spec.views:
        if v.pick:
            pick = view.pick_view(v.run)
            star, night = pick.star_id, pick.night
        else:
            assert v.star is not None and v.night is not None
            star, night = v.star, v.night
        figures.append(Figure(v, star, night, view.assemble_view(v.run, star, night)))
    return SiteData(spec, runs, tuple(figures))


# ---------------------------------------------------------------------------- HTML


CSS = """
:root { color-scheme: light dark; --bg: #fff; --fg: #1d2330; --muted: #5d6679;
  --rule: #d9dde6; --panel: #f5f6f9; --link: #1f5fbf; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171f; --fg: #e4e7ee; --muted: #9aa3b5; --rule: #2e3443;
    --panel: #1c212c; --link: #7fb0ff; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); line-height: 1.55;
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
a { color: var(--link); }
h1 { font-size: 2.2rem; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 1.35rem; margin: 40px 0 8px; padding-bottom: 6px;
  border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.05rem; margin: 24px 0 2px; font-family: ui-monospace, Menlo, Consolas, monospace; }
p { margin: 8px 0; }
.tagline { font-size: 1.15rem; margin: 0 0 8px; }
.links { display: flex; flex-wrap: wrap; gap: 4px 20px; margin: 8px 0 0; padding: 0; list-style: none; }
figure { margin: 20px 0; }
figure img { display: block; max-width: 100%; height: auto; background: #fff;
  border: 1px solid var(--rule); border-radius: 4px; }
figcaption { margin-top: 8px; }
.meta { display: block; color: var(--muted); font-size: 0.85rem; margin-top: 2px; }
.meta, td, th, pre, code { font-family: ui-monospace, Menlo, Consolas, monospace; }
.run-meta { color: var(--muted); margin: 0 0 8px; font-size: 0.9rem; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem;
  font-variant-numeric: tabular-nums; }
table.all { font-size: 1rem; }
th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--rule);
  white-space: nowrap; }
th { color: var(--muted); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
table.kinds th:nth-child(2), table.kinds td:nth-child(2) { text-align: left; }
tr.all-kind td { font-weight: 600; }
details { margin: 12px 0; }
summary { cursor: pointer; padding: 4px 0; }
.legend { color: var(--muted); font-size: 0.85rem; margin: 8px 0 0; }
pre { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  padding: 12px 14px; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; }
footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 0.9rem; }
"""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text)


def _cell(tag: str, text: str) -> str:
    return f"<{tag}>{escape(text)}</{tag}>"


def _table(table_id: str, css: str, header: tuple[str, ...], rows, row_class=None) -> str:
    out = [f'<div class="table-wrap"><table id="{escape(table_id)}" class="{css}">']
    out.append("<tr>" + "".join(_cell("th", h) for h in header) + "</tr>")
    for row in rows:
        cls = f' class="{row_class(row)}"' if row_class and row_class(row) else ""
        out.append(f"<tr{cls}>" + "".join(_cell("td", c) for c in row) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def _image_path(data: SiteData, fig: Figure) -> Path:
    return data.spec.path.parent / fig.spec.file


def _figure(data: SiteData, fig: Figure, out_dir: Path) -> str:
    src = Path(os.path.relpath(_image_path(data, fig).absolute(), out_dir.absolute())).as_posix()
    where = f"{fig.data.run_id} \u00b7 star {fig.star_id} \u00b7 night {fig.night}"
    alt = f"{where}: {fig.spec.caption}" if fig.spec.caption else where
    return (
        "<figure>\n"
        f'<img src="{escape(src)}" alt="{escape(alt)}" width="1200" height="800">\n'
        f"<figcaption>{escape(fig.spec.caption)}"
        f'<span class="meta">{escape(where)}</span></figcaption>\n'
        "</figure>"
    )


def _simulate_command(run: RunSection) -> str:
    parts = [
        "skywatch simulate",
        f"--seed {run.seed}",
        f"--stars {run.n_stars}",
        f"--nights {run.n_nights}",
    ]
    if run.cadence_s != DEFAULT_CADENCE_S:
        parts.append(f"--cadence {run.cadence_s}")
    parts.append(f"--run-id {run.run_id}")
    parent = run.run_dir.parent.as_posix()
    if parent not in ("runs", "."):
        parts.append(f"--out {parent}")
    return " ".join(parts)


def _scoreboard(run: RunSection) -> str:
    rid = _slug(run.run_id)
    meta = (
        f"seed {run.seed} \u00b7 {run.n_stars} stars \u00b7 {run.n_nights} nights "
        f"\u00b7 {run.cadence_s} s cadence"
    )
    all_header = tuple(h for h in SCOREBOARD_HEADER if h != "kind")
    all_rows = [r[:1] + r[2:] for r in run.rows if r[1] == ALL]
    return "\n".join(
        [
            f"<h3>{escape(run.run_id)}</h3>",
            f'<p class="run-meta">{escape(meta)}</p>',
            _table(f"all-{rid}", "all", all_header, all_rows),
            "<details>",
            "<summary>Every kind</summary>",
            _table(
                f"kinds-{rid}",
                "kinds",
                SCOREBOARD_HEADER,
                run.rows,
                lambda r: "all-kind" if r[1] == ALL else "",
            ),
            "</details>",
            f'<p class="legend">{escape(LEGEND)}</p>',
        ]
    )


def _regenerate(runs: tuple[RunSection, ...]) -> str:
    lines: list[str] = []
    for run in runs:
        lines.append(_simulate_command(run))
        lines += [
            f"skywatch detect --run {run.run_dir.as_posix()} --detector {name}"
            for name in run.detectors
        ]
        lines.append(f"skywatch score --run {run.run_dir.as_posix()}")
    lines.append(SITE_COMMAND)
    return "\n".join(lines)


def render_html(data: SiteData, out_path: Path | None = None) -> str:
    """The whole page as text. `out_path` only decides how image paths are written."""
    out_path = Path(out_path) if out_path else data.spec.path.parent / "index.html"
    out_dir = out_path.parent
    head, rest = data.figures[0], data.figures[1:]
    sections = [
        "<header>",
        "<h1>skywatch</h1>",
        f'<p class="tagline">{escape(DESCRIPTION)}</p>',
        (
            "<p>skywatch is an agent-driven project: an AI agent designs it, builds it and "
            "decides how it grows, while Charles Coonce sets the constraints and reviews what "
            "ships.</p>"
        ),
        '<ul class="links">',
        f'<li><a href="{REPO_URL}">Repository</a></li>',
        f'<li><a href="{REPO_URL}/blob/main/docs/design.md">Design</a></li>',
        f'<li><a href="{REPO_URL}/tree/main/docs/log">Build log</a></li>',
        "</ul>",
        "</header>",
        "<h2>What you are looking at</h2>",
        f"<p>{escape(HOW_TO_READ)}</p>",
        _figure(data, head, out_dir),
        "<h2>Scoreboard</h2>",
        *[_scoreboard(r) for r in data.runs],
    ]
    if rest:
        sections += ["<h2>More views</h2>", *[_figure(data, f, out_dir) for f in rest]]
    sections += [
        "<h2>Regenerate</h2>",
        "<p>Everything on this page rebuilds from the seeds with these commands.</p>",
        f"<pre>{escape(_regenerate(data.runs))}</pre>",
        "<footer>Generated by <code>skywatch site</code> from the runs above.</footer>",
    ]
    body = "\n".join(sections)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>skywatch</title>\n"
        f'<meta name="description" content="{escape(DESCRIPTION)}">\n'
        f"<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )


def generate(spec_path: Path, out_path: Path | None = None) -> Path:
    """Draw every picture (overwriting its file) and write the page; returns the page path.

    Everything is validated and assembled first, so a bad spec writes nothing.
    """
    data = assemble(spec_path)
    out_path = Path(out_path) if out_path else data.spec.path.parent / "index.html"
    html = render_html(data, out_path)
    for fig in data.figures:
        view.render_view(fig.data, _image_path(data, fig))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    return out_path
