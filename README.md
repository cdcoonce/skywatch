# skywatch

[![ci](https://github.com/cdcoonce/skywatch/actions/workflows/ci.yml/badge.svg)](https://github.com/cdcoonce/skywatch/actions/workflows/ci.yml)

A synthetic sky survey where detectors compete to tell science from glitches.

## What it is

skywatch simulates a small telescope array watching a field of stars night after night. Each star
yields a light curve, and the simulator hides two kinds of event in them: real astrophysics
(transits, flares, eclipses, pulsations) and instrument or weather glitches (cosmic rays, clouds,
tracking drift, hot pixels, dropouts). Every injected event is recorded as ground truth at the
moment it is placed. Detectors read the curves blind, and a scorer reports what each one caught,
missed, and mistook for something else. The scoreboard and the world behind it are the product.

Full plan: [docs/design.md](docs/design.md).

## How this project is run

skywatch is an agent-driven project. An AI agent sets direction, writes the design, orchestrates worker agents that build each slice, reviews their work and lands it. Charles Coonce sets the constraints, reviews what ships, and can reverse any decision. Decisions and their alternatives are recorded in docs/log/ as they are made. Commits are not attributed to individual models.

## Status

**Day 1 (2026-10-10):** the whole v0 pipeline exists and three detectors are
scored on two survey seeds. `skywatch simulate` builds a field of stars over
several nights with every science event and glitch from the design injected
and labelled; `detect` runs a registered detector (`rolling_z`, the
predecessor's method; `median_residual`; `field_common_mode`); `score` matches
detections to truth and `report` prints a per-kind scoreboard that separates
finding an event from timing it from naming it. Findings so far:
[0002](docs/log/0002-the-first-finding-is-about-the-scorer.md) (the scorer, not
the detector, needed fixing),
[0003](docs/log/0003-two-more-detectors-and-what-the-night-gap-does.md) (the
two new detectors trade recall for span quality; only the field detector can
name a cloud; windowing across the night gap is the baseline's biggest flaw),
[0004](docs/log/0004-two-fixes-and-what-each-cost.md) (per-night windows
and scorer edge tolerance; rolling_z's seed-7 false alarms went from 199 to 7,
with the recall cost split into withdrawn fake credit and a real first-window
blind spot), and
[0005](docs/log/0005-the-drift-the-warm-up-and-the-labels-nobody-could-see.md)
(the field detector now catches the held drift with a causal baseline, a
growing window recovers the first-window events, and truth labels that touch
no sample are excluded from the denominator and shown as `unobs`).

`skywatch view` draws one star's night with the injected truth and every
detector's spans on it. This is seed 7, star 33, night 1, where only the field
detector names the tracking drift:

![seed 7, star 33, night 1](docs/img/seed7-pick.png)

`skywatch site --spec docs/site.json` regenerates [docs/index.html](docs/index.html),
a single static page with this picture and two more beside the scoreboard.
[0006](docs/log/0006-the-first-picture-and-what-it-found.md) records the
picture and the finding it produced: fragments of an event a detector had
already found were being booked as false alarms, and now corroborate instead.

`skywatch site --spec docs/site.json` generates [docs/index.html](docs/index.html),
the committed views beside each run's scoreboard, regenerable from the seeds
with the commands it prints. `skywatch planets` lists which planets in a run a
periodic search could find, and it showed that the 3-night runs hold none, so
20-night surveys of the same seeds are the bench for `bls`, the box least
squares search across nights.
[0007](docs/log/0007-the-page-the-survey-that-could-not-be-searched-and-the-first-periodic-find.md)
records the page, the measurement and the detector:
`bls` bounds every event of every confirmable planet on the 20-night runs
except a planet on an eclipsing binary (issue #32), and finds a 1.4% dip over
five hours that no per-night detector sees:
[seed 7, star 185, night 6](docs/img/seed7-bls-star185-night6.png).

![seed 7, star 185, night 6](docs/img/seed7-bls-star185-night6.png)

Open: hosting (the page is built; publishing it is the owner's call) and #32.

## Develop

```
uv sync --group dev
uv run pytest
uv run ruff check .
```
