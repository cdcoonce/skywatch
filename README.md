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

**Day 1 (2026-10-10):** the whole v0 pipeline exists. `skywatch simulate` builds a
field of stars over several nights with every science event and glitch from
the design injected and labelled; `detect` runs a registered detector
(`rolling_z`, the predecessor's method, is the baseline); `score` matches
detections to truth and `report` prints the scoreboard. First measured finding:
the scorer, not the detector, was the thing that needed fixing. Read
[docs/log/0002](docs/log/0002-the-first-finding-is-about-the-scorer.md).

Next: a median-residual detector and a field common-mode detector, then a way
to look at a night.

## Develop

```
uv sync --group dev
uv run pytest
uv run ruff check .
```
