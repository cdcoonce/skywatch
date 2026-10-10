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

v0 scaffold. No simulator, detectors or scoreboard yet. See docs/design.md for the plan and docs/log/ for progress.

## Develop

```
uv sync --group dev
uv run pytest
uv run ruff check .
```
