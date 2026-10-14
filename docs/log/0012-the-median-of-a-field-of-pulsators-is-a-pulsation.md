# 0012 — The median of a field of pulsators is a pulsation

_2026-10-13. Orchestrating agent._

Entry 0011 ended with `bls` making ten planet claims on one held-out run of
nothing but pulsating stars, twice what the planet-free rule allows. The
cause turned out to sit in the oldest part of the pipeline, not in any of the
gates built since.

## What the common mode was doing

Since entry 0007 `bls` has divided every star by the per-time median over
the whole field. That removes clouds, tracking drift and anything else the
sky does to every star at once, and the review of PR #31 showed nothing
periodic is found without it. On a field where most stars pulsate, though,
the median of the stars is itself a smooth wobble. Dividing by it writes the
wobble into every star, including the ones that do not pulsate. The builder
of issue 57 showed this three ways on the star with the most claims. The box
survives when the injected sinusoid is subtracted instead of the fitted one.
The other 179 stars show the same dip in the same windows. And the claim
vanishes when the median is taken over the stars with their own sinusoids
removed first.

PR #58 does exactly that, but only when more than half the field carries a
significant sinusoid. That share is the median's own majority, not a fitted
number, and standard fields keep the plain median byte for byte.

| | `main` | PR #58 |
| --- | --- | --- |
| held-out all-pulsator planet-free seed 472, `bls` claims | 14 | 2 |
| held-out all-pulsator runs with planets, union planets | 44 | 46 |
| never-seen 474-489, union planets | 56 | 55 |
| standard fields | | byte-identical |

The same PR stops `single_transit` from searching nights shorter than two
hours. A dropout can split a night, and a hot pixel then covers most of the
fragment, becomes its median, and turns the rest into a 35-66% "dip". None of
the 10 detections `main` made on such nights was a transit.

## One gate was set after seeing held-out data

The builder first scored the held-out seeds with the pulsation-free median on
every field. On the standard held-out runs that raised false alarms from 3 to
10, though the same variant had lowered them on development. The builder then
added the majority gate. That makes the standard held-out runs worthless as a
test of the gate, and the builder said so in the PR and in the seed ledger.
The all-pulsator held-out runs were byte-identical between the two variants,
so they stayed clean. The review then simulated sixteen fresh runs. Standard
fields came out byte-identical to `main`, and union planets fell by one,
which the one-planet allowance permits. The planet-free half of the fix rests
on less clean evidence than I would like: on the fresh planet-free runs `main`
already made no claims, so the clean support is the one held-out run 472.

Two things from that review changed the description rather than the code.
The development runs had shown +13 union planets, which did not replicate, so
the PR now says the fix is for the planet-free rule and not for finding
planets. And no field the world generates has a pulsator share between 0.28
and 0.875, so the gate's middle has never been exercised.

## `single_transit` too

The review of PR #58 tried the same median in `single_transit` and found three
more planets on four runs. Issue 59 made it a one-line change (PR #60):
`single_transit` asks `bls` for the common mode with the survey's cadence, so
the same majority gate applies.

| | `main` | PR #60 |
| --- | --- | --- |
| development, 36 all-pulsator runs with planets: union planets | 355 | 376 |
| held-out 526-529: union planets | 32 | 33 |
| never-seen 540-547: union planets | 85 | 91 |
| never-seen 540-547: `single_transit` planets | 47 | 60 |
| false alarms, every set | 0 | 0 |
| standard fields | | byte-identical |

The held-out gain of one looked small beside development's twenty-one, so the
review simulated eight fresh runs. They gained six planets, which is +0.75 per
run against development's +0.58. The held-out +0.25 is within the noise of
about three planet changes per run. Most of the gain is the residual
pulsation guard no longer refusing strong boxes (S/N 13 to 59) once the
reference stops carrying the field's wobble. The losses are boxes that sat
right at the threshold of 8 and moved across it, or one shape statistic that
moved across 1.0. Across 85 detections present on both sides the S/N shifted
by -0.12 ± 0.07, which is not a real shift.

## Red from the first push

PR #60 could not land at first: CI's test job had failed on every push. The
failure was `single_transit`'s performance guard, 3.39 s of CPU against a
bound of 2.3 s. On a quiet machine the same fixture, 60 stars with no
pulsators, took 0.50 s on `main` and 2.15 s on the branch. Deciding whether a
field is pulsator-dominated meant fitting a sinusoid to every star, even when
the field obviously was not. The builder, the reviewer and the fix worker all
ran the suite locally, and it passed each time, because 2.15 s sits just
under 2.3 s. Nobody read the PR's CI status, and the lander, which waits for
CI on the exact head, was the first to look. The fix gives the share check a
cheap first stage: one Fourier peak per star, which skips the full fits when
the pulsators clearly cannot be a majority. Its output was byte-identical on
16 detection files across 8 runs, and the fixture came back to 0.64 s. The
guard's bound did not move. Every review brief now starts by reading the
PR's CI status.

## Standing

`main` is `33591cf` with 972 tests. Open: issue 44, the repeat gate,
which stays off until `single_transit` can reach the pulsators whose residual
still fails its guard. Every held-out bar since entry 0011 has been written in
planets with a one-planet allowance.
