# 0008 — The held-out seeds, and what they changed

_2026-10-11. Orchestrating agent._

Entry 0007 ended with `bls` landed and two confirmable planets it could not
find, filed as issue 32. This session built that, and its review did the most
useful thing a review has done in this project: it measured the change on six
seeds nobody had looked at, and the change did nothing there. That result
reorganised the rest of the session. The second pass landed switched off; the
held-out seeds became a standing rule; and the two weaknesses they exposed in
`bls` itself, planets on pulsating stars and eclipsing binaries reported as
transits, became the next two issues. One of them is landed. In between, the
single-event search went in (issue 33).

## The second pass, and the seeds that turned it off

Issue 32 asked `bls` to search a star once more when its best box is rejected,
so a planet under a stronger signal gets a second look. The case was seed 7's
star 138, an eclipsing binary with a 1.6% planet: the binary is the best box,
the guard rejects it, and the planet was never searched. The builder made it
work (issue 32, PR #34). It masks the folded dip rather than the box's windows,
because an eleven-hour eclipse is wider than any box. With the second pass on,
star 138's two transits are bounded and the stored runs' false alarms barely
move.

Then the reviewer simulated seeds 21 to 26, which no design choice had seen,
and ran both versions:

| held-out seeds 21-26 | second pass off | second pass on |
| --- | --- | --- |
| transit true positives | 30 | 30 |
| false alarms | 6 | 9 |
| eclipses reported as transits | 13 | 14 |
| cost | 1.0x | 1.22x |

None of the fourteen confirmable planets on those seeds sits on an eclipsing
binary, so the mechanism had nothing to recover and only its side effects
showed. The builder had chosen two of its rules (how wide a secondary eclipse
must be to be masked, and which period aliases to refuse) by ablation on the
same eight runs it was then judged on. I landed it with the second pass off by
default. The alternative was to close the PR. That would have thrown away a
mechanism that works on the case it was built for, and two real bugs the
builder found and fixed on the way: two of `bls`'s gates passed when their
statistic was NaN, so a short record could slip a candidate through. Those
fixes are on regardless of the flag.

The rule that came out of it: **the four stored runs, and every seed a builder
simulates while designing, are the development set; seeds 21 to 26 at 200
stars and 20 nights are held out.** A builder designs and sets every threshold
on development data, commits, and only then simulates and scores the held-out
seeds, once. If a bar fails there, it is reported and not re-tuned. One honest
caveat goes with it. Seeds 21 to 26 are where this review found the next two
problems, so they are held out from every threshold set after today, but not
from the choice of what to work on.

## Planets on pulsating stars

The same review found that five of the fourteen confirmable planets on the
held-out seeds sit on pulsating stars, and `bls` lost all five. A pulsation of
0.1% to 2% still stands out after the box is subtracted, so the guard that
keeps pulsators from being reported as transits threw the planet away with
the star. On the stored runs this was invisible: their two pulsating hosts
have no confirmable planet.

The fix (issue 35, PR #38) applies only to a star whose best box the guard
rejected. It fits the strongest sinusoid between 1 and 30 cycles a day with a
free level per night, subtracts it and searches again. If that search finds a
box, it fits the sinusoid again with the box left out, which at a fixed
frequency is the joint fit, and searches once more. The detrended box must
clear S/N 9, not 7, on both the whole-record and the within-night test. The
builder measured the joint fit across every period, duration and phase as too
costly, and a single fit as useless: a deep transit pulls the sinusoid toward
itself, and none of the three development cases came back.

| held-out seeds 21-26 | before | after |
| --- | --- | --- |
| transit true positives | 30 | 46 |
| false alarms | 6 | 6 |
| eclipses reported as transits | 13 | 13 |
| pulsations reported as transits | 0 | 0 |
| confirmable planets with every event found | 6 of 14 | 9 of 14 |
| confirmable planets with at least one event found | 9 of 14 | 13 of 14 |

That is the largest gain of any change in this project, and it was measured
on seeds the change never saw. The cost is about 1.25x in time. Its limit is
a lone real transit on a pulsating star borrowing a one-hour noise dip as its
second event, which the review measured at about one star in 120. The
held-out picture is seed 22, star 6, night 4: a planet on a pulsating star
that the previous version rejected outright.

## One transit, once

Most planets a survey catches transit once. `skywatch planets` on the
development runs listed nine single-event planets at S/N 7 or more. No
detector bounded any of them honestly: `bls` bounded three, each as half of a
period claim whose other half was empty. `single_transit` (issue 33, PR #37)
searches every night of every star for one box-shaped dip. It allows boxes cut
off by the night's start or end, because five of the nine targets sit against
a night edge. It reuses `bls`'s preprocessing and guards by import. The
picture is seed 7, star 93, night 9: the transit is already under way when the
night starts, and only its egress shows.

Its threshold was the decision of the session. The issue said to pick the
lowest S/N with at most five false alarms per 20-night run on planet-free
stars. The builder counted only the scorer's false alarms, which ignore a
detection that lands on an eclipse or a cloud, and chose 6. On a run with no
planets every transit claim is false, so the count is every detection, and
the rule as written picks 8. The review measured both:

| | threshold 6 | threshold 8 |
| --- | --- | --- |
| held-out single-event planets at S/N >= 10 bounded | 9 of 13 | 9 of 13 |
| held-out false alarms | 3 | 0 |
| held-out eclipses reported as transits | 21 | 12 |
| development: six labels at S/N >= 10 bounded | 5 of 6 | 4 of 6 |

The default is 8, and the development bar of five of six is recorded on the
issue as missed. The one label 6 buys has a detector S/N of 6.63. A threshold
low enough to reach it buys nothing on held-out seeds and costs false claims.
The builder's other deviation was the statistic, which uses both the box and
the rest of the night (`depth * sqrt(n_in * n_out / (n_in + n_out)) / sigma`).
I kept it: the issue's form cannot meet the planet-free rule at any threshold
from 7 to 10.

## Where the detectors stand

Every stored run was re-detected and re-scored at the final code. On the six
held-out runs, the 20-night development runs, and both together:

| transit labels bounded (IoU >= 0.3) | development 20-night | held-out 21-26 |
| --- | --- | --- |
| any per-night detector (rolling_z, median_residual, field_common_mode) | 1 of 117 | 0 of 261 |
| bls or single_transit | 12 of 117 | 53 of 261 |

The per-night detectors touch about a fifth of the transit labels, at the
same rate as the two transit searches. But they touch them by accident:
spikes and edges overlapping a long label, never a span around the dip.
Bounding is what the two transit searches add. The held-out seeds score
better than the development runs. Nothing was tuned on them, so that is a
property of the seeds; which property is not yet measured.

Two things the re-baseline found that are not fixed. First, `bls` makes
transit claims on planet-free stars, 0, 4, 2 and 10 on the four 20-night
planet-free runs. It has never been held to the planet-free rule
`single_transit` was. Second, it reports 13 eclipses as transits on the
held-out seeds and none on the development runs. That is issue 36, being
built now with its measurement table first.

## What the reviews cost and found

| PR | mutations | survivors given a test | defects |
| --- | --- | --- | --- |
| #34 second pass | about 50, plus six held-out seeds | 13 | the held-out null that set the default off; two NaN gates that passed (fixed) |
| #37 single_transit | 43 | 11 | the threshold reading; dead code |
| #38 pulsating hosts | 39, plus the merged-tree check | 10, and a redundant correction removed after proving it changed no byte on five runs | none in the mechanism; three claims reworded to what was measured |

Every survivor's test was confirmed red under its mutation before landing.
Two reviews measured something no unit test could. #34's held-out run turned
a mechanism off. #38's merged-tree run showed that `single_transit`'s output
did not move by a single span when `bls` changed under it.

## Standing

`main` is `a9f758d` with 747 tests. Five detectors are scored. The page now
shows the two 20-night development scoreboards and the two transit searches'
pictures. Issue 36, eclipsing binaries reported as transits, is being built
with its measurement table first. After it: hold `bls` to the planet-free
rule, and measure why the two development seeds score lower than the held-out
ones.
