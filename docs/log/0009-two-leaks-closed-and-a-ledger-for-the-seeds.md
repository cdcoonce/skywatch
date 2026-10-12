# 0009 — Two leaks closed, and a ledger for the seeds

_2026-10-12. Orchestrating agent._

Entry 0008 ended with `bls` reporting eclipsing binaries as transits on the
held-out seeds, and making planet claims on stars that have no planets. This
session closed both, each measured on seeds its design never saw. It also
found that "held out" was being kept in my head, and twice in one day that
was not good enough.

## Eclipses that look like transits

On the six held-out seeds `bls` reported 13 eclipses as transits, and none on
the development runs. The builder of issue 36 measured the development
binaries first. 40 detections came from 8 binaries, and the pulsation guard
was nowhere near rejecting any of them. Five of the eight had every reported
event on the same eclipse. An odd-even depth test, the textbook check,
needs two kinds of event to compare, so it had nothing to work with. What
all eight shared was shape: a one-to-three-hour box laid over a V.

The fix (PR #40) fits both a box and a triangle to the nights holding the
candidate's events, each with a free level and its own centre within half an
hour, and refuses the candidate when the triangle fits better by more than
1.5 in units of the box's own S/N. A transit, flat-bottomed with short ramps,
reads negative. An eclipse reads positive.

| | before | after |
| --- | --- | --- |
| held-out 21-26: eclipses reported as transits | 13 | 0 |
| held-out 21-26: transit true positives | 46 | 46 |
| never-seen 61-66 (the review's check): eclipses reported as transits | 26 | 5 |
| never-seen 61-66: transit true positives | 69 | 69 |

The builder's margins looked thin: real eclipses at 1.7 to 1.8 against the
bar of 1.5, and the highest planet at 1.04. So the reviewer simulated six more
seeds that no one had touched. There the planets read 0.12 at most and every
binary but one read 3.4 or more, so the gap is wider on unseen data than on
the data the bar was chosen from. The one exception is a binary whose
7-hour eclipse nearly fills the 8-hour night. Each night then holds only part
of a V, and part of a V with a free level looks like a box. That is issue 41.

## Planet claims on stars with no planets

`single_transit`'s threshold was set by a rule on planet-free stars: on a run
with no planets every detection is a false claim, so allow at most five per
20-night run. `bls` had never been held to it. On the four planet-free runs
it made 0, 4, 2 and 10 claims, and the measurement for issue 39 found two
sources (PR #42).

**Pulsators.** Seven of the eight planet-free pulsators were accepted at the
first pass. The old box-versus-sine test compared a Fourier power with no
night levels against the box, in units of a scatter that contains the
pulsation, and it read 0.53 to 0.99 on them, under its bar of 1. The new test
gives both the box and the best sinusoid a free level per night and compares
their gains, so the scatter cancels. Its bar, 0.81, is not fitted. One
sinusoid can keep at most 81% of a square wave's variance, so a box that a
sinusoid fits that well is not a box. Pulsators reported as transits went
from 81 to 3 on the development runs, from 12 to 0 on held-out seeds 71-76,
and from 14 to 2 on the review's extra never-seen seeds. No transit was lost
anywhere. The two left are fast pulsators, 15 to 20 cycles a day, that pass
through the pulsating-host detrend from entry 0008. That is issue 44.

**Flare stars.** These make claims at 41 of 763 star-runs, against 2 of 4,986
for constant stars. The builder required a local S/N of 7.2 instead of 7.0 on
stars with at least 3% of their samples clipped as flares. That is the
lowest bar on a 0.1 grid that brings every development planet-free run under
five. I kept it, because the issue set the rule before the build. The review
showed what the bar is: it sits inside the spread of false claims, not in a
gap. Eleven claims between 7.0 and 7.19 are refused and seven between 7.2
and 7.4 survive. One development planet-free run passes because one star
reads 7.195. On the held-out planet-free seeds it changed nothing, and seed 82
still makes six claims, one over the rule. A threshold cannot separate flare
residue from a small planet, because the two overlap in S/N. Issue 43 looks for
a property that differs instead: a flare's residue follows a brightening, and
a transit is symmetric.

## The seeds needed a ledger

Twice this session a held-out set was spent before its issue was built. The
review of PR #42 simulated seeds 111-114 to find more pulsators, and I had
named 111-116 as issue 43's held-out set a few hours earlier. PR #42 then
scored 71-76 as its held-out set, which issue 41 had already claimed. Neither
was anyone's fault in the moment. Each agent picked unused-looking numbers,
and the only record of what was used was in issue texts and my context.
`docs/seeds.md` is now the record: every seed range, development or held-out,
and who used it. A held-out seed becomes development once it is scored. An
issue reserves its held-out range in the ledger when it is filed. The two
collided issues have moved to unused ranges.

## Where the detectors stand

Re-detected and re-scored at `main` after both merges:

| held-out 21-26 | entry 0008 | now |
| --- | --- | --- |
| `bls` transit true positives | 46 | 46 |
| `bls` false alarms | 6 | 6 |
| `bls` eclipses reported as transits | 13 | 0 |
| `bls` claims on planet-free runs 41-44 | 0, 4, 2, 10 | 0, 4, 2, 0 |
| transit labels bounded by any per-night detector | 0 of 261 | 0 of 261 |
| transit labels bounded by `bls` or `single_transit` | 53 of 261 | 53 of 261 |

Nothing that was found was lost; what changed is what `bls` no longer
claims. `single_transit` still reports 12 eclipses as transits on the
held-out seeds, because the shape test lives only in `bls`. That is issue 45,
being built now beside issue 43.

## What the reviews cost and found

| PR | mutations | survivors given a red-confirmed test | defects |
| --- | --- | --- | --- |
| #40 shape gate | 25, plus six never-seen seeds and a second-pass check | 4 | none; the margin was wider unseen than seen |
| #42 planet-free | 49, plus fourteen never-seen and planet-free seeds | 20 tests across 21 mutants | none in behaviour; the flare bar's nature and the planet-side margin reworded to what was measured |

## Standing

`main` is `a1fef82` code with 822 tests. Open, with seeds reserved: issue 41
(eclipses wider than the night), issue 43 (flare stars, building), issue 44
(fast pulsators) and issue 45 (`single_transit`'s eclipses, building).
Hosting is still the owner's decision.
