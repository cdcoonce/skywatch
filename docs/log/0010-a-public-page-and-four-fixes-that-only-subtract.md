# 0010 — A public page, and four fixes that only subtract

_2026-10-13. Orchestrating agent._

The page has a public address now. The owner gave the go on 2026-10-12, and
`docs/` is served by GitHub Pages at
[cdcoonce.github.io/skywatch](https://cdcoonce.github.io/skywatch/). It is the
same generated file it has been since entry 0007: no script, no server, and
the commands to rebuild it printed at the bottom.

Behind it, four more changes landed. Each one, measured on runs its design
never saw, took away claims the detectors should not have been making, and
none took away a planet. Two of them taught something about measuring
itself: one could not show its gain on unseen data, and one shipped
switched off.

## What landed

| PR | issue | what it does | on unseen data |
| --- | --- | --- | --- |
| #46 | 45 | `single_transit` refuses a V-shaped event, and a star with one V night is not reported at all | held-out eclipses reported as transits 11 -> 6; never-seen 9 -> 1 and planet-free 6 -> 0; no planet lost |
| #47 | 43 | `bls` leaves flare-sized clipped runs, and the 3 h after each, out of a box's contrast | held-out flare claims 3 -> 0, false alarms 7 -> 3; never-seen false alarms 5 -> 1; every planet-free run within the rule |
| #49 | 41 | `bls` reads the shape test with a depth per night as well as one depth, and adds a chi-square bar | identical to the old code on 42 never-seen runs; the gain is on development binaries only |
| #50 | 44 | `bls` can refuse a detrended box that does not repeat beyond its strongest night | ships **off**: it would lose real transits that nothing else finds |

## The measurement kept overturning the issue

Three of the four issues named a cause, and in three of the four the
builder's measurement, made before any code, found a different one.

Issue 43 assumed flare stars made false claims because a box sat on a
flare's decay. It does not. The box sits on a quiet stretch. It looks like a
dip because the unclipped tails of flares elsewhere in the night lift the
level the box is measured against. So the fix removes those tails from the
reference, not from the box. Its review then found that 60% of the runs the
gate treats as flares are hot pixels and cosmic rays, so the gate charges
marginal boxes for lost reference samples rather than contamination. That is
issue 48.

Issue 41 assumed the leaking binaries had eclipses wider than the night. One
of nine did. The rest had a box landing on alternate primary and secondary
eclipses, and a single shared depth fits a box to both. Their V was also
strong but read as weak, because the old statistic divides by the S/N.

Issue 44 offered three remedies for fast pulsators: fit a harmonic, set a
minimum box length, or re-run the sine test on the residual. The builder
probed 5,254 pulsators and found that the subtraction is clean at every
frequency from 2 to 25 cycles a day. The three false boxes each rest on one
short night beside a flare. None of the three remedies would have touched
them.

Counting entry 0007's transit survey, measurement has now corrected an
issue's premise four times in three days. Writing the measurement table into
an issue before its PR is now a habit of the repo, not a courtesy.

## A gain unseen data could not show

The shape fix for wide binaries (#49) cut development eclipses reported as
transits from 41 to 10 and lost no planet. Its held-out seeds had no binary
reach the shape gate at all. The review then simulated 42 more runs,
including 30 three-night runs where binaries are common. The new code was
byte-identical to the old on all 42, and on one of them a binary still
leaked just under the new bar. The leaking binaries are rare, so unseen data
mostly cannot see the fix. It also costs nothing anywhere: no planet, no
false alarm, no time. That is the difference from entry 0008's second pass,
which shipped off because it cost false alarms. This one shipped on. The
reviewer noted that a lower bar would have caught the unseen leak. I left the
bar alone, because setting it from the seeds that had just judged it would
have made them design data after the fact.

## A gate that ships off

The fast-pulsator fix (#50) removes pulsations reported as transits: 10 to 0
on development and 1 to 0 on held-out. What the gate refuses is a real single
transit that `bls` had paired with a noise dip into a wrong period. A single
real transit is `single_transit`'s job, so the review asked whether
`single_transit` already finds those events. It finds none of the 18. Every
host is a pulsating star, and `single_transit` does not search stars the
pulsation guard refuses. Under the union of the two detectors the gate
removes 28 false claims for 18 real transits lost, and the review showed that
the gap it was set in did not hold on never-seen seeds either. So it ships
off by default and issue 44 stays open. Issue 51 is the fix that makes the
gate safe: detrend the pulsator and let `single_transit` search the residual.
A prototype recovered 17 of the 18.

## The ledger did its job

Every held-out set in this entry was reserved in `docs/seeds.md` when its
issue was filed, and every review's never-seen seeds were recorded when
used. Nothing collided. The ledger runs to seed 440 now.

## Where the detectors stand

Re-detected and re-scored at `main` (`3c463de`). Seeds 21-26 have been
development since the PRs in entry 0009 scored them, so they read as a
benchmark here, not a held-out test:

| seeds 21-26 | entry 0009 | now |
| --- | --- | --- |
| `bls` transit true positives | 46 | 46 |
| `bls` false alarms | 6 | 3 |
| `single_transit` transit true positives | 14 | 14 |
| `single_transit` eclipses reported as transits | 12 | 3 |
| transit labels bounded by `bls` or `single_transit` | 53 of 261 | 53 of 261 |
| transit labels bounded by any per-night detector | 0 of 261 | 0 of 261 |
| claims on planet-free 20-night runs 41-44, `bls` / `single_transit` | 0,4,2,0 / 4,0,1,2 | 0,0,2,0 / 3,0,0,2 |

Every column that counts finds is unchanged, and every column that counts
false claims went down.

## What the reviews cost and found

| PR | mutations | survivors given a red-confirmed test | never-seen seeds | defects |
| --- | --- | --- | --- | --- |
| #46 | 25 | 3, plus the older guard tests re-pinned with the shape test off | 241-246, 251-254 | the shape test was hiding guard regressions in old tests |
| #47 | 34 | 6 | 261-266, 271-274 | the gate's "flare" runs are mostly not flares (issue 48) |
| #49 | 33 | 4 | 320-331, 340-369 | none; the unseen check showed no change |
| #50 | 21 | 3 | 380-395 | the trade loses real transits under the union; ships off |

## Standing

`main` is `3c463de` with 892 tests, and the page is public. Open: issue 48
(a flare run should mean a decaying run) and issue 51 (`single_transit` on
pulsating stars), both building; issue 44 stays open until issue 51 lands and
the repeat gate is switched on in a PR of its own.
