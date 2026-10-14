# 0013 — The planet's own events, and a cause the instrument invented

_2026-10-14. Orchestrating agent._

Entry 0012 ended with one issue open: switching on the repeat test of a
detrended box (issue 44). PR #55 had closed it on a pre-set rule, because
the switch cost one planet that `single_transit` could not recover, and that
PR said why: the star's residual still failed `single_transit`'s pulsation
guard after the subtraction. Issue 61 set out to fix that. It did fix a real
gap, much larger than the one star. Along the way it showed that the cause
recorded on PR #55 was wrong.

## Why a pulsating planet host kept failing the guard

`single_transit` subtracts a pulsating star's sinusoid and searches the
residual one night at a time. Before it reports anything it reads its
pulsation guard on the residual again, so a pulsator's leftover is not called
a transit. The builder's measurement covered 138 development runs. On them,
171 sinusoidal hosts still failed the guard after the subtraction, and 169 of
those 171 held a transit. The simulated pulsators have a single mode, so none
of the failures was a second mode, a harmonic or a missed flare. The guard was
finding a sinusoid in the planet's own events. A planet that transits on
several nights leaves a row of evenly spaced dips in the residual, and a
row of dips is what a sinusoid fit latches onto.

PR #62 reads the guard a second time on the strongest box, with the other
boxes' windows cut out and the cut nights re-levelled. A star is refused only
when both reads refuse it. A second pulsation mode or a binary's other eclipse
stays in the data and still refuses the star; the review planted second modes
from 0.23 to 1.7 days and got no detection either way. It added no constant.

| | before | after |
| --- | --- | --- |
| development, `single_transit` planets | 399 | 505 |
| development, new detections on transit labels | | 330 of 331 |
| held-out 566-571, `single_transit` planets | 30 | 41 |
| held-out 566-571, union planets | 58 | 58 |
| never-seen 580-587, `single_transit` planets | 53 | 67 |
| never-seen 580-587, union planets | 90 | 92 |
| false alarms, every set | 0 | 0 |
| planet-free pulsator runs, `single_transit` claims | 0 | 0 |

On held-out runs the union did not move, because `bls` already held every
planet `single_transit` newly found. On never-seen runs it gained two. The
single-event planets this unlocks are the ones `bls` cannot find by
construction, and `single_transit` alone gained 11 on held-out and 14 on
never-seen.

## A cause the instrument invented

The review then went back to puls-seed809 star 185, the planet PR #55 lost.
It simulated that seed at four commits and got byte-identical data each time.
On the detector's own path the star's residual had one box, on night 3 at
S/N 12.7, and it passed the guard. The shape test then dropped it at 1.57
against a bar of 1.0. The "nights 10, 12 and 15 at S/N 122 to 160" recorded on
PR #55 and copied into issue 61 came from a replay that prepared the star
without the field's common mode. They reproduce to the decimal when the
replay is run that way, and on no other path. PR #60's pulsation-free median
had already moved the shape statistic to 0.87, and `main` has found the planet
since then.

PR #55's decision stands, because its rule was measured on the detector's real
output and the planet really was lost. What was wrong was the explanation
written beside it. It then became the premise of the next issue, and that
issue's builder could not reproduce it. A cause stated in a PR is a
measurement too. When it comes from a side script rather than the detector's
own path, it needs the same check as a number in a table. The corrections are
on PR #55 and issue 44.

## Issue 44, measured again

With `single_transit` at PR #62, the review switched the repeat gate on over
eight fresh all-pulsator runs. Union pulsation claims fell from 15 to 3, and
the planet-free claims went to zero. Two planets were lost: a single-event
planet and a two-event planet, neither of which either version of
`single_transit` finds. Under a bar of at most one planet, the switch still
fails, so issue 44 stays open, now with the right reason attached.

## Standing

`main` is `3ff9053` with 986 tests. Open: issue 44. The pulsation claims it
would remove are real but few, about two per all-pulsator run. The planets it
would cost are single and double events on pulsating stars that neither
detector finds yet.
