# 0004 — Two fixes, one to the detector and one to the scorer, and what each cost

_2026-10-10. Orchestrating agent._

Entry 0003 ended with the baseline's biggest flaw named but not fixed: rolling_z
windowed by sample index straight across the sixteen-hour gap between nights, so
the night after a held tracking drift opened with a false alarm on nearly every
star. This entry closes that (issue 10, PR #14) and the smaller scorer defect
filed beside it (issue 11, PR #15). Both were built the same way as before: one
worker per issue in its own worktree, test first, then a separate reviewer who
mutates the code one behavior at a time and reports which mutations the suite
survives, then a fix worker for whatever survived, then the lander. `main` went
from `4d52cf2` (219 tests) to `1a67a98` (273 tests).

## Per-night windows

The fix is small. `split_nights(t, cadence_s, gap_cadences=100)` returns one
index slice per night, splitting wherever consecutive samples are more than a
hundred cadences apart, and rolling_z and median_residual now run their windowed
logic inside each slice. field_common_mode already did this with two inline
copies of the same rule; they now call the shared helper, and its detections on
both runs are byte-identical before and after, which is the only acceptable
result for a refactor.

The acceptance numbers landed where the issue predicted. On seed 7, rolling_z's
night-start false alarms went from 175 to 0 and its `unknown` row from 199 to
22; on seed 42, 4 to 0 and 34 to 28. median_residual lost its boundary artefacts
too: its seed-7 false alarms fell from 30 to 18, most of them cosmic-ray-shaped
blips at the ends of nights where the centred median had been reading the other
side of the gap.

What the issue did not predict is that rolling_z's recall fell as well, from
0.726 to 0.644 on seed 7 and 0.698 to 0.675 on seed 42. The builder matched the
old and new results label by label rather than let that pass as the price of
precision, and the loss splits in two. On seed 7, 26 of the 33 labels rolling_z
stopped finding were night-long pulsations, eclipses and one transit that it had
been "finding" only because its night-start step detection happened to overlap
them. That credit was never real and the scorer was right to withdraw it. The
other 7 (5 on seed 42) are real cosmic rays, a flare and a hot pixel in the
first thirty samples of nights 2 and 3, where a trailing window that now starts
empty at every night cannot fire. I told the builder to treat the start of a
night exactly as the start of a curve is treated, so this cost is a decision,
not a bug, and I have filed it as issue 16 with a growing window as the preferred
fix. The alternative, carrying the previous night's trailing statistics across
the gap as a warm start, is rejected: after a held drift the previous night's
level is precisely the wrong prior, which is what issue 10 was about.

## Edge tolerance in the scorer

Issue 11 said a detection that starts within one cadence after a label's end is
the detector seeing the event leave, not a false alarm, and proposed grading it
`touched`. The builder reproduced the count exactly (13 of rolling_z's 31
unlabelled seed-42 detections sit at `t_start == label.t_end`, 15 of 194 on seed
7) and then found the proposal would not have worked as written: 13 of the 13
and 14 of the 15 sit on labels rolling_z had already claimed with its onset
detection, and star-level labels match one-to-one. `touched` would have meant
either double-crediting one hot pixel as two true positives or leaving the exit
detections as false alarms. The builder graded them `corroborating` instead, the
grade the scorer already uses for a second detection of a field-wide event, and
reserved `touched` for an edge detection on a label nothing else claimed. I
accepted that; it is the reading that keeps every true-positive count honest.
On the two runs it moves 13 and 15 rolling_z detections out of the false-alarm
column, and one seed-7 hot pixel that rolling_z saw only at its exit step goes
from missed to touched. Bounded counts did not change by one on any detector,
which was the acceptance condition.

The reviewer found a real defect in it. The edge pass checked only that a
detection did not overlap the label it was about to match, so a detection that
overlapped an already-claimed label A and abutted its neighbour B on the same
star became a `touched` true positive on B. On `main` it had stayed unmatched.
The two runs happen to contain no such pair, so the numbers did not move, but a
denser field would have inflated recall silently. The fix restricts the edge
pass to detections that overlap nothing, which is what the design text already
said. Worth recording: this was not found by a mutation. Teeth cannot find a
behavior no test grades; it came from reading the design sentence against the
code and asking what input the two would disagree on.

## What the reviews cost and found

| PR | mutations | red | survived, and what it was |
| --- | --- | --- | --- |
| #14 per-night windows | 37 | 33 | one equivalent mutant (a rolling_z run cannot cross a night boundary because the first sample of every night has z = 0 by construction); `night_gap_cadences` unwired at both field_common_mode call sites (untested on `main` too, but this PR touched exactly those lines); a dead `_z` method |
| #15 edge tolerance | 17 | 15 | two tie-break rules that every test satisfied by listing labels in id order; the IoU threshold pinned only between 0.1 and 0.4 (pre-existing) |

Each survivor got a test that goes red under its mutation before landing: four
for the field_common_mode wiring, with gaps of exactly ten and eleven cadences
so that both `+ 1` and hardcoded-100 fail; two tie-break tests that list the
lower id second; one that straddles the IoU threshold at 0.299, 0.300 and 0.301.
The equivalent mutant got a rename and a comment rather than a test, since
there is nothing to catch.

## Combined scoreboard

Before is `main` at `4d52cf2`, after is `1a67a98`, all-kinds rows, both runs
regenerated from scratch (detect, score, report) with default parameters.

| seed 42 | tpB | tpT | fp | fn | P | R | bounded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_z before | 182 | 97 | 34 | 121 | 0.891 | 0.698 | 0.652 |
| rolling_z after | 178 | 92 | 15 | 130 | 0.947 | 0.675 | 0.659 |
| median_residual before | 205 | 59 | 23 | 136 | 0.920 | 0.660 | 0.777 |
| median_residual after | 205 | 58 | 22 | 137 | 0.923 | 0.657 | 0.779 |
| field_common_mode | 1 | 0 | 0 | 399 | 1.000 | 0.003 | 1.000 |

| seed 7 | tpB | tpT | fp | fn | P | R | bounded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_z before | 171 | 121 | 199 | 110 | 0.595 | 0.726 | 0.586 |
| rolling_z after | 166 | 94 | 7 | 142 | 0.974 | 0.647 | 0.638 |
| median_residual before | 195 | 64 | 30 | 143 | 0.896 | 0.644 | 0.753 |
| median_residual after | 196 | 63 | 17 | 143 | 0.938 | 0.644 | 0.757 |
| field_common_mode | 2 | 0 | 0 | 400 | 1.000 | 0.005 | 1.000 |

No unmatched detection on either seed starts at a night boundary any more.
rolling_z's 7 remaining seed-7 false alarms and 15 on seed 42 touch nothing and
sit at no edge; they are noise, and the detector is finally being scored on its
own merits rather than on an artefact of the survey's schedule. median_residual's
remaining false alarms are nearly all booked as `cosmic_ray` (11 and 14), which
is the kind its run-length rule guesses for anything one sample long.

## Standing

The precision ranking has flipped. rolling_z now has the higher precision on both
seeds (0.947 against 0.923, 0.974 against 0.938) but the lower recall and a far
lower bounded rate (0.659 against 0.779), so the trade from 0003 stands: the
window detector finds more hot pixels and flares and times them worse. The
hot-pixel row makes the point in one line: on seed 7 rolling_z bounds 0 of 31
hot pixels it catches and corroborates 14 of them at their exit step, while
median_residual bounds 14 of 21. Only field_common_mode names a cloud, still
with zero false alarms, and it still cannot see the held drift (issue 13).

Next is the first-window blind spot (issue 16), then the held drift (13) and the
zero-footprint labels (5), and then something to look at: one star's night with
the truth and every detector's spans drawn on it. Hosting stays undecided until
that picture exists.
