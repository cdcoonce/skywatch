# 0005 — The drift, the warm-up, and the labels nobody could see

_2026-10-10. Orchestrating agent._

Entry 0004 left three specced issues open and said the one-star night view would
come after them. This entry closes the three (13, 16 and 5, as PRs #17, #18 and
#19), built in parallel by one worker each because they touch disjoint files,
each reviewed by mutation and probed for spec deviations before landing. `main`
went from `e08cdbe` (273 tests) to `40a867c` (383), then `5af8af4` (392) with
the follow-up. The picture is now the next
thing on the list, with nothing specced ahead of it.

## The held drift

Issue 13 was the field detector's one known blindness: the centred running
median follows a held step within about an hour, so a tracking drift that is
held to the end of the night fades out of the common mode and is missed. The
reviewer of PR #12 had already monkeypatched three baselines and found that a
causal (past-only) median catches the drift but then guesses `cloud`, because
the negative run ends when the step is absorbed rather than at the night end.

The fix is the two halves of that observation: a `baseline="causal"` option,
and a kind rule that a negative common-mode run longer than the injector's
maximum cloud duration is a `tracking_drift` whether or not it reaches the night
end. The builder measured both baselines on both runs and made causal the
default because it met every acceptance clause: on seed 7 the drift is caught as
`[92460, 100980)` with score 11.2 and `kind_guess` tracking_drift (IoU 0.557
against truth, so bounded), both clouds stay bounded, there are still zero false
alarms, and seed 42's scoreboard is unchanged to the digit. The cloud spans
start one cadence later under the causal lag, which the scorer does not notice.

Two things to say plainly. The drift span runs about an hour past the ramp end
because the causal baseline absorbs the step; the scorer books it bounded, but
the span is loose, and a long cloud and a long held drift remain separable only
by a duration prior. And the detector now reads a world constant, the maximum
cloud length, as that prior. That is disclosed, not hidden: the rule is "longer
than any cloud the world can make", and it is only as honest as the constant it
reads. The builder copied the literal because another worker was editing the
injector file at the same time; issue 20 moves it to a shared constant with a
test that the two cannot drift apart.

## The warm-up

Issue 16 was 0004's honest cost: after per-night windows, rolling_z's trailing
window starts empty at every night, so the first thirty samples of nights 2 and
3 were blind, and that cost 7 real events on seed 7 and 5 on seed 42. The fix is
a growing window: the first `warmup` (5) samples of a night stay blind, then the
window is whatever has been seen so far, with the z-score scaled by
`sqrt(i / window)` so a five-sample baseline cannot fire on noise. The builder
measured the alternative, raw z with the shorter window: it recovers the same
events and adds 3 first-window false alarms on seed 42 and 1 on seed 7, plus one
seed-7 "true positive" that was a night-start alarm overlapping a night-long
pulsation, the same fake credit 0004 withdrew. Scaled won on the numbers.

All 7 seed-7 events came back and 3 of 5 on seed 42, with neither `unknown` row
moving (7 and 15). The two seed-42 misses are a cosmic ray at sample 2, blind by
definition, and a faint flare at sample 5 whose raw z is 3.88 against a 4.0
threshold; I did not let anyone tune the threshold to collect it. The start of
night 1 had been blind too, so six more cosmic rays came back as a side effect,
three per seed. median_residual was left alone on measurement: its per-night
median already grows from the edge, the only analogue the builder could build
cost more false alarms than it recovered, and its 1 to 3 edge losses per seed
have a different cause.

## The labels nobody could see

Issue 5 was a lie in the denominator. A flare whose peak lands after a night's
last sample, or a sub-cadence span that falls between samples, gets a truth
label and changes no sample, so it counts as a miss against every detector
equally. The design principle is that truth is written at injection and never
inferred afterwards, so the fix follows it: every injector now records
`footprint_samples`, measured by diffing the scene's flux and mask around the
injection and counting changed samples that were observed on entry, not by span
arithmetic. The scorer excludes zero-footprint labels from false negatives,
never matches them, and prints an `unobs` column so the exclusion is visible. A
detection over one is treated as if the label were absent, so it can still be a
false alarm; the alternative, a free pass, would have hidden real ones.

On the two runs this changed nothing: zero labels on either seed have an empty
footprint, which is a measured absence rather than an untested one. The builder
counted the rate across thirty seeds at the 60-second cadence, 4 of 11,602
labels, and demonstrated the mechanism at a 600-second cadence, where 5 of 400
and 3 of 402 labels are unobservable and every false-negative row drops by
exactly its count with nothing else moving. Re-simulation with the same seed is
byte-identical on observations, which is what let the two stored runs be rebuilt
in place with the new field rather than thrown away. One limit is on record: a
hot pixel overwrites the flux of an earlier cosmic ray or flare and runs before
the dropout pass, so that event keeps its footprint while its signal is erased.
It is then an unavoidable miss, and nothing hides it.

## What the reviews cost and found

| PR | mutations | survived, and what it was |
| --- | --- | --- |
| #17 drift | 23, 20 red | the night-end rule could be deleted unnoticed, because every night-end fixture was also longer than a cloud; the duration rule's units were pinned only at 60 s; a pre-existing wall-clock test that flakes on a loaded machine |
| #18 warm-up | about 40, 3 real holes | a short night under the default warm-up was no longer covered, because the PR had changed the old short-curve test's `warmup` to accommodate the new default; the recorded score of a run starting in the growing region; the threshold equality |
| #19 footprint | 337, 321 caught | a counter that, moved out of the per-detector loop, would double the unobservable count for every detector after the first; one dead term that only an explicit `kinds=` reaches |

The second row is the one worth remembering. When a PR edits an existing test's
parameters so it keeps passing under a new default, the old pin is gone and
nothing says so; the fix is to leave the old test as it was and add the
default-path twin. Every survivor got a test that goes red under its mutation
before landing, except the equivalent mutants, which got comments.

## Combined scoreboard

Before is `main` at `e08cdbe`, after is `40a867c`, all-kinds rows, both runs
re-simulated, detected, scored and reported from scratch.

| seed 42 | tpB | tpT | fp | fn | P | R | bounded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_z before | 178 | 92 | 15 | 130 | 0.947 | 0.675 | 0.659 |
| rolling_z after | 184 | 92 | 15 | 124 | 0.948 | 0.690 | 0.667 |
| median_residual | 205 | 58 | 22 | 137 | 0.923 | 0.657 | 0.779 |
| field_common_mode | 1 | 0 | 0 | 399 | 1.000 | 0.003 | 1.000 |

| seed 7 | tpB | tpT | fp | fn | P | R | bounded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_z before | 166 | 94 | 7 | 142 | 0.974 | 0.647 | 0.638 |
| rolling_z after | 174 | 96 | 7 | 132 | 0.975 | 0.672 | 0.644 |
| median_residual | 196 | 63 | 17 | 143 | 0.938 | 0.644 | 0.757 |
| field_common_mode before | 2 | 0 | 0 | 400 | 1.000 | 0.005 | 1.000 |
| field_common_mode after | 3 | 0 | 0 | 399 | 1.000 | 0.007 | 1.000 |

No detector has a false alarm at a night start on either seed. rolling_z's
hot-pixel row on seed 7 now reads recall 1.000 with 0 bounded of 32: it sees
every one and times none of them, which is the cleanest statement yet of what a
window detector is. Only field_common_mode credits the drift as bounded; the
per-star detectors see it as one touched span and fourteen corroborations.

## Standing

Every issue filed from the first two days is closed; issue 20 landed last as
PR #21, which moved the cloud duration law into one exported constant, pinned
its literal values and one seed's draw sequence so a change to the law is a
visible edit, and pinned the one-sample field dropout (`main` `5af8af4`, 392
tests). The three detectors now stand where their designs put them: rolling_z
finds the most and bounds the least, median_residual bounds what it finds and
misses hot pixels it should catch, and the field detector names every cloud and
the drift with no false alarm and sees nothing else. The denominators are
honest, the night gap is gone from the baseline, and the scorer's edge cases are
pinned.

What comes next is the thing a visitor would actually look at: one star's night
with the truth drawn under the flux and every detector's spans drawn over it,
so the difference between touched and bounded is a picture rather than a column.
Hosting stays undecided until that exists. After it, `bls` across nights for the
shallow transits every detector still misses (recall 0.11 to 0.20).
