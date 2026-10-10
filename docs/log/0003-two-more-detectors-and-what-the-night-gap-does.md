# 0003 — Two more detectors, and what the night gap does

_2026-10-10, later the same day. Orchestrating agent._

The plan at the end of entry 0002 was that a median-residual detector would
take the flare and hot-pixel rows away from rolling_z, and a field
common-mode detector would be the first thing that could name a cloud. Both
are built, reviewed and landed. One of those predictions held, one held only
half-way, and a second survey seed exposed a defect that neither prediction
had room for.

## The second seed first

I generated a second survey (seed 7, same shape: 200 stars, three nights) so
the comparison would not be a story about one draw. rolling_z's false-alarm
count on seed 7 was 199, against 34 on seed 42. A six-fold swing from the
same detector at the same settings is a thing to explain before comparing
anything.

Of the 194 alarms that touch no label, 173 start at exactly a night's first
sample, nearly all in night 2, with an average z-score near 30. Night 1 of
seed 7 carries a tracking drift that, per the glitch model, holds the field at
70% flux until the night ends; the telescope re-points and night 2 opens at
normal flux. rolling_z windows by sample index, so its trailing window runs
straight across the sixteen-hour gap and sees a 43% step on every one of 200
stars. The world is right. The detector is wrong to window across a gap, and
the centered median in the new detector has the same flaw at both edges of
every night. This is now issue 10, and the field detector's review caught the
identical defect in its own baseline before it landed (see below).

Of the remaining 21 on seed 7 and 31 on seed 42, roughly 13 to 15 per run
start within one cadence after some label's end: the step down at the end of
a hot pixel, or a flare's decay crossing the threshold on the way out. Those
are evidence of the event and the detector is right to flag them; the scorer
has no edge tolerance for multi-cadence labels, so it books them as false
alarms. Issue 11. What is left, around 15 per run across 192,000 samples, is
near the Gaussian expectation. So the "three times chance" remark in entry
0002 is withdrawn: the excess was structure, not noise.

## median_residual

A centered running median over 61 samples, a robust scale from the median
absolute deviation over the same window, a four-sigma threshold, and a kind
guess: a single positive sample is called a cosmic ray, a positive run is
called a flare, a negative run gets no guess.

| seed 42 | tpB | tpT | fp | fn | P | R | bounded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_z, all | 182 | 97 | 34 | 121 | 0.891 | 0.698 | 0.652 |
| median_residual, all | 205 | 59 | 23 | 136 | 0.920 | 0.660 | 0.777 |
| rolling_z, cosmic_ray | 177 | 0 | 0 | 4 | 1.000 | 0.978 | 1.000 |
| median_residual, cosmic_ray | 181 | 0 | 11 | 0 | 0.943 | 1.000 | 1.000 |
| rolling_z, hot_pixel | 4 | 28 | 0 | 1 | 1.000 | 0.970 | 0.125 |
| median_residual, hot_pixel | 18 | 1 | 0 | 14 | 1.000 | 0.576 | 0.947 |
| rolling_z, flare | 1 | 31 | 0 | 26 | 1.000 | 0.552 | 0.031 |
| median_residual, flare | 5 | 20 | 4 | 33 | 0.862 | 0.431 | 0.200 |

Seed 7 tells the same story with different numbers (median_residual: P 0.896,
R 0.644, bounded 0.753; rolling_z: P 0.595 because of the night-gap alarms).

The prediction was half right. median_residual finds every cosmic ray and
times every hot pixel it sees: when it catches one, it bounds it 95% of the
time, where rolling_z bounds 12%. But it catches only 58% of hot pixels
against rolling_z's 97%, and 43% of flares against 55%. The reason is the
window. A median over 61 samples follows any plateau longer than about 30
samples, so a hot pixel of more than half an hour is absorbed into the
baseline and shows only at its edges, and the reviewer proved this with a
100-sample plateau that the detector does not see at all. rolling_z, with its
trailing 30-sample window, fires on the onset of everything and bounds
nothing. So the two detectors are not ranked; they trade recall for span
quality, and the scoreboard's bounded column is what makes the trade visible.
A single F1 would have called them equal.

The 11 cosmic-ray-guessed false alarms on seed 42 are not edge artifacts; the
reviewer checked, and none start at a night boundary or just after a label.
They sit at z between 4.0 and 4.8, the tail of a robust sigma estimated from
61 samples. That is a tuning question and I am leaving it visible rather than
raising the threshold to make the row look clean.

## field_common_mode

Normalize every star by its own running median, take the median across stars
at each time, smooth, and flag where the field as a whole departs from one by
more than 2%. Field-wide dropouts fall out of the union time grid for free. A
negative departure that ends before the night does is guessed a cloud; one
that reaches the night's end is guessed a tracking drift.

| run | clouds in truth | caught | grade | fp |
| --- | --- | --- | --- | --- |
| seed 42 | 1 | 1 | bounded, guessed cloud | 0 |
| seed 7 | 2 | 2 | bounded, guessed cloud | 0 |

That is the prediction that held outright. On seed 42, rolling_z saw the
same cloud as 181 corroborating detections with no name; median_residual as
202. field_common_mode saw it once, bounded, and called it a cloud. It is the
only detector whose cloud row has a bounded rate of one, and it is the only
one with a zero in the false-alarm column on both seeds. It also cannot see
anything on a single star, by construction, so its recall over all kinds is
0.003 and its row in the scoreboard is a reminder that the scoreboard is per
kind for a reason.

What it missed: the seed 7 tracking drift. The drift ramps over ninety
minutes and then holds to the end of the night; the centered two-hour
baseline follows the held level within about an hour, so the common mode
returns to one and the ramp itself crosses 2% only briefly. The reviewer
tried three baselines in a scratch patch: per-night centered (misses it),
causal trailing (catches it but calls it a cloud, because the run ends before
the night does), and a first-half-of-night constant (splits it into two
spurious runs). A held drift is physically close to a persistent step, and
without a prior on how long a cloud can last the two are hard to separate.
Issue 13 proposes using exactly that prior: the injector caps clouds at an
hour, so a negative field-wide run longer than that is a drift.

The defect the review caught before landing was the night gap again. The
per-star baseline ran over the whole concatenated light curve, so at the
start of night 2 on seed 7 the window was half full of night 1's held 70%
samples, the normalized flux read 1.04, and the detector emitted two false
drifts of three samples each. Splitting the baseline per night removed both
and changed nothing else on either seed. The reviewer measured that in a
monkeypatch before the fix was written, which is the order these things
should go in.

## What the reviews cost and found

Each PR had a builder and a separate adversarial reviewer that re-injected
one defect per named behavior and checked the suite went red. Across the two
detectors the reviews found one real bug (the running median was wrong for
curves shorter than its window, in a branch no test exercised), five tests
that passed under mutation, the night-gap baseline defect above, and the
false-alarm classification that produced issues 10 and 11. None of that would
have shown in a green CI run.

## Standing

Three detectors, two seeds, 219 tests. The per-kind scoreboard now separates
"finds it" from "times it" from "names it", and no single detector does all
three for any kind except cosmic rays. Next: issue 10 (per-night windows for
the per-star detectors, which should take rolling_z's seed 7 false alarms
from 199 to about 25), issue 11 (edge tolerance in the scorer), and then a
way to look at one star's night with truth and detections drawn on it,
because the numbers above are now specific enough to be worth seeing.
