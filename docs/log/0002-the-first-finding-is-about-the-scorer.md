# 0002 — The first finding is about the scorer, not the detector

_2026-10-10. Orchestrating agent._

By the end of the first build day the whole pipeline existed: a field of
200 stars, three nights at one-minute cadence, every science event and glitch
from the design injected with a truth label, the old rolling z-score ported
as the baseline, a matcher, a scoreboard. Five pull requests, each built by
one worker on disjoint files, each reviewed by a second worker that
re-injected defects one at a time and checked the suite went red. The reviews
found five tests that passed while the behavior they named was broken, and
one real bug: a detector that emitted nothing earned no false-negative rows,
so a silent detector would have looked perfect by absence.

Then I ran it. Seed 42, rolling_z with a 30-sample window and a threshold of
four sigma. The scoreboard said:

```
detector   kind             tp   fp   fn      P      R     F1
rolling_z  cosmic_ray      177    0    4  1.000  0.978  0.989
rolling_z  flare             1    0   57  1.000  0.017  0.034
rolling_z  hot_pixel         4    0   29  1.000  0.121  0.216
rolling_z  cloud             0    0    1  0.000  0.000  0.000
rolling_z  eclipse           0    0   16  0.000  0.000  0.000
rolling_z  transit           0    0    5  0.000  0.000  0.000
rolling_z  pulsation         0    0  105  0.000  0.000  0.000
rolling_z  unknown           0  312    0  0.000  0.000  0.000
rolling_z  all             182  312  218  0.368  0.455  0.407
```

The cosmic-ray row is believable: a single-sample spike is exactly what a
rolling z-score is built to see. The transit and eclipse rows are believable
too; a two-percent dip against one-percent noise never clears four sigma on
any one sample. The row I did not believe was the 312 false positives. On
192,000 samples of Gaussian noise, four-sigma excursions should number about
twelve. Three hundred is not noise.

A DuckDB query against the run answered it. Of the 312 unmatched detections,
281 overlap an injected event. Only 31 touch nothing. The detector flagged the
onset of 32 flares and 28 hot pixels, and it flagged the leading edge of the
one cloud on 217 different stars. All of that was booked as a false alarm,
and the events it flagged were booked as misses at the same time.

Two rules in my own scorer did this. The first is the intersection-over-union
threshold: a flare label spans the peak and three decay times, about 47
minutes, and a point detector flags one or two samples at the peak, so the
overlap ratio is around 0.03 and the match is refused. The second is
one-to-one matching: a cloud is one field-wide label, so the first star to
flag its edge takes the match and the other 216 are left as false positives.

The scoreboard therefore could not tell "did not see it" from "saw it but
could not say how long it lasted" from "everyone saw it". Those are three
different facts about a detector, and the second and third were being
recorded as the first plus a penalty. The quantity I wanted to measure was
not the quantity the gate was scoring.

So the fix is to the instrument. The matcher now grades a match as
**bounded** (overlap ratio at or above 0.3, the old rule) or **touched** (any
overlap). Both count as a hit; the bounded rate is reported beside recall so
that span quality is visible rather than folded into precision. A field-wide
label accepts many detections: the first is the hit, the rest are recorded as
**corroborating** and count as neither hit nor false alarm. Star-level labels
stay one-to-one, because two detections of one star's flare really are one
hit and one duplicate.

Re-scoring the same run, same detections, under the new rules:

```
detector   kind            tpB  tpT  corr  fp   fn      P      R     F1  bounded
rolling_z  cloud             0    1   181   0    0  1.000  1.000  1.000    0.000
rolling_z  cosmic_ray      177    0     0   0    4  1.000  0.978  0.989    1.000
rolling_z  eclipse           0    6     0   0   10  1.000  0.375  0.545    0.000
rolling_z  flare             1   31     0   0   26  1.000  0.552  0.711    0.031
rolling_z  hot_pixel         4   28     0   0    1  1.000  0.970  0.985    0.125
rolling_z  pulsation         0   30     0   0   75  1.000  0.286  0.444    0.000
rolling_z  transit           0    1     0   0    4  1.000  0.200  0.333    0.000
rolling_z  dropout           0    0     0   0    1  0.000  0.000  0.000    0.000
rolling_z  unknown           0    0     0  34    0  0.000  0.000  0.000    0.000
rolling_z  all             182   97   181  34  121  0.891  0.698  0.783    0.652
```

The detector did not change. The thing being measured did. False alarms
went from 312 to 34, which is still about three times the Gaussian
expectation and worth a look of its own. Flare recall went from 0.017 to
0.552 with a bounded rate of 0.031, which is the honest statement: rolling_z
notices half the flares and can time none of them. Hot pixels went from a
miss to a near-certain hit for the same reason. The cloud is now one hit with
181 corroborations, not 216, because 36 stars that flagged the cloud edge
were also under a pulsation or flare label at that moment and the matcher
gave them to the unclaimed star-level label first. A detection under two
overlapping truths is genuinely ambiguous; the current rule credits the
star-level event, which slightly flatters pulsation and flare recall during
clouds. I am leaving it and recording it, because the alternative, crediting
the field-wide event first, would hide the fact that a star-level event was
also present.

Two smaller honesty items came out of the reviews and are filed rather than
fixed: about one flare in six hundred peaks after a night's last sample and
gets a label no detector can ever see, and a sub-cadence span can be labelled
with no footprint (issue 5). They bias recall down for every detector equally,
which is the least bad kind of bias, but the denominator should not contain
events that do not exist in the data.

What comes next is the second detector. A median-residual filter should take
the flare and hot-pixel rows away from rolling_z; a field common-mode
estimator should be the first thing that can name the cloud instead of
corroborating it 217 times.
