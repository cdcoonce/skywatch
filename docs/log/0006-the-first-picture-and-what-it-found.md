# 0006 — The first picture, and what it found

_2026-10-10. Orchestrating agent._

Every issue filed from the first two days was closed by entry 0005, so this
session built the thing a visitor would actually look at: one star's night,
with the flux on top, the injected truth shaded under it, and one lane per
detector with every span it claimed, coloured by how the scorer graded it
(issue 22, PR #23). Then the picture did what the scoreboard could not, and
found a rule in the scorer that was wrong (issue 24, PR #25).

## The view

`skywatch view --run <id> --star <s> --night <n>` renders one PNG or SVG;
`--pick` chooses the star-night with the most distinct truth kinds, ties broken
by detection count, then lowest star, then lowest night. The data assembly is a
separate function from the drawing so the night filter, the field-wide label
rule, the grade join and the pick order are unit-tested without rendering, and
matplotlib is imported only inside the renderer so `detect` and `score` stay
light. The figure is a fixed 1200 by 800 pixels, metadata stripped, so two
renders are byte-identical and the three pictures under `docs/img/` are
reproducible from the stored runs. The headline one is seed 7, star 33, night 1:
a held tracking drift, a cosmic ray, a hot pixel and an end-of-night flare on
one curve, and under it the exact story entry 0004 told in numbers, rolling_z
bounding the cosmic ray, touching the hot pixel's onset and corroborating its
exit, while the field detector alone names the drift.

Things decided while building that the issue left open, each with the road not
taken: the flux axis takes its limits from the 0.5th to 99.5th percentile with
clipped samples drawn as pinned markers annotated with their value, because one
cosmic ray at five times the baseline otherwise squashes a hot-pixel plateau and
a flare into the floor (the first render did exactly that; the reviewer caught
it); samples the instrument flagged are drawn as hollow black rings rather than
a colour, because the first choice collided with the drift kind's orange;
unobservable labels do not count toward `--pick` because a visitor should be
shown events that are visible; a lane for a detector that was run after scoring
says "unscored N" rather than claiming zero false alarms, which the first
version did; and an unmatched detection is drawn as `false alarm (matched no
truth label)`, wording chosen after the reviewer pointed out that the legend's
first draft, "no truth here", was false in the very picture it was committed
with.

## What the picture found

That last point is the finding. In the seed-42 pick (star 24, night 2) the
median_residual lane showed two red false-alarm boxes sitting inside the shaded
span of a flare the detector had already found with a bounded span at the
onset. They are fragments of one event's decay. The scorer booked them as false
alarms because the label was claimed and star-level labels match one-to-one;
since 0004 a fragment that merely abuts a claimed label is `corroborating`, and
a fragment overlapping a claimed field-wide label has been `corroborating`
since scoring v1. The same physical situation, the detector seeing the event
again, was getting three grades depending on label scope and a one-cadence gap.

PR #25 makes an overlapping duplicate on a claimed star-level label
`corroborating` against its best-IoU label, with the one-to-one rule still
deciding which single detection earns the true positive. A claimed field-wide
label keeps precedence, which the builder chose after a pure best-IoU version
re-attributed one existing cloud corroboration to a pulsation; nothing already
graded should move because a new grade appeared. Bounded and touched counts did
not change by one on any detector, and the false-alarm column dropped by exactly
the fragment count: 10 and 9 for median_residual on seeds 42 and 7, 2 and 2 for
rolling_z, 0 for the field detector.

| precision | seed 42 before | after | seed 7 before | after |
| --- | --- | --- | --- | --- |
| rolling_z | 0.948 | 0.955 | 0.975 | 0.982 |
| median_residual | 0.923 | 0.956 | 0.938 | 0.970 |

median_residual's `cosmic_ray` false-alarm row on seed 42 went 11 to 6: five of
what the report had been calling cosmic rays were one-sample fragments of flare
decays. Nobody would have read that off the confusion table.

## What the reviews cost and found

PR #23's reviewer ran 38 mutations, 34 red; the four survivors were two benign
(a visual text flip, a "no observations" caption) and two worth pinning (PNG
metadata suppression, which the in-process determinism test cannot see across a
matplotlib upgrade; and whether unobservable labels count in `--pick`). The
review's real value was the visual probe: the squashed axis, the colour
collision, the false legend, and the unscored lane's false counts were all found
by looking, not by mutating, and each got a test that reads the figure's legend
text, axis limits and lane summaries rather than its pixels. PR #25's reviewer
ran 16 mutations and a 40,000-case fuzz that found the only transition the new
rule ever produces is unmatched to corroborating; the one real survivor was the
`iou > 0` boundary, which a relaxed `>=` would have turned into abutting
detections corroborating claimed one-cadence labels through the wrong pass. It
got its test, and the edge pass's overlap guard, now provably dead, got a
comment saying so rather than a test that cannot fire.

## Standing

`main` is `dc1d791` with 466 tests. The scoreboard now has a
picture behind it, the scorer's grades are consistent across label scope, and
every number in entries 0002 to 0006 can be regenerated from two seeds with
four commands. Nothing is specced ahead. Hosting is the next decision, and it
is Charles's: the pictures are static PNGs, so the cheapest honest option is a
page that shows the three committed views with the scoreboard beside them, and
the alternative is a small server that renders `--pick` for any seed on demand.
After that, `bls` across nights for the shallow transits every detector still
misses.
