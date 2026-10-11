# 0007 — The page, the survey that could not be searched, and the first periodic find

_2026-10-11. Orchestrating agent._

Entry 0006 ended with hosting as the next decision, and that decision is
Charles's. This session built the part of it that is the same under either
answer, a generated page with the pictures beside the scoreboard (issue 26,
PR #27), and then went to the detector every entry since 0005 had promised
next: `bls` across nights for the shallow transits. The promise turned out to
rest on a premise nobody had measured. The runs we score contain no transit a
periodic search could find. So the session first built the instrument that
says which planets are findable (issue 28, PR #29), used it to choose a survey
on which the question can be asked, and then built `bls` and scored it there
(issue 30, PR #31).

## The page

`skywatch site --spec docs/site.json` writes `docs/index.html`: a header, four
sentences on how to read a picture, the headline view, each run's `all` rows
with the per-kind table folded under them, the other views, and the exact
commands that regenerate everything on the page. The spec is data (which runs,
which views, the captions); the generator is code; the PNGs it renders go
through the same `view` path and come out byte-identical to the committed
ones, which the acceptance checked with `git status` after a regeneration.
There is no JavaScript, no external asset, and no timestamp, so two
generations from the same runs are the same bytes.

I chose a generated page over a hand-written one because a hand-written
scoreboard drifts from `report` the first time the scorer changes, and the
scorer has changed in four of six entries. The alternative I did not take was
rendering the page in CI from the seeds: it would prove the committed page is
regenerable, but the PNG bytes depend on the pinned matplotlib and its font
build, and a check that goes red on a font difference is a toll, not a guard.
The regeneration stays a command a person runs.

Hosting itself is still Charles's call, and the measured facts that bear on
it are now on the project note: a fresh 200-star, 3-night seed simulates,
runs all three detectors, scores and renders its pick in about five seconds
and 3.5 MB; the home server is on the LAN only, so a server that renders any
seed is a public process somewhere else; and a dozen seeds pre-rendered
statically would cost about a minute of compute and about a megabyte of PNG,
which is most of the "any seed" feeling with nothing running.

## The survey that could not be searched

The design says `bls` "should be the only thing that finds shallow transits",
and 0005 put a number on what the others miss: transit recall 0.11 to 0.20.
Before speccing `bls` I had the transit population of the two stored runs read
out of the truth and planet tables, with each event's matched-filter S/N
(`depth * sqrt(sum(profile^2)) / sigma`, where `sigma` is the star's own
per-sample noise, floor and photon term together) and each planet's stacked
S/N across its events.

| run | planets | with 2+ events | best recurring stacked S/N | best single event |
| --- | --- | --- | --- | --- |
| seed7-200s-3n | 34 | 1 (star 11) | 0.1 | star 185, one event, 10.1 |
| seed42-200s-3n | 25 | 1 (star 49) | 1.7 | star 58, one event, 2.1 |

Nothing periodic is there to find. The 0.20 recall on seed 42 was two
unrelated spikes overlapping one night-long clipped label, not a found
transit. Across seeds 0 to 99 the default 200-star field holds a planet with
two or more in-night events at stacked S/N 7 or better in 21% of 3-night runs
(0.24 per run), 83% of 10-night runs (1.48) and 99% of 20-night runs (3.24).
The planets a periodic search can confirm at 3 nights have periods of a day
or less, because eight hours of sky per day gives a longer planet one chance
in three of landing in a night at all; lengthening the survey adds longer
periods, not shallower depths. Single events above S/N 7 are plentiful at every
length (1.3 per 3-night run), which is a different detector's job.

I had three ways to go and took the first. Keep the world's planet laws and
survey two more nights-counts of the same seeds: a 20-night run of each seed
is the transit bench, the 3-night runs keep every published number, and a
20-night run costs about 45 seconds to simulate, detect three ways and score,
and 12 MB on disk. The second way was to raise `planet_rate` or shorten the
period range so 3-night runs had something to find; that bends the world
toward the detector, and the design records the planet population as a law
of the world. The third was to build the single-transit box search instead,
since that population exists on every run; the design names `bls` first and
the single-event search is a separate claim with its own false-alarm problem,
so it stays on the list.

The instrument went in first as `skywatch planets --run <dir>`, one row per
planet with its events and stacked S/N, because a target quoted from an
estimate in an issue is a memory, and a command on the stored run is a fact.
On the 20-night runs it names the bench:

| run | star | period d | depth | dur h | events | stacked S/N |
| --- | --- | --- | --- | --- | --- | --- |
| seed7-200s-20n | 138 | 8.012 | 0.01600 | 3.51 | 2 | 40.8 |
| seed7-200s-20n | 185 | 4.240 | 0.01424 | 6.00 | 3 | 19.5 |
| seed7-200s-20n | 82 | 3.191 | 0.00601 | 3.21 | 3 | 16.4 |
| seed7-200s-20n | 163 | 8.733 | 0.00527 | 3.64 | 2 | 9.1 |
| seed42-200s-20n | 119 | 4.254 | 0.00756 | 5.58 | 2 | 19.4 |
| seed42-200s-20n | 164 | 1.283 | 0.00309 | 2.85 | 7 | 5.4 |

The current detectors on those runs touch transits only by accident
(seed 7: median_residual 1 bounded and 10 touched of 70 labels, rolling_z 8
touched; seed 42: 11 touched of 47 each; the field detector none). The S/N the
tool reports counts instrument-flagged samples, so it is an upper bound for a
detector that drops them; the design says so.

## `bls`, and what it found

`bls` folds each star's whole record at trial periods from half a day to half
the baseline, uniform in frequency with a relative step of a third of the
shortest duration over the baseline (8,247 periods at 20 nights), tries box
durations from half an hour to six hours, and reports the best box when its
S/N is 7 or more and its predicted windows cover samples on at least two
nights. It emits one detection per predicted in-night event, so each event
can be graded against its own label, and one candidate per star. The issue
left three mechanisms to the builder and the builder needed two more than the
issue named, each of which the review then switched off one at a time on
seed 7 to see what it carried:

| gate | what it does | with it off (seed7-200s-20n) |
| --- | --- | --- |
| field common-mode division | divide every star by the field's per-time median before folding | 0 detections; every planet lost, because a held drift or cloud wing the quality flag does not cover is the deepest "transit" on every star |
| local S/N | score the box as a within-night contrast as well as a whole-record one | 42 detections, 14 false alarms; the night-median normalisation makes 4 to 6 h boxes in 8 h nights look like S/N 8 to 12 on planet-free stars |
| two nights at 2 sigma | each of two nights must show the dip on its own | no change on this run |
| box versus sine | reject when a sinusoid fits at least as well as the box | 19 pulsations reported as transits |
| residual sinusoid | reject when a sinusoid still stands out after the box is subtracted | 100 eclipses and 3 pulsations reported as transits |

The two halves of the pulsation guard are not redundant: one stops pulsators,
the other stops eclipsing binaries, whose second eclipse is what the box
leaves behind. The false-alarm column cannot see either of them, because a box
that lands on a pulsation or an eclipse is a true positive of that kind in the
scorer; only the confusion rows show the guard working.

What it finds:

| run | transit tp (all bounded) | fp | what they are |
| --- | --- | --- | --- |
| seed7-200s-20n | 7: stars 82 and 185, every event (IoU 0.68 to 0.95), plus one event on star 189 | 2 | star 189's claimed partner event, and a flare star (175) |
| seed42-200s-20n | 4: star 119 both events (IoU 0.87 and 0.86), plus one event each on stars 24 and 83 | 3 | star 24's claimed partner event, and a flare star (156) twice |
| seed7-200s-3n | 0 | 1 | the eclipsing binary star 143: one window bounds an eclipse, the other predicts one where none is labelled |
| seed42-200s-3n | 0 | 0 | the eclipsing binary star 0: both windows bound eclipses |

The extra true positives on stars 189, 24 and 83 deserve a plain
description. Each is a planet with exactly one observed transit (stacked S/N
24.3, 8.5 and 14.8), which a periodic search cannot confirm; `bls` paired the
real dip with a second window that holds nothing (189, 24) or a tracking
drift (83), and reported the pair as a period. The scorer grades per event, so
it credits the real dip and books the phantom as a false alarm or a touched
drift, and says nothing about the period being wrong. That is the right grade
for each span and the wrong grade for the claim; the scorer has no notion of
a planet, and I have left it that way for now rather than invent a periodic
match rule on three cases. The 3-night runs show the other face of the same
thing: each has one eclipsing binary the residual test cannot reject with
three nights, reported as a transit at the binary's period, and the scorer
books those windows as bounded eclipses detected as `transit`, which is
exactly what they are.

Star 185 is the picture: a 1.4% dip over five and a half hours on a star
whose noise is 1.7% per sample, invisible to every per-night detector,
bounded by `bls` on each of its three nights. Star 119 on seed 42 is the same
story at 0.76%. No sinusoidal star is reported as a transit on any run, and on the 20-night runs no eclipse is.

Two confirmable planets are not found, and the acceptance line for one of
them is not met. Star 138 is an eclipsing binary with a 28% primary eclipse
every four days and a 1.6% planet every eight; the binary is the star's best
box at S/N 361, the residual test rejects it, and with one candidate per star
the planet is never searched. The review masked the eclipses on its own side
and the same search returned the planet at 8.016 days and S/N 38.9. Star 163
is a flare star whose best box is noise at S/N 6.0 on the local test; the
real planet at S/N 9.1 is never the best box. I landed the detector with both
losses disclosed and filed the second pass as issue 32 (re-search a star once
after its best box is rejected), rather than hold the PR to recover 138 now:
the issue had scoped a second signal on one star out, the mechanism is a
separate claim with its own acceptance, and the two 20-night runs already
show what the single pass can and cannot do. The alternative, folding the
second pass into this PR, would have landed an untested interaction between
the guard and the re-search in the same change as the guard.

The grid reading is worth a line. The issue wrote the frequency step as
`(min_duration / baseline) / oversample`, which as a step in cycles per day
is not dimensionally a frequency; the builder read it as a relative step, so
periods are log-uniform in frequency, and the review confirmed that is the
standard criterion: the last event's predicted time moves by a third of the
shortest duration at every period, where the literal reading would have
moved it by 96 minutes at 9.6 days.

## What the reviews cost and found

| PR | mutations | red | survivors that got a test | behavioural defects |
| --- | --- | --- | --- | --- |
| #27 site | 46 | 29 | 9 (pick honoured, duplicate files, empty spec, path forms, page-after-pictures, regenerate `--out`, ordering, meta line) | 1: `--out` was validated after the pictures were drawn, so a bad target left PNGs behind and a traceback |
| #29 planets | 41 | 33 | 3 (`--min-snr` on the right field, the summary threshold at exactly 7, samples from the label's own star) | 0; 118 planets recomputed independently to 4e-15 |
| #31 bls | 43 | 33 | 10 (guard band, robust sigma, defaults, a half-hour transit at 9.6 days, the low end of the period range, unsorted durations) | 0; four gate counterfactuals and the star 138 oracle |

Every survivor's test was confirmed red under its mutation before landing.
The #27 review's remaining green was equivalent (a suffix check that a later
check repeats). Three new tests on #29 came from a reviewer who noticed the
fixture had no planet between S/N 5 and 7, which is exactly where the summary
line's threshold lives; a fixture that never crosses a threshold cannot pin
it.

## Standing

`main` is `c2652cc` with 583 tests. Four runs are stored: the two
3-night surveys every number since 0002 comes from, and 20-night surveys of
the same seeds on which `bls` is scored. The page is generated and committed, regenerated after `bls` so every view carries its lane;
publishing it is Charles's decision, with the costs on the project note.
Issue 32 (the second pass) is specced with its targets. The single-transit
box search, the 3-night eclipsing binaries the residual test cannot reject,
and adding the 20-night scoreboard to the page are the next three things
worth doing, in that order.
