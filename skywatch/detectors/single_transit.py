"""One-event box search per night: the single-transit detector.

Most planets a survey catches transit once, and a transit at the edge of a night has flux on
one side only, so neither a periodic search (``bls`` needs two events) nor a detector that
compares a dip with its surroundings on both sides can see it. This detector searches each
night of each star on its own for the one box-shaped dip that best explains it and reports it
when its S/N reaches ``snr_threshold`` (``kind_guess="transit"``, ``score`` = the S/N), unless
the dip is V-shaped, which is an eclipse (see Shape test).

Preprocessing
-------------
The same as :class:`skywatch.detectors.bls.BlsDetector`, by import: samples with ``quality``
bit 0 set are dropped, each night is divided by its median, the field's per-time median is
divided out (a tracking drift or cloud wing the flag does not cover is a "dip" on every star)
and each night is renormalised, then positive outliers are masked one-sidedly with their flare
tails. The reported span runs from the first to the last prepared sample the box covers (a
masked flare sample inside the box is inside the span, since the box is contiguous).

Search
------
Per night, every box of duration ``durations_h`` (0.5 to 6 h) at every start sample, and at
every start before the night's first sample or after its last one that still holds
``min_in`` (15) samples, so a box truncated by the night's start or end is a trial like any
other. A box is a candidate when it holds at least ``min_in`` samples and the night has at
least ``min_out`` (30) samples outside it; the reference level is the mean of those
out-of-box samples of the same night (no night median is involved). With ``n_in`` and
``n_out`` samples and ``depth = mean_out - mean_in`` (only ``depth > 0``, a dip, counts), boxes
are ranked by ``depth * sqrt(n_in * n_out / (n_in + n_out))``, which needs no scale, and the
winner's ``sigma`` is ``1.4826 * MAD`` of the same night's normalised flux *outside* the box (a
dip that fills half the night would otherwise inflate it: seed7-200s-20n star 93 scored 11
instead of 17.7). The S/N is the matched-filter S/N of the contrast, ``depth * sqrt(n_in * n_out
/ (n_in + n_out)) / sigma``. The issue's ``depth * sqrt(n_in) / sigma`` assumes the reference
level is known; it is the ``contrast=False`` option and agrees with the default for a box that
is small against its night. It is not the default because the level is estimated from the
same night: on the four planet-free 20-night runs it gives 14, 32, 40 and 39 detections that
overlap no truth label (the scorer's fp) at S/N 7 and 0, 6, 1 and 5 at 10, and 102, 98, 118
and 112 detections in all at 7 and 16, 23, 11 and 20 in all at 10, where the contrast form
gives no such fp at either and 8, 2, 1 and 2 detections in all at 7 (at most 5 per run at 8
and 9). So the issue's form cannot meet the planet-free rule below at any threshold from 7 to
10; this is the one place the statistic departs from the issue. The best box per night is
reported when its S/N reaches the threshold; its :class:`Detection` spans the covered
samples of the box (half-open: ``t_end = t_last + cadence_s``).

Threshold
---------
``snr_threshold = 8``, by the rule the issue gives: the lowest level with at most 5 false alarms
on each of four planet-free 200-star 20-night runs (``planet_rate=0``, seeds 41 to 44). On a
planet-free run every ``single_transit`` detection is a false claim of a transit, so a false
alarm is any detection, not only one that overlaps no truth label (the scorer's fp, which is 0
on every one of these runs from 6 to 10 and 3, 3, 1 and 6 at 5). Every detection on those runs:
10, 2, 1 and 3 at 6; 8, 2, 1 and 2 at 7; 4, 0, 1 and 2 at 8; 3, 0, 1 and 2 at 9. At 7 seed 41
alone fails (8), and what it holds is the guards' residue (3 eclipses bounded, 1 eclipse
touched, 2 clouds, a flare, a pulsation), not noise. Reading the rule as the scorer's fp alone
would have chosen 6; that is not what the issue says.

What 8 costs (measured; 6 gave more recall and more confusion):

* Development (the stored runs seed7 and seed42, 3 and 20 nights): of the six single-event
  labels at S/N 10 or more, 4 are bounded; the issue asks for 5. That bar is **not met**. The
  misses are star 99 of seed7-200s-20n, a planet on an eclipsing binary against an eclipse the
  mask removes (see Limits), and star 196 of seed42-200s-20n, whose detector S/N is 6.63
  against a true stacked 14.3: it is bounded only at 6 (through guard (b)'s mask, and only
  between 6 and 6.63) and lost at 7 to 9. 5 of 6 held at 6 only. fp 0 on all four runs;
  eclipse->transit 2 (seed7, 3 and 20 nights) and 0 (seed42); pulsation->transit 1 (seed42, 3
  nights; the issue allows 5).
* Held-out seeds 21 to 26 (20 nights, never used to choose anything): of the 13 single-event
  labels at S/N 10 or more, 9 are bounded at every threshold from 6 to 9 (69 percent, against
  67 percent on the development set at 8); fp 0; eclipse->transit 0, 0, 3, 2, 0 and 7 per run
  (12 in all; at 6 it is 0, 1, 4, 2, 3 and 11); pulsation->transit 0.
* The 0.5 h mask padding (``mask_pad_h``) was tuned on the development set.

Guards
------
Both reuse the ``bls`` pulsation guard unchanged and run only for a star that has a candidate
box, because the second needs a full periodic search.

* The star's strongest box is put through :meth:`BlsDetector._pulsating` as a one-event box
  (a "period" longer than the survey, so one window): box versus sine, and a sinusoid left in
  the residual once the box is subtracted. A transit leaves white noise; a pulsator leaves its
  pulsation and an eclipsing binary its other eclipses. The night of that box is first divided
  by the median of its out-of-box samples, because the night median includes the dip and lifts
  the whole night, a step the residual test would read as a sinusoid (without it a clean 1.5
  percent dip in white noise was rejected). Measured on eight 20-night runs, the residual's
  power over the false-alarm level of ``bls`` (``guard_fap``, 1e-3) is 0.55 to 1.4 for the
  stars with a planet and no other variability, up to 2.9 for flare stars, at least 4.5 for
  eclipsing binaries and at least 7.8 for sinusoidal stars.
* A star the guard rejects is searched with ``bls`` (:meth:`BlsDetector._search`). When that
  periodic search's best box is rejected by the pulsation guard, ``bls``'s dip mask
  (``_dip_mask``: the dip round the box, a binary's secondary included) gives the windows of
  the eclipses; they are narrowed from a box duration to ``mask_pad_h`` (0.5 h, the shortest
  box) each side, the samples inside are left out, each night is divided by the median of what
  is left, and the star is searched again; its strongest box must pass the guard again. An
  eclipsing binary passes on that second look; a pulsator does not, and a star whose periodic
  search found no mask is not searched.

``guards=False`` switches both off.

Shape test
----------
A box fit to a V-shaped eclipse is a transit claim on a binary: with the guards in place
``single_transit`` still reported 12 eclipses as transits on the held-out seeds 21-26 (seed26
alone 7) and, on the planet-free 3-night runs 41-43, 5 of its 7 detections were on eclipses.
Every box that survives the guards is therefore put through the shape test of ``bls`` (#36,
:meth:`BlsDetector._shape_excess`) read as a one-event box: the box's night alone, the box's
centre and covered span as the event, ``sigma`` the night's scatter outside the box (``bls``
passes a star-wide one). The dip ``1 - flux`` is fitted with boxes (15 widths, 0.5 to 6 h) and
triangles (``max(0, 1 - 2|x| / w)``, 23 widths, 0.5 to 24 h), each with a free depth and a free
level and a centre within 0.5 h of the box; with ``g_box`` and ``g_triangle`` the largest
chi-square gains the statistic is ``(g_triangle - g_box) / (sigma * sqrt(g_box))``: negative for
a flat bottom, positive for a V, zero when the night holds fewer than 10 samples. The rule:

* A star with any reported box whose statistic reaches ``shape_threshold`` (1.0; ``None``
  switches the test off) is not reported at all. A V on one night makes the star an eclipsing
  binary, and its other nights are eclipse residue: the wing of an eclipse at a night's edge
  is a short ramp that no single-night test can tell from a transit's step (below).
* The test runs last, after both guards, so a pulsator is still dropped by the guard; it adds
  one ``_shape_excess`` call per reported box (about 10 ms) and no measurable CPU time to a
  20-night run (seed26: 35.9 s at ``main``, 36.2 s here).

``scripts/single_transit_shape_audit.py`` prints the statistic of every box a run would report.

Threshold, and what it was set on. 1.0 is a margin above the planets, not a fit. The pool is
every box the detector reports (without the test) on 80 development runs (the stored runs, the
seeds of docs/seeds.md that earlier changes designed on, and #45's design seeds 210-219: 200
stars, 20 nights, plus the 3-night runs of seeds 7, 42 and the planet-free 41-43): 142 boxes on
a planet's transit (90 at S/N 10 or more), 119 on an eclipse (115 distinct: a planet and a
planet-free run of one seed share their binaries), 2 on a pulsation, 33 on other labels, 6 on
nothing. The planets' statistic has median -0.64 and a top of 0.73, 0.72, 0.66, 0.60 and 0.59;
the eclipses' has 10th, 25th, 50th, 75th and 90th percentiles of 0.12, 0.53, 1.29, 2.16 and
3.47. The test on each night alone at 1.0 removes 73 of the 119 eclipse boxes and no planet; the
star rule removes 89 and one planet (seed212-200s-20n star 96, S/N 8.3, a planet on an eclipsing
binary whose eclipse night reads 2.19). 0.8 would remove 91 and leave the planets 0.07 of margin;
1.5 (the ``bls`` value, which sees several nights) only 62. The statistic is ``bls``'s function
on one night instead of several: there the planets read at most 0.93 (development) and 0.12
(unseen seeds) against a threshold of 1.5, a margin of 0.57; here they read at most 0.73 on
development and 0.82 on the held-out seeds (a margin of 0.27 and 0.18 against 1.0). The margin
is narrower than ``bls``'s and is a sample maximum, not a separation. The never-seen planets of
seeds 241-246 (11 boxes) top at 0.05, so they do not narrow it further.

Edge-truncated boxes (a transit cut by the night's start or end; five of the nine single-event
development targets, 78 of the 142 planet boxes in the pool) are not treated differently:
the test is the same, and so is the threshold. What it costs: none of the 78 planet boxes
touching a night's first or last sample is refused (their statistic is -5.3 to 0.72). What it
buys is less: of the 111 eclipse boxes that touch an edge the night alone refuses 66, against 7
of 8 away from one, because the visible half of a V is a ramp down to the edge and a transit cut by the
edge is a step, and with 20 to 60 samples at 1 to 4 percent the ramp is not long enough to be
told from the step (a 1.6 h V centred 0.4 h past a night's end reads 0.4 to 2.4 across noise
seeds; ``tests/test_single_transit.py`` pins one that reads 0.49 and stays). A half V of
several hours at S/N 20 or more is refused. The residue is therefore the eclipses seen only
as a short wing, on stars with no other V night.

Measured, development (the 27 runs of #45's table: seeds 7 and 42 at 3 and 20 nights, 21-26,
the planet-free 41-44 and 41-43 at 3 nights, design 210-219), ``main`` to this: eclipse->transit
44 to 7, transit true positives 32 to 31 (the planet above), single-event planets at S/N 10 or
more bounded 20 to 20, false positives 1 to 1, pulsation->transit 2 to 2; no single-event
development target lost (five of the nine are found at both; the other four are missed at
both). On the planet-free 20-night runs 41-44 all detections go 4, 0, 1, 2 to 3, 0, 0, 2; on
the 3-night ones 3, 1, 3 to 1, 1, 1. Held-out, scored once (seeds 221-226 at 200 stars and 20
nights, planet-free 231-234; chosen before the test and the threshold existed): eclipse->transit
11 to 6, transit true positives 14 to 14, single-event planets at S/N 10 or more bounded 11
of 14 to 11 of 14, false positives 1 to 1, all detections on the planet-free runs 8 to 2. The
held-out planets' highest statistic is 0.82 (seed224 star 101), nearer the threshold than any
development planet; the six eclipse boxes that stay on the planet runs read 0.90, 0.96, 0.84,
0.35, 0.20 and -0.11.
Never-seen seeds 241-246 (planets) and 251-254 (planet-free), checked in the review of #46:
eclipse->transit 9 to 1 on the planet runs and 6 to 0 on the planet-free ones, planets
lost 0 (transit true positives 11 to 11, single-event planets at S/N 10 or
more bounded 4 of 7 to 4 of 7), false positives 0 to 0.

Pulsating hosts
---------------
Guard (a) refuses a sinusoidal star whole, and guard (b) cannot help it (a pulsator has no
eclipses to mask), so a planet that transits once on a pulsating host was found by nothing
(issue #51: PR #50's review counted 17 such hosts holding 18 transit labels that ``bls`` loses
when its repeat test is on, and ``single_transit`` bounded none of them). For a star guard (a)
refuses, the search now runs on the star with its pulsation subtracted
(``detrend_pulsation``, on by default; ``False`` is the old behaviour), before guard (b):

1. **Subtract.** The sinusoid of :meth:`BlsDetector._pulsation_trend` (the best sinusoid in the
   guard band, a free level for every night, used only when its power is 3 times the guard's
   false-alarm level) is subtracted from the prepared flux and each night is divided by its
   median again. No sinusoid: the star goes on to guard (b) as before.
2. **Search and refit.** The per-night search runs on the residual. When it finds a box, the
   sinusoid is fitted again with that box and a window 1.5 times as wide left out (a transit left
   in the fit pulls the sinusoid towards itself, the reason ``bls`` refits) and the nights are
   divided again.
3. **Gates.** A box that falls under the threshold once the residual's own flares are left out
   is refused first (Flares on a pulsator's residual, below). The boxes at the threshold (8) on that
   residual must pass guard (a) again as one-event boxes: a residual that still holds a sinusoid
   (the pulsation was not removed), another event of a periodic planet, or an eclipsing binary's
   eclipses fails it, and the star goes on to guard (b) unchanged. The shape test then runs on the
   residual like on any other star.

No threshold or bar was added or changed: the subtraction is ``bls``'s, imported unchanged, the
gates are this detector's own.

Measured on the development runs of docs/seeds.md (199 runs of 200 stars and 20 nights: 149
standard with planets, 38 standard planet-free, 8 all-sinusoidal with planets, 4 all-sinusoidal
planet-free; ``scripts/single_transit_pulsator_audit.py``). Guard (a) refuses 4324 sinusoidal
hosts, 2956 eclipsing binaries, 171 flare stars and 119 constant stars (hot pixel, cosmic ray,
cloud) on them. Only sinusoidal hosts come through: the residual of an eclipsing binary keeps
its other eclipses and fails guard (a) again (2371 of 2956; the other 582 and 3 have no single
sinusoid or no night at 8), and none of the flare or constant hosts is reported. Of the 4324
sinusoidal hosts 4168 have no night at 8 on the residual, 66 fail guard (a) on the residual
(most of them a planet with a second event), 2 fail the shape test and 85 are reported: 101
detections, 86 on a transit, 4 on a flare, 1 on a hot pixel, 1 on a cosmic ray and 9 on nothing
but the pulsation (S/N 8.2 to 10.7). The real transits it reports read from 8.2 up, so no higher
bar (``bls`` asks 9 of a detrended box) separates them from those 9: it would cost more planets than
it removes boxes. The review of the PR read the six pulsation->transit claims the detrend adds on
fresh seeds: all six sit beside a flare or a hot pixel in the same night (held-out seed 424 star 41
at 8.8 and 427 star 103 at 8.3; never-seen 712 star 75 at 182.9 beside a hot-pixel plateau, 712
star 79 at 9.8, 721 star 181 at 9.1, 722 star 55 at 10.8), about 0.4 per all-sinusoidal run (6 in 16
runs), not noise dips. On a pulsator guard (a) used to refuse the whole star, so a night with a
flare in it never reached the per-night search; after the subtraction one box is left, guard (a)
is satisfied and the flare or hot-pixel residue reads as a dip (the class this detector already
reports on quiet stars, flare->transit 4). Issue #54 gates them (Flares on a pulsator's
residual, below). Of the 18 labels of PR #50's review, 16 are bounded (``bls`` with its repeat
test on loses them, ``single_transit`` before: 0): the other two are seed 44 star 44 (the transit
night reads 7.0 raw, so guard (a) never refuses the star and the subtraction is not tried) and
the second label of puls-seed393 star 162 (the residual reads 4.0). Planet-free, all detections:
the 38 standard runs go 23 to 27 in all (four runs change, 1, 1, 0, 0 to 2, 2, 1, 1), at most 4 on
any run (4 at ``main``, seed 41), and the all-sinusoidal planet-free runs 370-373 go 0, 0, 0, 0
to 0, 1, 0, 0.

Held-out, scored once after the code was committed (seeds 420-425 standard, 426-429 every star
sinusoidal with ``planet_rate`` 0.5, 430-433 every star sinusoidal and planet-free; 200 stars,
20 nights), ``main`` to this. ``single_transit``: transit labels bounded or touched 12 to 44
(standard 12 to 15, all-sinusoidal 0 to 29), false positives 0 to 0, eclipse->transit 1 to 1,
pulsation->transit 0 to 2, detections 16 to 50; the 34 new detections are 32 transits and the 2
pulsation claims, both beside a flare in the same night (S/N 8.8, seed424 star 41, a flare label
0.4 h after the box; 8.3, seed427 star 103, the box starts inside a flare label's tail), and no
detection of ``main`` is lost. Union with ``bls`` (which does not change): transit labels
bounded or touched 154 to 176, false positives 3 to 3, eclipse->transit 5 to 5, flare->transit 0
to 0, pulsation->transit 6 to 8, detections 182 to 216. The planet-free all-sinusoidal runs
430-433 have no detection from either detector, before or after.

Never-seen check of the PR's review (seeds 700-711 standard, 712-715 every star sinusoidal with
``planet_rate`` 0.5, 720-723 every star sinusoidal and planet-free; 20 runs), union with ``bls``,
``main`` to this: transit labels bounded or touched 224 to 238 (standard 127 to 129, 712-715 97 to
109, 720-723 0 to 0), false positives 20 to 20, pulsation->transit 7 to 11, detections 289 to 328;
``single_transit`` alone bounds 25 to 60 labels and no detection of ``main`` (35) is lost. Planet-
free pulsators: 0, 1, 1, 0 detections on 720-723 (S/N 9.1 and 10.8), inside the rule of 5 a run.

Acceptance, pulsation->transit not higher: missed on both fresh sets, 6 to 8 held-out and 7 to 11
never-seen. The detector ships on by default regardless (the owner's decision): the cost is one
fixable class, the flare- or hot-pixel-adjacent claims above, and the gain is real finds (union
transit labels 154 to 176 held-out, 224 to 238 never-seen, no detection lost, fp flat).
Issue #54 is that fix (the section below).

Limits of the subtraction: one sinusoid per star (a multi-mode pulsator keeps its second mode in
the residual and fails the guard), a planet whose other events stand out of the residual is left
to ``bls`` (its periodic search), and a star whose raw nights never reach 8 is not tried.

Flares on a pulsator's residual
-------------------------------
The review of the PR found that all six new pulsation->transit claims sit beside a flare or a hot
pixel. What they have in common was measured on 60 development runs
(``scripts/single_transit_adjacent_audit.py table``; standard 420-425, 700-711, 800-805; every
star sinusoidal with ``planet_rate`` 0.5: 374-377, 392-395, 426-429, 712-715, 806-809; every star
sinusoidal and planet-free: 370-373, 430-433, 720-723, 810-813; the table is in the comment of
issue #54): 142 detections of ``main`` come from the detrended path, 127 on a transit label and 15
on a pulsation. 12 of the 15 sit inside a run of 10 to 85 consecutive samples, or up to 1.05 h
before one, that the clip of ``bls`` flags **on the residual** and not on the raw flux (11 of the
12 runs are a flare by the truth under them, one a cluster of cosmic rays). The first clip works
with the scale of the raw night, which the pulsation inflates, so a flare that would be masked on
a quiet star stays in the data; the residual's scale is the white noise's and the same clip finds
it. The leftover tail lifts the reference level of the night, and the best box of the night is a
stretch away from the flare against that lifted level. Of the 127 transit detections 14 have a run
of 10 or more samples in their night, 13 of them runs the first clip had already removed (hot
pixels, cosmic-ray clusters), and one a residual-only run: seed395 star 114 (S/N 8.6, a flare run
of 48 samples ending 1,200 s before the box).

The gate (``clip_residual``, on by default; ``False`` is the old behaviour) runs on the detrended
path only, on the boxes of the residual at the threshold, before they go to guard (a). The clip of
``bls`` (:func:`positive_outliers` with ``bls``'s ``clip_sigma`` and ``tail_sigma``) is applied to
the night of the box on the residual; a night it flags nothing in is left as it is. Otherwise the
box's S/N is measured again over its own samples with the flagged samples left out of both the box
and the reference (the contrast statistic of :func:`search_night`, ``sigma`` the scatter outside
the box), and the box is refused when that S/N is under ``snr_threshold`` (8, the detector's own
bar), or when fewer than ``min_in`` samples are left in the box or ``min_out`` outside it. A
refusal removes a box; a star left with no box goes to guard (b) like any star whose detrended
search failed. It added no detection on 88 runs, but that is not structural: the gate runs before
guard (a), so removing the strongest box changes which box the guard reads, and a star left with
no box falls to the eclipse-mask search. No distance, window length or bar was added: a variant
that also leaves out 1 h or 3 h after each run of 10 or more samples decides the same 15
detections, and a distance rule would not separate them (the pulsation claims lie from 0 to 3,780
s before their runs, the one transit 1,200 s after its).

Measured, ``main`` to this. Development (the 60 runs above, where the gate was read; it has no
constant to fit): it removes 13 detections, 12 pulsation claims and 1 transit, and added none.
``single_transit`` alone: pulsation->transit 15 to 3, transit labels bounded 176 to 175, bounded or
touched 177 to 176, planets with a transit event bounded or touched 150 to 149, false positives 1
to 1, eclipse->transit 4 to 4, flare->transit 5 to 5, detections 211 to 198. Union with ``bls``
(unchanged): transit labels bounded or touched 768 to 767, planets 253 to 252, fp 34 to 34,
pulsation->transit 69 to 57, detections 1018 to 1005. The six known claims: seed424 star 41 (8.8),
seed427 star 103 (8.3), seed712 star 79 (9.8), seed721 star 181 (9.1) and seed722 star 55 (10.8)
are no longer reported (their S/N with the residual's flagged samples left out is 6.1, 1.3, 2.8,
3.3 and 1.0); seed712 star 75 (182.9) is still reported, for the cause below. The planet-free
all-sinusoidal runs go from 5 ``single_transit`` detections (371, 721, 722, 811 and 813, one each)
to 0. The one transit lost is seed395 star 114, planet 56 (depth 0.17 percent), one of its twelve
labels, found by ``single_transit`` alone (``bls`` has none of its labels): the flare run ends 20
minutes before the box and the box reads 3.8 without it.

Held-out, scored once after the code was committed (seeds 820-825 standard, 826-829 every star
sinusoidal with ``planet_rate`` 0.5, 830-833 every star sinusoidal and planet-free; 200 stars, 20
nights), ``main`` to this. ``single_transit``: pulsation->transit 4 to 0, transit labels bounded 37
to 37, bounded or touched 38 to 38, planets 34 to 34, fp 0 to 0, detections 42 to 38; the four
detections removed are all pulsation claims (seed826 star 162 at 10.6, seed830 stars 54 and 108 at
10.3 and 9.8, seed831 star 185 at 8.9) and no transit is lost. On the held-out the bar is not as
slack as on development: seed826 star 162 reads 7.58 without its flagged samples, so a bar under
7.5 would have kept it (8 is the detector's own bar, not a fit). Union with ``bls``: transit labels
bounded or touched 203 to 203, planets with a transit event bounded or touched 62 to 62, fp 7 to 7,
eclipse->transit 0 to 0, flare->transit 1 to 1, pulsation->transit 20 to 16 (the 16 are ``bls``'s
own), detections 253 to 249. The planet-free runs 830-833 go from 3 to 0 ``single_transit``
detections.

Never-seen check of the PR's review (seeds 500-505 standard, 506-509 every star sinusoidal with
``planet_rate`` 0.5, 510-513 every star sinusoidal and planet-free; 14 runs), ``main`` to this,
``single_transit`` alone / union with ``bls``: planets with a transit event bounded or touched
31 to 31 / 62 to 62, fp 0 to 0 / 3 to 3, pulsation->transit 5 to 2 / 11 to 8, detections 47 to 44 /
225 to 222. Standard 500-505 nothing changes (19 detections, union 59, on both); 506-509
pulsation->transit 2 to 1 (detections 25 to 24); the planet-free 510-513 go from 3 to 1 detection
(the two removed are pulsation claims beside a flare). ``bls`` is unchanged on them.

What the gate does not see: a hot-pixel plateau that covers most of a short night is the night's
median, so the unaffected stretch is the dip and nothing is above the level for a clip to flag
(seed712 star 75, S/N 182.9, a night of 96 samples; seed375 star 80, 142.0, 53 samples; seed376
star 37, 51.2, 67 samples; the boxes are 40 to 60 percent deep, the deepest planet injected in
these runs 3 percent). These three are the pulsation->transit that remain on the development
runs; none appears in the held-out ones. The gate judges the boxes the search found and does not
clip the residual before the search, so that it can only remove claims; a night with a long
flare and a real dip still has the flare in the residual that guard (a) reads, and the star is
refused as before (a transit and a short flare on one pulsator are both kept).

Limits
------
* A second strong event stays in the residual of the guard: a star with two single events of
  S/N 10 or more each is read as a sinusoid and not searched (the residual power of a dip grows
  with its S/N squared; at S/N 9 it is at the level of the false-alarm limit). A periodic
  planet's other events do the same.
* A planet whose transit overlaps or abuts the masked eclipses of its binary host is lost with
  them (seed7-200s-20n star 99: the planet's transit starts 20 minutes after an eclipse ends and
  the mask, 21 h of every 44 h cycle, covers it). Other eclipse residue gets through: the
  eclipse->transit confusion on the held-out 20-night runs 21-26 was 0 to 7 per run at the
  default before the shape test (12 over six runs) and is 0, 0, 3, 0, 0, 0 with it: not zero.
* A star the guards reject is lost with its planet when the star is a flare variable or a
  sinusoidal one the subtraction does not clean (see Pulsating hosts): held-out seed25 star 52
  (a flare star) is found at S/N 9.5 to 15.6 with ``guards=False`` and dropped with them on;
  seed22 star 81 (sinusoidal) is now reported (S/N 11.4), and seed25 star 39 (sinusoidal) still
  is not (its residual reads 4.8). A planet-only host whose guard ratio (see Guards) crosses 1 is
  the same loss.
* Only the best box per night is reported: two dips in one night are one detection.
* A dip that fills most of a night leaves little reference (``min_out`` 30) and a night
  median taken inside the dip; the S/N is then the lower, contrast, one. When the night holds
  too few samples for any box to leave ``min_out`` outside it, the night is not searched:
  held-out seed25 star 121 (a constant host, true stacked S/N 15.2) has 56 observed samples on
  its night, 44 inside the label, so no box has ``n_out >= 30`` and the dip is missed.

Coupling with ``bls``
---------------------
Guard (b) tests ``cand.rejected != "pulsation guard"``: a star whose periodic search is not
rejected as a pulsator is not masked and re-searched, it is dropped. ``bls`` now detrends
pulsating hosts (``detrend_pulsation``, #35), so ``_search`` returns a reportable candidate for
some of them, and a host it accepts is no longer masked and re-searched here. Measured with
both merged (14 runs: the 4 development runs and 6 held-out, 4 more planet-free): four stars
flip from guard-rejected to accepted inside this module's ``bls`` searches (periods 8.225, 3.601,
0.869 and 3.446 d, planets of seed 21 stars 83 and 85, seed 22 star 6 and seed 24 star 37), and
guard (b) now drops them instead of masking them; their re-search found nothing on a pulsating
host before either, and no eclipsing binary flips (its harmonics are not accepted as a
sinusoid). The detections are identical, span for span and score for score, on the 4
development and 6 held-out runs, and the planet-free runs give the same all-detection counts
(4, 0, 1, 2). The cost is real: each guard-rejected star is searched by ``bls`` again with its
pulsation subtracted, about 2.2 times the CPU time of this detector on a 20-night run (seed26:
15.8 s to 35.0 s). It would change the output only for a pulsating host that carries a
single-event planet and also a periodic box that clears the bar of 9, which no run contains.
Issue #39 gave ``bls`` two more reasons to reject a box (a sinusoid that gains 0.81 of the box's
chi-square, named ``"pulsation guard"`` like the other, and a local S/N below 7.2 on a flare star,
named ``"local snr"``); guard (b) reads only whether the reason is ``"pulsation guard"``. Measured:
the detections of this module are identical, span for span and score for score, on 33 runs (seeds 7
and 42 at 3 and 20 nights, 21-26, 71-76, the planet-free runs 41-44 and 51-58, and five development
runs with a planet-free pulsator).
Issue #43 adds a last gate to ``bls``, ``"flare tails"`` (it runs after the pulsation guard, so a
pulsation rejection is never replaced by it), and a candidate that was accepted is now rejected for
it on some flare stars; guard (b) reads only whether the reason is ``"pulsation guard"``, as before.
Measured: identical detections, span for span and score for score, on 27 runs (seeds 7 and 42 at 3
and 20 nights, 11, 12, 21-26, 61, 62, 94, 96, 114 and the planet-free runs 42, 43, 51, 52, 55-58, 82
and 104).
The module also uses ``BlsDetector._clean``,
``_common_mode``, ``_prepare``, ``_pulsating``, ``_search``, ``_shape_excess`` and the fields
of ``Box`` and ``Candidate.mask``; a rename in ``bls`` breaks it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from skywatch.detectors import register
from skywatch.detectors.bls import (
    _MAD_SCALE,
    BlsDetector,
    Box,
    _Arrays,
    _Clean,
    _renormalise_nights,
    in_box,
    positive_outliers,
)
from skywatch.types import Detection, LightCurve

log = logging.getLogger(__name__)

HOUR_S = 3600.0
DEFAULT_DURATIONS_H = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIN_OUT_FOR_LEVEL = 5  # a night is re-levelled on at least this many out-of-box samples
_ONE_EVENT_PERIOD_S = 1e9  # a "period" longer than any survey: the box has one window


@dataclass(frozen=True)
class NightBox:
    """The best box of one night: sample indices into that night's arrays and its statistics."""

    night: int
    t_start: int  # first covered sample (on the global axis)
    t_end: int  # last covered sample
    depth: float
    n_in: int
    n_out: int
    sigma: float
    snr: float
    shape: float | None = None  # the shape test's statistic (None when it was not run)


def search_night(
    t: np.ndarray,
    flux: np.ndarray,
    cadence_s: int,
    durations_s: tuple[float, ...],
    min_in: int,
    min_out: int,
    contrast: bool = True,
) -> tuple[float, float, int, int, float, int, int] | None:
    """Best box of one night: ``(snr, depth, n_in, n_out, sigma, first, last)`` or None.

    `t` is the night's sample times (sorted) and `flux` its normalised flux. `first` and
    `last` are the positions in `t` of the first and last sample the box covers. A box is
    positioned on the night's regular sample grid; the grid extends past both ends of the
    night so a box may be truncated by either end (it must still hold `min_in` samples).
    """
    n = len(t)
    if n < min_in + min_out:
        return None
    slot = np.rint((t - t[0]) / cadence_s).astype(np.int64)
    length = int(slot[-1]) + 1
    count = np.zeros(length + 1)
    total = np.zeros(length + 1)
    count[slot + 1] = 1.0
    total[slot + 1] = flux
    count = np.cumsum(count)
    total = np.cumsum(total)
    n_all, f_all = float(count[-1]), float(total[-1])
    # (gain, depth, n_in, n_out, lo, hi); gain = depth * sqrt(effective n) is S/N * sigma, so the
    # ranking needs no sigma and sigma can be measured outside the winning box
    best: tuple[float, float, int, int, int, int] | None = None
    for dur in durations_s:
        w = max(1, round(dur / cadence_s))
        start = np.arange(-(w - 1), length)
        lo = np.clip(start, 0, length)
        hi = np.clip(start + w, 0, length)
        n_in = count[hi] - count[lo]
        n_out = n_all - n_in
        ok = (n_in >= min_in) & (n_out >= min_out)
        if not ok.any():
            continue
        lo, hi, n_in, n_out = lo[ok], hi[ok], n_in[ok], n_out[ok]
        s_in = total[hi] - total[lo]
        depth = (f_all - s_in) / n_out - s_in / n_in
        gain = depth * np.sqrt(n_in * n_out / n_all if contrast else n_in)
        i = int(np.argmax(gain))
        if gain[i] > 0 and (best is None or gain[i] > best[0]):
            best = (
                float(gain[i]),
                float(depth[i]),
                int(n_in[i]),
                int(n_out[i]),
                int(lo[i]),
                int(hi[i]),
            )
    if best is None:
        return None
    gain, depth, n_in, n_out, lo, hi = best
    inside = (slot >= lo) & (slot < hi)
    out = flux[~inside]
    sigma = _MAD_SCALE * float(np.median(np.abs(out - np.median(out))))
    if not sigma > 0:
        return None
    covered = np.flatnonzero(inside)
    return gain / sigma, depth, n_in, n_out, sigma, int(covered[0]), int(covered[-1])


@register
class SingleTransitDetector:
    """A one-event box search per night; see the module docstring."""

    name = "single_transit"

    def __init__(
        self,
        snr_threshold: float = 8.0,
        durations_h: tuple[float, ...] = DEFAULT_DURATIONS_H,
        min_in: int = 15,
        min_out: int = 30,
        contrast: bool = True,
        guards: bool = True,
        mask_pad_h: float = 0.5,
        shape_threshold: float | None = 1.0,
        detrend_pulsation: bool = True,
        clip_residual: bool = True,
        min_night_h: float = 2.0,
    ) -> None:
        if not snr_threshold > 0:
            raise ValueError(f"snr_threshold must be > 0, got {snr_threshold}")
        if len(durations_h) == 0 or not all(d > 0 for d in durations_h):
            raise ValueError(f"durations_h must be non-empty and positive, got {durations_h}")
        if min_in < 1:
            raise ValueError(f"min_in must be >= 1, got {min_in}")
        if min_out < 1:
            raise ValueError(f"min_out must be >= 1, got {min_out}")
        if mask_pad_h < 0:
            raise ValueError(f"mask_pad_h must be >= 0, got {mask_pad_h}")
        if min_night_h < 0:
            raise ValueError(f"min_night_h must be >= 0, got {min_night_h}")
        if shape_threshold is not None and not shape_threshold > 0:
            raise ValueError(f"shape_threshold must be > 0 or None, got {shape_threshold}")
        self.snr_threshold = snr_threshold
        self.durations_h = tuple(sorted(durations_h))
        self.min_in = min_in
        self.min_out = min_out
        self.contrast = contrast
        self.guards = guards
        self.mask_pad_h = mask_pad_h
        self.shape_threshold = shape_threshold
        self.detrend_pulsation = detrend_pulsation
        self.clip_residual = clip_residual
        self.min_night_h = min_night_h
        self._bls = BlsDetector()

    # -- search ----------------------------------------------------------------------------

    def _boxes(
        self, t: np.ndarray, flux: np.ndarray, night: np.ndarray, cadence_s: int
    ) -> list[NightBox]:
        """The best box of each night whose S/N reaches the threshold (night order)."""
        durations = tuple(h * HOUR_S for h in self.durations_h)
        shortest = self.min_night_h * HOUR_S / cadence_s  # samples a night needs to be searched
        out: list[NightBox] = []
        for k in np.unique(night):
            idx = np.flatnonzero(night == k)
            if len(idx) < shortest:
                continue
            found = search_night(
                t[idx], flux[idx], cadence_s, durations, self.min_in, self.min_out, self.contrast
            )
            if found is None or found[0] < self.snr_threshold:
                continue
            snr, depth, n_in, n_out, sigma, first, last = found
            out.append(
                NightBox(
                    int(k), int(t[idx[first]]), int(t[idx[last]]), depth, n_in, n_out, sigma, snr
                )
            )
        return out

    # -- guards ----------------------------------------------------------------------------

    @staticmethod
    def _level_corrected(
        t: np.ndarray, flux: np.ndarray, night: np.ndarray, boxes: list[NightBox]
    ) -> np.ndarray:
        """`flux` with the night of each box in `boxes` divided by its out-of-box median.

        The first-pass night median includes the dip, which lifts the whole night by a fraction
        of the depth: a step the pulsation guard would read as a low-frequency sinusoid.
        """
        flux = flux.copy()
        for b in boxes:
            on_night = night == b.night
            out = on_night & ~((t >= b.t_start) & (t <= b.t_end))
            if out.sum() >= _MIN_OUT_FOR_LEVEL:
                med = float(np.median(flux[out]))
                if med > 0:
                    flux[on_night] /= med
        return flux

    def _pulsating(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        night: np.ndarray,
        boxes: list[NightBox],
        cadence_s: int,
    ) -> bool:
        """True when the `bls` pulsation guard rejects the star's strongest box as a one-event box."""
        best = max(boxes, key=lambda b: b.snr)
        flux = self._level_corrected(t, flux, night, [best])
        box = Box(
            period_s=_ONE_EVENT_PERIOD_S,
            epoch_s=(best.t_start + best.t_end) / 2.0,
            duration_s=float(best.t_end - best.t_start + cadence_s),
            depth=best.depth,
            n_in=0,
            n_total=len(t),
            power=0.0,
        )
        box = replace(box, n_in=int(in_box(t, box).sum()))
        outside = flux[~in_box(t, box)]
        if len(outside) < 2:
            return True
        sigma = _MAD_SCALE * float(np.median(np.abs(outside - np.median(outside))))
        if not sigma > 0:
            return False
        rejected, _ = self._bls._pulsating(t, flux, sigma, cadence_s, box)
        return rejected

    # -- shape test ------------------------------------------------------------------------

    def _shape(
        self, t: np.ndarray, flux: np.ndarray, night: np.ndarray, b: NightBox, cadence_s: int
    ) -> float:
        """The `bls` shape statistic of one night's box, read as a one-event box.

        A triangle against a box fitted to the night's dip, both with a free level
        (:meth:`BlsDetector._shape_excess`): positive for a V, negative for a flat bottom, zero
        when the night holds too few samples for a verdict. `sigma` is the night's scatter
        outside the box (the star-wide scatter ``bls`` passes is not defined here).
        """
        on = night == b.night
        box = Box(
            period_s=_ONE_EVENT_PERIOD_S,
            epoch_s=(b.t_start + b.t_end) / 2.0,
            duration_s=float(b.t_end - b.t_start + cadence_s),
            depth=b.depth,
            n_in=b.n_in,
            n_total=int(on.sum()),
            power=0.0,
        )
        return self._bls._shape_excess(
            t[on], flux[on], np.zeros(int(on.sum()), dtype=np.int64), b.sigma, box
        )

    def _event_box(self, b: NightBox, cadence_s: int, n_total: int) -> Box:
        """The night box `b` as a one-event :class:`Box` (a "period" longer than the survey)."""
        return Box(
            period_s=_ONE_EVENT_PERIOD_S,
            epoch_s=(b.t_start + b.t_end) / 2.0,
            duration_s=float(b.t_end - b.t_start + cadence_s),
            depth=b.depth,
            n_in=b.n_in,
            n_total=n_total,
            power=0.0,
        )

    def _residual(
        self,
        c: _Clean,
        t: np.ndarray,
        flux: np.ndarray,
        night: np.ndarray,
        level: np.ndarray,
        cadence_s: int,
    ) -> np.ndarray | None:
        """The star's flux with its pulsation subtracted, or None when no sinusoid stands out.

        The sinusoid of :meth:`BlsDetector._pulsation_trend` (fitted to the whole record, the
        night levels free) is subtracted and each night divided by its median again. When the
        per-night search finds a box on that residual, the sinusoid is fitted again with the
        box's window left out (a transit left in the fit pulls the sinusoid towards itself) and
        the nights are divided again.
        """
        bls = self._bls
        a = _Arrays(t, flux, night, level, c.t, c.night_of)
        fit = bls._pulsation_trend(a, cadence_s)
        if fit is None:
            return None
        resid = _renormalise_nights(flux - fit[0], night)
        boxes = self._boxes(t, resid, night, cadence_s)
        if not boxes:
            return resid
        best = max(boxes, key=lambda b: b.snr)
        fit = bls._pulsation_trend(a, cadence_s, self._event_box(best, cadence_s, len(t)))
        if fit is None:
            return None
        return _renormalise_nights(flux - fit[0], night)

    def _stands_unflared(
        self, t: np.ndarray, flux: np.ndarray, night: np.ndarray, b: NightBox
    ) -> bool:
        """False when box `b` falls under the threshold once its night's flares are left out.

        The flares are the positive outliers of ``bls``'s clip (:func:`positive_outliers`)
        applied to the night of the *residual*, whose scale is the white noise's and not the
        pulsation's. A night the clip flags nothing in stands as it is. Otherwise the box's S/N is
        measured again without the flagged samples, with the statistic of :func:`search_night`
        (the contrast form, ``sigma`` the scatter outside the box) over the box's own samples, and
        must reach ``snr_threshold``; a box left with fewer than ``min_in`` samples inside or
        ``min_out`` outside, or with no scatter to measure, is refused.
        """
        idx = np.flatnonzero(night == b.night)
        tk, xk = t[idx], flux[idx]
        flagged = positive_outliers(xk, self._bls.clip_sigma, self._bls.tail_sigma)
        if not flagged.any():
            return True
        inside = (tk >= b.t_start) & (tk <= b.t_end)
        box, out = xk[~flagged & inside], xk[~flagged & ~inside]
        n_in, n_out = len(box), len(out)
        if n_in < self.min_in or n_out < self.min_out:
            return False
        sigma = _MAD_SCALE * float(np.median(np.abs(out - np.median(out))))
        if not sigma > 0:
            return False
        snr = (float(out.mean()) - float(box.mean())) * math.sqrt(n_in * n_out / (n_in + n_out))
        return snr / sigma >= self.snr_threshold

    def _detrended(
        self,
        c: _Clean,
        t: np.ndarray,
        flux: np.ndarray,
        night: np.ndarray,
        level: np.ndarray,
        cadence_s: int,
    ) -> tuple[list[NightBox], np.ndarray] | None:
        """The star's boxes once its pulsation is subtracted, and that flux; None if there is none.

        The per-night search runs on :meth:`_residual`. The result stands only if the residual
        passes the pulsation guard: a pulsator this subtraction does not clean, a binary, and a
        planet whose other events stand out of the residual all fail it, and are left to the
        eclipse mask.
        """
        resid = self._residual(c, t, flux, night, level, cadence_s)
        if resid is None:
            return None
        boxes = self._boxes(t, resid, night, cadence_s)
        if boxes and self.clip_residual:
            boxes = [b for b in boxes if self._stands_unflared(t, resid, night, b)]
        if not boxes or self._pulsating(t, resid, night, boxes, cadence_s):
            return None
        return boxes, resid

    def _star_boxes(
        self,
        star_id: int,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
    ) -> tuple[list[NightBox], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The star's reportable boxes after the guards, with the arrays they were found in."""
        t, flux, night, level = self._bls._prepare(c, common_mode)
        none: list[NightBox] = []
        if len(t) < self.min_in + self.min_out:
            return none, (t, flux, night)
        boxes = self._boxes(t, flux, night, cadence_s)
        if not boxes:
            return none, (t, flux, night)
        refused = self.guards and self._pulsating(t, flux, night, boxes, cadence_s)
        if refused and self.detrend_pulsation:
            found = self._detrended(c, t, flux, night, level, cadence_s)
            if found is not None:
                log.info("single_transit star %s: pulsation subtracted", star_id)
                boxes, flux = found
                refused = False
        if refused:
            cand = self._bls._search(c, common_mode, cadence_s, plan_mask=True)
            if cand is None or cand.rejected != "pulsation guard" or not cand.mask:
                log.info("single_transit star %s: pulsation guard, not searched", star_id)
                return none, (t, flux, night)
            drop = np.zeros(len(t), dtype=bool)
            # bls pads each window by one box duration per side; a box here is at least
            # `mask_pad_h` wide, so a narrower leftover is not a candidate
            mask = tuple(
                replace(
                    m,
                    duration_s=m.duration_s
                    - 2 * cand.box.duration_s
                    + 2 * self.mask_pad_h * HOUR_S,
                )
                for m in cand.mask
            )
            for m in mask:
                drop |= in_box(t, m)
            t, flux, night = t[~drop], flux[~drop], night[~drop]
            flux = _renormalise_nights(flux, night)
            boxes = self._boxes(t, flux, night, cadence_s) if len(t) >= self.min_in else []
            if not boxes or self._pulsating(t, flux, night, boxes, cadence_s):
                log.info("single_transit star %s: pulsation guard, not searched", star_id)
                return none, (t, flux, night)
            log.info(
                "single_transit star %s: eclipses masked (%d windows)", star_id, len(cand.mask)
            )
        if self.shape_threshold is not None:
            boxes = [replace(b, shape=self._shape(t, flux, night, b, cadence_s)) for b in boxes]
        return boxes, (t, flux, night)

    def _star(
        self,
        star_id: int,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
    ) -> list[Detection]:
        boxes, _ = self._star_boxes(star_id, c, common_mode, cadence_s)
        if self.shape_threshold is not None and any(
            b.shape is not None and b.shape >= self.shape_threshold for b in boxes
        ):
            log.info("single_transit star %s: V-shaped event, read as an eclipse", star_id)
            return []
        out = []
        for k, b in enumerate(boxes):
            out.append(
                Detection(
                    detector=self.name,
                    det_id=f"{self.name}-s{star_id}-{k}",
                    star_id=star_id,
                    t_start=b.t_start,
                    t_end=b.t_end + cadence_s,
                    score=b.snr,
                    kind_guess="transit",
                )
            )
        return out

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        bls = self._bls
        cleaned: dict[int, _Clean] = {}
        for star_id in sorted(curves):
            c = bls._clean(curves[star_id], cadence_s)
            if c is not None:
                cleaned[star_id] = c
        common_mode = bls._common_mode(cleaned)
        out: list[Detection] = []
        for star_id, c in cleaned.items():
            out.extend(self._star(star_id, c, common_mode, cadence_s))
        return out
