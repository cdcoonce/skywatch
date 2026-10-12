"""Box least squares across nights: the periodic-transit detector.

Every other detector looks at one night at a time, so a transit shallower than the night's
noise is invisible to it. A planet that comes back every few days shows up only when the
nights are folded on its period. This detector searches each star's whole record for the
box-shaped dip that best explains the folded flux and reports every predicted event of the
best box (``kind_guess="transit"``, ``score`` = the box's S/N), one :class:`Detection` per
event. It is plain numpy; stars are processed in ``sorted`` order and nothing is random.

Preprocessing
-------------
Per star:

1. Samples with ``quality`` bit 0 set (instrument-flagged field-wide spans) and non-finite
   samples are dropped.
2. Nights are split with :func:`skywatch.detectors.nights.split_nights` (a gap of more than
   ``night_gap_cadences`` = 100 cadences starts a new night) and each night is divided by its
   own median.

Field-wide, once: **common-mode removal.** A tracking drift keeps its new level after the
flagged span ends and a cloud's wings are not flagged, so a field-wide step of 5-30 percent
survives the quality flag on a few nights and, for a BLS, is the deepest "transit" on every
star. Each star's night-normalised flux is divided by the per-time median over all stars
(times seen by fewer than ``common_mode_min_stars`` = 10 stars are left alone) and
renormalised per night. Fewer than 10 stars in the field switches this off. Then per star:

3. Positive outliers are masked one-sidedly (:func:`positive_outliers`). With ``s =
   1.4826 * MAD`` of the night's normalised flux, samples above ``1 + clip_sigma * s``
   (``clip_sigma = 4.0``) are seeds, each grown into the contiguous run where the 5-sample
   running mean stays above ``1 + tail_sigma * s / sqrt(5)`` (``tail_sigma = 1.0``): a
   flare's decay stays above the noise long after its peak crossed the seed threshold, and
   the mean-based box statistic is moved by that mass (on a faint star one flare tail was
   half of a "transit"). Flares, cosmic rays and hot pixels are positive, a transit is
   negative, so a dip is never masked. Masked samples are still *observed*: they count when
   deciding which samples an event window covers, so a flare on a transit does not move the
   reported span.
4. After the search each night is divided again by the median of its samples outside the box
   (the first-pass median includes the transit, which pulls it low on a night with a long
   event), and ``sigma``, the scatter used for the S/N, is ``1.4826 * MAD`` of the
   normalised flux outside the box.

Grid
----
Trial periods run from ``min_period_d`` (0.5 d) to ``max_period_d`` (default
``baseline / 2``, so every candidate can have two events), uniform in ``log(frequency)``
with relative step ``(min_duration / baseline) / oversample`` (``oversample = 3``). The
relative step keeps the last event of a candidate within ``min_duration / oversample`` of
where the grid period puts it, at every period; read literally as a step in 1/day the
formula is not dimensionally consistent. Durations are ``(0.5, 1, 1.5, 2, 3, 4, 6)`` hours.
For each trial period the samples are folded into phase bins no wider than half the shortest
duration (``P / ceil(P / bin_s)``, ``bin_s = min_duration / 2``); samples are first merged into
``bin_s / 3`` time buckets (300 s: a 5x saving at 60 s cadence, at most a sixth of a phase bin
of timing error), binned sums and counts are accumulated with ``np.bincount`` for a chunk of
periods at once, and every duration is a box sum over consecutive bins (wrapping through the
phase origin) read from a cumulative sum. A box is ``round(duration / bin_width)`` bins wide, so
the evaluated width differs from the nominal duration by at most half a bin; the reported event
windows use the evaluated width. The phase is stepped by one bin. The winning box is then
re-measured on the samples themselves (:func:`refine`).

Statistic
---------
For a box with ``n_in`` samples of mean ``m_in`` among ``N`` samples, the out-of-box mean
is ``m_out`` over ``n_out = N - n_in`` samples and ``depth = m_out - m_in``. Boxes are
ranked by ``depth**2 * n_in * n_out / N`` (the chi-square gain over a constant, with sigma
common to all boxes), only ``depth > 0`` (a dip) counting. The reported significance is
``snr = depth * sqrt(n_in) / sigma``, the matched-filter S/N of the box when the
out-of-box level is known.

Acceptance of the best box
--------------------------
The best box per star is a candidate when all of these hold, tested in this order
(:attr:`Candidate.rejected` names the first that fails):

* ``snr >= snr_threshold`` (7.0).
* ``snr_local >= snr_threshold``. Dividing each night by its own median makes the
  reference level noisy, so a box that covers most of its nights (4-6 h of an 8 h night)
  has a much wider null distribution than ``snr`` assumes: on the stored runs such boxes
  reached ``snr`` 8-12 on stars with no planet. ``snr_local`` is the S/N of the same box
  measured as a contrast against the same night's out-of-box samples, no night level
  involved: ``sum(w_k d_k) / (sigma * sqrt(sum w_k))`` with ``d_k`` the night's
  out-minus-in mean and ``w_k = n_in n_out / (n_in + n_out)``. It equals ``snr`` when the
  box is small against its nights and is lower otherwise. A flare star (``flare_fraction`` =
  0.03 of its clean samples or more masked as positive outliers in step 3) must reach
  ``flare_snr_threshold`` = 7.2 instead (see Planet-free stars).
* The predicted windows cover observed samples on at least ``min_events`` (2) distinct
  nights.
* At least ``min_events`` nights each show the dip at ``support_snr`` (2.0) on their own, so
  one bad night cannot make a candidate.
* The pulsation guard passes (below), including the sinusoid-against-box ratio at the bar of
  0.81.
* The shape guard passes (below).
* The flare-tail gate passes (below): ``snr_clean``, the local S/N with the samples inside and the
  ``flare_window_h`` = 3 hours after each flare-sized masked run left out of the contrast, reaches
  ``snr_threshold``. It is the last gate, so a pulsator or a binary is still named by its own.

Pulsation guard
---------------
A sinusoidal star folds into a dip on many periods and its pulsations are far above any
transit threshold, so a box is also tested against sinusoids (:meth:`BlsDetector._pulsating`):

* *Box versus sine.* The chi-square gain of the best sinusoid in 1-30 cycles/day, fitted
  to the whole record, against the box's: rejected when ``2 * power >= guard_ratio *
  depth**2 * n_in * n_out / (N * sigma**2)`` (``guard_ratio = 1``). One sinusoid keeps at
  most 81 percent of a square wave's variance (54 percent at a quarter duty cycle, 10
  percent at five), and the overlap is symmetric: the best box keeps 81 percent of a sinusoid's
  variance at most. So this half is the weaker one against a pulsator (measured
  ``box chi2 / (2 * sine power)`` is 0.94 at 1.3 d, 0.28 at 5 h, 0.47 at 7 h: at 1.3 d it is
  6 percent from passing) and the residual test below is what guarantees its rejection.
* *Residual.* The box (over the inner 0.8 of its duration, the rest of a window 1.5 times
  as wide left out) is subtracted and the Fourier power of what is left is searched for a
  peak: a transit leaves white noise, a pulsator its pulsation, an eclipsing binary its
  second eclipse. The false-alarm level is ``ln(M / guard_fap)``, ``M`` the number of
  independent frequencies and ``guard_fap = 1e-3`` per star. ``guard_fap=None`` switches
  both off.

Box versus sine with the nights' levels free
--------------------------------------------
The box-versus-sine test above is the weaker half against a pulsator and, since the nights are
divided by their own medians, it compares unlike things: a plain Fourier power (no night level) and
a box chi-square, both in units of a scatter that contains the pulsation. A third test
(:meth:`BlsDetector._night_sine_ratio`, ``guard_night_ratio`` = 0.81, ``None`` switches it off,
``Candidate.sine_ratio``) puts both on the same footing: the chi-square gain of the best
sinusoid of 1-30 cycles/day fitted to the whole record with a free level for every night (the
six strongest peaks placed by least squares, as the detrending does) over the box's gain with a
free level for every night, ``(sum w_k d_k)**2 / sum w_k`` over the nights of
:func:`night_contrasts` (the square of ``snr_local * sigma``). The scatter cancels. A sinusoid keeps
at most 81 percent of a square wave's variance and a box at most 81 percent of a sinusoid's, so the
ratio is at most about 0.81 for a box train and at least about 1.23 for a pulsator whose best box is
its fold; the bar is the first number, a bound of the model and not a fitted value. The box is
rejected as ``"pulsation guard"`` (so a pulsating host is still detrended) when the ratio reaches
it. It runs after the two tests above and only for a candidate that passed them.

Planet-free stars
-----------------
On a run with no planets every transit claim is false. ``single_transit``'s threshold was set by a
rule on such runs (at most 5 detections on each planet-free 20-night run, issue #33); ``bls`` was
held to it in issue #39. At ``main`` after the shape gate the four 20-night planet-free runs (200
stars, seeds 41 to 44) gave 0, 4, 2 and 10 detections and eight more simulated for design (seeds
51 to 58) 2, 2, 3, 12, 2, 2, 6 and 23. Over the 41 runs without a planet on the star (the 33
development runs and those eight) 41 of 763 flare-star runs, 8 of 920 sinusoidal ones, 2 of 4986
constant ones and 3 of 558 eclipsing binaries made a claim (129, 122, 4 and 9 detections). Two
gates, chosen on those runs and nothing else:

* **Sinusoid against box** (above). Seven of the eight pulsators had been accepted at the first
  pass (6 to 12 hour sinusoids of 0.15 to 0.85 percent in 8 hour nights, boxes of 0.5 to 1.35 d on
  the same pulsation phase every night). The box-versus-sine test read 0.53 to 0.99 on them (it
  rejects at 1); the ratio with free nightly levels reads 1.15 to 1.40 on six and 0.97 on the
  seventh (seed 38 star 119: 9.3 h, 0.15 percent). On the 93 development candidates on a star with
  a planet it reads at most 0.75 (seed 33 star 118, whose detections match a cloud and a flare, not
  the planet), 0.73 for the highest where the detection is the planet (seed 38 star 168, a 0.509 d
  planet with 20 events; a second real planet reads 0.73 on a seed used before, seed 23 star 114,
  1.39 d, 5.6 h, 6 events) and 0.57 or less for every other. The eighth pulsator (seed 13 star 130)
  came through the detrending and is not a pulsation effect: its box holds 9 samples with a 9.9
  percent depth in two windows of 10 and 8 minutes, beside a cosmic ray and just before a flare
  onset; it is left (2 pulsation-labelled detections).
* **Flare stars.** Flare tails are red noise that ``sigma`` (the MAD) does not contain: the claims
  on flare stars have a local S/N of 7.0 to 19.5 (median 7.7) with boxes of 2 to 6 hours, and on
  seed 44 star 199 (7 events, P 1.512 d) every event touches the start or the end of its night and
  five of the seven nights have 32 to 74 flare samples masked. A star with ``flare_fraction``
  (0.03) of its clean samples masked as positive outliers must reach a local S/N of
  ``flare_snr_threshold`` (7.2) instead of 7.0. 32 of the 41 flare-star claims are on such stars and
  no candidate of another class is (they are under 2.8 percent), so the cut is a measured
  separation and not a fit. The bar is the lowest on a grid of 0.1 that brings every 20-night planet-free run
  of the development and design seeds to 5 or fewer: at 7.1 seed 44 is still at 10 (star 199's
  local S/N is 7.107), from 7.15 to 7.19 at 3 (star 126, 7.195), at 7.2 at 0. The same bar applied to every star
  would have lost a development transit true positive (seed 61 star 56, local S/N 7.04); on flare
  stars the lowest local S/N of a transit true positive is 7.28 (seed 12 star 32), 0.08 above the
  bar. The bar is not in a gap of the false claims, though: see Flare tails, which is what holds the
  rest.

Measured. Development (seeds 7 and 42 at 3 and 20 nights, 11-14, 31-50 and 61-66 at 20 nights; 33
runs), ``main`` to the branch: transit true positives 271 to 271 (270 bounded and 1 touched, both
before and after), false positives (all kinds) 74 to 58, eclipses reported as transits 8 to 8,
pulsations reported as transits 81 to 3 (52 to 3 on seeds 7-50 and 29 to 0 on 61-66; the 3 are seed
13 star 130 and seed 44 star 44, a planet host), detections 533 to 402. Planet-free 20-night runs
(seeds 41-44, then the design seeds 51-58), detections per run: 0, 4, 2, 10 to 0, 4, 2, 0 and 2, 2, 3,
12, 2, 2, 6, 23 to 2, 2, 0, 0, 2, 2, 3, 5; the 3-night runs (0, 2, 0) are unchanged. Held-out,
scored once after both gates were committed (seeds 71-76 at 200 stars and 20 nights): transit true
positives 33 to 33 (32 bounded and 1 touched, both before and after), false positives 8 to 8,
eclipses reported as transits 5 to 5, pulsations reported as transits 12 to 0 (one pulsator, seed 75
star 181, ratio 1.46), detections 62 to 50. Planet hosts there read at most 0.57. **The
planet-free bar is not met on the held-out planet-free runs** (seeds 81-84, 20 nights): detections
0, 6, 0, 0 at ``main`` and on the branch. pf-seed82 gives 6: two flare stars, star 136 with 4 events at
local S/N 7.5 and star 184 with 2 at 9.1, both above the flare bar. It was not re-tuned (a flare bar
above 7.5 would have removed star 136 and cost seed 12 star 32 on the development runs); issue #43
and the Flare tails gate below met it. Seeds 21-26
(used once before) and 7 and 42: ``bls`` detections identical to ``main``, span for span. ``single_transit``,
which reads ``Candidate.rejected``, reports the same detections span for span and score for score on
33 runs (the 4 of seeds 7 and 42, seeds 21-26, 71-76, the planet-free runs 41-44 and 51-58, and the
five development runs with a planet-free pulsator). The time of ``bls`` on one 20-night run is the
the same (about 92 s on seed 26 at ``main`` and on the branch).

``scripts/bls_planet_free_audit.py`` prints the ratio, the flare share and the verdict of every
star a run would report.

Flare tails
-----------
Flare stars were what the planet-free rule still failed on (issue #43): the bar of 7.2 could not
separate flare residue from small planets because the two overlap in S/N. What a claim on a flare
star sits on was measured on the development runs (the 9 claims on flare stars on the planet-free
runs, 36 on the others and the 10 true positives on flare hosts; the table is in the issue's comment):

* **The box is a stretch free of flare residue; the residue is in the rest of the event nights.**
  With the injected flare tails at the kept samples (truth, as a measurement), the mean residue in
  the box is 0.01 sigma and in the other samples of the event nights 0.35 sigma (the median of those
  is 0.01 sigma): a minority of samples, the tails the clip's growth stops short of (a running mean
  within a standard error of the night; the exponential goes on below that), lifts the mean the
  contrast is taken against. The residue's own S/N in the box is 77 percent of its local S/N in the
  middle of the 45 claims (38 to 108 percent), 3 percent for the true positives (0 to 74). 42 percent
  of it is the leftover tail of a flare whose peak the clip seeded, the rest flares it never seeded
  (under 4 sigma, or a peak in a gap).
* **The box does not sit on a decay.** 31 of 45 boxes have a smaller share of their samples within
  1 h after a clipped sample than the rest of the star, 18 of 45 within 3 h, so a veto on boxes that
  start within some hours after a flare does not describe them. **Nor is it asymmetric:** the level
  just before and just after the box differ by 0.10 sigma in the middle of the claims (-0.04 for true
  positives), the two halves of the box by 0.05 (0.03), and the shape statistic is negative in 39 of
  45 (a flat-bottomed dip, as a transit is).

The gate (``flare_window_h`` = 3 hours, ``flare_min_run`` = 10; ``flare_window_h=None`` switches it
off) leaves out of the contrast the samples a flare may still lift. A *flare-sized run* is 10 or more
consecutive samples masked as positive outliers within a night (:func:`flare_windows`); its window
runs from its first masked sample to 3 hours after its last, and ``Candidate.snr_clean`` is the local
S/N of the same box with every sample in a window left out of both sides of the contrast
(``snr_local`` when there is no window; NaN when no night keeps both box and reference, which is
refused). The box is rejected as ``"flare tails"`` when ``snr_clean`` is under ``snr_threshold``
(7.0: not the flare bar). It runs last, after the pulsation and shape guards, so a pulsator or a
binary is still named by its own and ``single_transit`` (which reads whether a rejection is the
pulsation guard) sees no change. It removes claims and adds none, and a star with no flare-sized run
is untouched (``snr_clean == snr_local``).

The constants. *10 samples*: the clip masks a cosmic ray's 5 samples and, when noise keeps the
running mean above its threshold, more: 4.5 to 6 percent of 400 simulated spikes reach 10 (10 to 12
percent reach 8, 1.5 to 2 percent reach 12), whereas a 6 sigma flare with a 4 minute decay reaches 10 in 92
percent of 400 and one with a 10 minute decay in all of them. *3 hours*: 6 e-foldings of the longest
decay the world makes (30 minutes), where an exponential is under 0.25 percent of its peak. Excluding
instead the true window, ``k`` times each flare's true decay after its true onset (an oracle, a
measurement), leaves 8, 3, 1 and 0 of the 45 claims at S/N 7 or more for ``k`` = 2, 3, 4 and 6, and
costs 2, 2, 4 and 3 of the 10 true positives; with the detector's kind of windows (runs of 6 or
more samples, a fixed span, estimated after the fact) 16, 11, 9, 7 and 7 of the 45 survive at 1, 2,
3, 4 and 5 hours. The 3 hours is the principled
number, not the minimum of that count.

Measured, ``main`` to the branch (constants fixed on the development runs; the held-out seeds were
simulated and scored once, afterwards). Development, the 64 stored runs (seeds 7 and 42 at 3 and 20
nights; 11-14, 21-26, 31-50, 61-66 and 71-76; planet-free 41-44, 51-58 and 81-84 at 20 nights and
41-43 at 3): transit true positives 350 to 347 (344 bounded and 3 touched), false positives (all
kinds) 81 to 50, eclipses reported as transits 15 to 15, pulsations reported as transits 3 to 3,
flares reported as transits 32 to 6, detections 546 to 442; claims on flare stars 55 stars to 17
(159 detections to 59). More development runs (91-100 and 111-114, planet-free 101-104, used by
PR #42's review; 18 runs): true positives 187 to 187, false positives 21 to 15, eclipses 18 to 18,
pulsations 2 to 2, flares 5 to 0, detections 248 to 230. Planet-free 20-night runs, detections per
run: seeds 41-44 0, 4, 2, 0 to 0, 0, 2, 0; 51-58 2, 2, 0, 0, 2, 2, 3, 5 to 0, 0, 0, 0, 0, 2, 0, 2;
81-84 0, 6, 0, 0 to 0, 2, 0, 0; 101-104 0, 2, 3, 3 to 0, 0, 0, 3 (the 3-night runs, 0, 2, 0, are
unchanged). Held-out, scored once (seeds 131-136 at 200 stars and 20 nights; planet-free 141-144):
transit true positives 29 to 29, false positives 7 to 3, eclipses reported as transits 0 to 0,
pulsations reported as transits 0 to 0, flares reported as transits 5 to 0, detections 48 to 35;
planet-free detections 2, 0, 0, 0 to 0, 0, 0, 0. **Three development true positives are lost**:
seed 12 star 32 and seed 62 star 38 lose to flare windows over the box (the box is a flare
star's: P 6.680 d, five times the planet's 1.336 d, one of its two events on one of the planet's six;
and P 7.500 d against a planet with one event, which one of its two events overlaps; residue S/N 5.4
of local 7.3 and 4.5 of 7.8; an oracle that removes the flares loses both too: 116 of the 180
in-box samples of 12/32 and 148 of 542 of 62/38 are in a window), and seed 61 star 56 (a constant
star, local S/N 7.04) loses to a window opened by an 11-sample cosmic-ray run (night 5; a hot pixel
and a flare on night 13), not to a flare. All three, and the never-seen seed 261 star 29 (below,
lost to a cosmic-ray run that removes 152 reference samples), are overlap credit at periods
unrelated to the planet's, not planets found: the box's period is 5.00, 0.515, 1.69 and 0.239 times
the planet's. The other 8
true-positive flare hosts keep every grade (seed 12 star 37, seed 32 star 103, seed 40 star 166, seed
42 star 24, seed 47 star 93, seed 64 star 172, seed 73 star 131, seed 75 star 51), and the sets of true
positives of the 14 more development and the 6 held-out planet runs are unchanged. ``single_transit`` reports the same
detections span for span and score for score on 27 runs (seeds 7 and 42 at 3 and 20 nights, 11, 12,
21-26, 61, 62, 94, 96, 114 and the planet-free runs 42, 43, 51, 52, 55-58, 82 and 104); the time
of ``bls`` on a 20-night run is the same (109.2 s against 109.5 s on seed 26, 111.9 against 112.1 on
seed 93, user CPU).

**Never seen** (planets 261-266 and planet-free 271-274, 200 stars and 20 nights, simulated after
every constant was fixed; review of PR #47), ``main`` to the branch: transit true positives 78 to
77, false positives 5 to 1; planet-free detections 25 to 12, false positives 7 to 2, flares
reported as transits 8 to 4 (eclipses reported as transits 3 to 3 and pulsations 1 to 1 are
unchanged); detections per planet-free run 11, 6, 6, 2 to 5, 3, 2, 2 (at most 5 is met, 271
exactly at the limit; what survives there is a flare star, star 13, local S/N 9.43, clean 9.04).
The one true positive lost is seed 261 star 29: a constant star, a 4.529 d box on the planet's
single event (0.239 times its period), local S/N 7.04. No in-box sample is in a window and the
nightly contrasts barely move (0.0099 to 0.0098), but the 3 hours after one cosmic-ray run on the
event night remove 152 reference samples (that night's weight 104 to 71) and the clean S/N is 5.88.

**What opens a window.** On the five never-seen planet runs (1,000 stars) there are 2,142
flare-sized runs: 40 percent have a flare under them, 39 percent only a hot pixel and 20 percent
only a cosmic ray; 800 of the 1,000 stars carry at least one window. A hot pixel is a flat plateau
that masks any number of samples from 10 up, and a cosmic ray reaches 10 by chance in 4.5 to 6
percent, so ten consecutive clipped samples detect a clip run, not a flare. The gate therefore
taxes lost reference samples on marginal boxes: a box with a local S/N under about 7.5 on a night
with a window can fall under 7.0 on weight alone, with no flare residue under it, and that is
most stars, not only flare stars. A run definition with a decay shape (a monotone fall of the
masked run) would separate a flare from a plateau; it is issue #48.

**The flare bar is not redundant.** With ``flare_snr_threshold`` at 7.0 the 64 stored runs are
unchanged, but seed 96 star 10 (flare share 9.7 percent, local S/N 7.13, clean S/N 7.93) is claimed
again, so the bar stays beside the gate.

Tried and left: leaving the tails out of the *search* (tail growth to half a standard error, to a 15
sample mean, a mask of the run's length after it) removed about half of the flare claims and added
others elsewhere; a matched filter for a step with an exponential decay (subtracted, or masked) took
the planet-free runs to zero and fired on 40 percent of the nights of a constant star and on the
egress of transits, losing 12 to 15 development true positives; the night-shuffled null of the
best box (is the claim above what the star's own nights give at random phase) is as high for real
planets as for flare residue, since a transit is a dip in every night wherever it is put.

``scripts/bls_planet_free_audit.py`` prints the clean S/N beside the ratio and the flare share.

Shape guard
-----------
An eclipsing binary seen on few nights passes the pulsation guard, and a box fit to its V sits on
its core. At ``main`` (before this gate), 27 development runs (the stored seed 7 and 42 runs at 3
and 20 nights, seeds 11-14 and 31-50 at 20 nights; issue #36) reported 40 detections of eclipses as
transits, on 8 of the 428 binaries in them. The pulsation guard was nowhere near rejecting them:
twice the best sine's power over the box's chi-square was 0.09 to 0.38 (it rejects at 1) and the
residual's power over the false-alarm level 0.37 to 0.90 (it rejects at 1). The box was 1 to 3 h
wide on eclipses of 1.5 to 13.5 h. On 5 of the 8 every reported event was the same eclipse (the core
of the other was never inside a predicted window; only its wings reached a night), so an odd-even
depth test or a half-phase secondary search had nothing to compare there: a per-night odd-even
contrast at ``|z| >= 3`` would have removed 2 of the 8 stars (20 of the 40 detections). What all 8
share is a V fit by a box, so the last gate is a shape test (:meth:`BlsDetector._shape_excess`):

* The samples of the nights that hold a box event are put at their offset ``x`` from the nearest
  event centre, and the dip ``1 - flux`` is fitted with two families of templates, each with a free
  depth and a free level for every night: boxes (15 widths in a geometric ladder from the shortest
  to the longest of ``durations_h``) and triangles (``max(0, 1 - 2|x| / w)``, 23 widths from the
  shortest duration to 4 times the longest, none wider than the period).
* Each night takes its own centre within 0.5 h of the box, in steps of a tenth of the template's
  width, for both families. The phase bins are 15 minutes wide and the period grid's step lets the
  last events sit up to a third of the shortest duration (10 minutes) off, which a 0.8 h flat dip
  cannot forgive: with one centre for all nights planted 0.8 h transits read as a V at 1.4 and 2.4,
  against the threshold of 1.5.
* With ``g_box`` and ``g_triangle`` the largest chi-square gains of the two families (in flux units,
  ``amplitude**2 * template.template``), the statistic is ``(g_triangle - g_box) / (sigma *
  sqrt(g_box))``: about ``(chi2 of the triangle - chi2 of the box) / S/N``, so it does not grow with
  the signal. A box-shaped dip makes it negative, a V positive. The candidate is rejected
  (``rejected == "shape"``) when it reaches ``shape_threshold`` (1.5; ``None`` switches the gate
  off), and ``Candidate.shape`` holds it. The gate runs last, so a pulsator is still named by the
  pulsation guard. It costs 8 to 17 ms for each candidate that reaches it (11 of them on one
  20-night run, 0.1 percent of its CPU time).

Measured. The development runs and the planted cases below fixed the form of the statistic and the
1.5; the held-out seeds were simulated afterwards. Of the 109 stars ``main`` reported on the
development runs, six of the eight binaries read 7.5, 6.6, 4.2, 3.7, 1.8 and 1.7 and the two 3-night
ones 0.13 (seed 7 star 143: a 12.7 h eclipse that fills the 8 h nights) and -1.29 (seed 42 star 0: a
2.4 h eclipse, 50 and 15 percent deep, that does not fill them; its two event nights read 3.72 and 4.11
when each is fitted alone, so it is the joint fit that loses it, one depth and one width for both
nights, and why is not established); of the other 101 stars the highest are 1.29 (a
flare star's false positive), 1.12 (a pulsating star reported as a transit, 20 events), 0.97 and
0.96 (false positives) and 0.93 (a planet: the highest of the 66 planets, the next 0.46), and the 9
candidates found by detrending a pulsation read -4.6 to -0.4. The branch against ``main`` on those
27 runs: transit true positives 202 to 202, false positives (all kinds) 56 to 56, eclipses reported
as transits 40 to 3 (the three are on the 3-night runs), pulsations reported as transits 52 to 52.
Planted, 20 nights: of 300 planets drawn as the world draws them (ramps of 10 percent, 0.5 to 6 h,
depth 0.5 to 3 percent, noise 0.3 to 2 percent) 172 reach the gate and the highest statistic is 1.04
(three at 0.5 or more); of 600 binaries drawn as the world draws them only 3 reach it, at 0.62 (a 13
h eclipse at S/N 19 that fills the nights), 3.6 and 4.5. Held-out, scored once (seeds 21-26, 200
stars, 20 nights; they are the seeds on which #36 found the leak, so they are held out from every
threshold but not from the choice of the problem): transit true positives 46 to 46, false positives
6 to 6, eclipses reported as transits 13 to 0 (four binaries, nothing else changed), pulsations
reported as transits 0 to 0. ``single_transit``, which reads ``Candidate.rejected``, reports the
same detections span for span and score for score on 4 development runs (seeds 7 and 42, 3 and
20 nights) and the 6 held-out ones.

Never-seen seeds 61-66 (200 stars, 20 nights; reviewed after the numbers above were fixed): transit
true positives 69 to 69, false positives 18 to 18, eclipses reported as transits 26 to 5, pulsations
reported as transits 29 to 29 (planet-free sinusoidal stars, which the gate does not touch). Of the 36
stars either version reports, the planet hosts read at most 0.12 (23 of them), the flare stars -0.79
to -3.09, and the four binaries 5.51, 5.09, 3.43 and 0.70. The remaining leak is seed 64 star 195, a
7.04 h eclipse in 8 h nights (88 percent of a night; 5 events, S/N 19.2, statistic 0.70). The planets
sit far from the gate on these seeds; the thin margin is on the binary side.

``scripts/bls_shape_audit.py`` prints the statistic of every star a run would report.

The centring window (``_SHAPE_SHIFT_H``, 0.5 h) is pinned on both sides by behaviour: at 0 h planets
with ramps read 1.95 and 2.75 (development seeds 40 and 33), and with a 2 h window a 2 h transit
beside a shallow V that reads 2.3 to 2.9 at 0.5 h reads 0.3 to 0.9 (both in ``tests/test_bls.py``).
The 0.5 h itself is a choice with a stated rationale (the phase bins and the period grid), not a
measured optimum: on 11 audited runs a 2 h window kept every binary above 1.5 and every planet below
0.54. The triangle's cap at the period binds only for periods of 0.5 to 1 d (4 x 6 h = 24 h) and
changes the statistic of a pure V train by under 3 percent (P 0.7 d: 13.97 against 14.36); a
tuned test pins it, no planted planet reaches it.

Limits of the gate: a V that fills most of every night is not told from a box (seed 7's 3-night
binary above, and the unseen 7 h eclipse in 8 h nights), and seed 42's 3-night binary, whose eclipse
does not fill the nights, is lost for a reason not established; the margin is thin at both ends (the
eclipses at 1.7 and 1.8 and a planet at 0.93, against 1.5; a planted "both eclipses seen" binary
reads 1.6 and 1.9); an eclipse wider than the nights has little curvature left to read. A shape-
rejected star plans the second pass's mask like any other rejection and its re-search refuses the
first period and its halves and double like a pulsation rejection (``second_pass`` was not measured
with the gate).

Pulsating hosts
---------------
The guard rejects a whole star: a sinusoid of 0.1-2 percent stands out of the residual after
the box is subtracted, so a planet on a pulsating star went with it (5 of the 14 strong planets
on held-out seeds 21-26, issue #35). ``detrend_pulsation`` (on by default; needs the guard) searches
such a star again with its pulsation subtracted. It is a *subtract-then-search*, not a joint
fit, in three steps, and only for a star whose best box was rejected by the pulsation guard:

1. **Fit.** The best sinusoid in the guard band (1-30 cycles/day) with a free intercept for
   every night (:meth:`BlsDetector._pulsation_trend`): the six strongest peaks of the Fourier
   power are placed by least squares (:func:`_refine_frequency`) and the largest chi-square
   gain wins (the Fourier power only ranks the peaks: the nights' intercepts take most of a
   sinusoid near a night's length, 62 percent of a 0.9 day one, so the strongest peak can be
   an alias one cycle/day away and the least-squares fit of the six is what places it). The
   noise is the scatter of within-night first differences, which a slow pulsation does not
   inflate, not the guard's. The sinusoid is used only if its power is at least
   ``trend_min_ratio`` (3) times the guard's false-alarm level.
2. **Subtract and search.** Subtract it, divide the nights by their median again, and search.
3. **Refit around the box.** When that search finds a box at the S/N gate, fit the sinusoid
   again with the box and the window 1.5 times as wide left out, and search once more. At a
   fixed frequency this is the sinusoid of the joint fit of sinusoid and box (the box takes the
   in-box samples), so the joint solution is reached without a joint search over every period,
   duration and phase; a 1 percent, 3 h transit left in the first fit pulled the sinusoid until
   the residual failed the guard (power 43 against the level 13) and no planet came out. Leaving
   out the *rejected* box instead removes most of the record (that box is the fold of the
   pulsation).

The result is a candidate in its own right: every gate applies (the guard on the detrended
flux), and ``snr`` and ``snr_local`` must reach ``trend_snr_threshold`` (9), because a detrended
star is a second look at a star the first look rejected, and a second look is a second chance
at a false alarm. A star that comes out rejected keeps its original candidate and is rejected as
before. Detections still carry ``kind_guess="transit"``; ``Candidate.trend`` is the
``(cycles/day, amplitude)`` removed and the INFO log has one line for each such star.

Three failures designed against, each with a test: a sinusoid slower than the guard band (a 3 day
one is a ramp in every night, folds into the 1.33 cycles/day alias, and subtracting that leaves
the other alias as a dip in every night: S/N 14, 20 events) is detected by trying 0.1-0.99
cycles/day and refused when one of them fits better; an eclipsing binary has harmonics that are
sinusoids, so a single sinusoid is subtracted, not a series (with three, an eclipsing binary of
5 percent was accepted once); and flare stars, whose red residue gives a peak 1-2 times the
white-noise level, are below ``trend_min_ratio`` (every weak real pulsator on the development
runs is at 4 or more).

Measured on the 24 development runs (the stored seed 7 and 42 runs at 3 and 20 nights, seeds 31-50
at 20 nights; these chose ``trend_min_ratio``, ``trend_snr_threshold``, one sinusoid and the
refit): 7 more transit true positives (all events of three strong planets on pulsating hosts, seed
31 star 32, seed 39 star 194, seed 46 star 25, and one event of a candidate on seed 44 star 44),
no transit true positive lost, no change in false positives or in eclipses reported as transits,
one more pulsation reported as a transit (seed 44, the candidate's other event). 300 synthetic
pure pulsators (0.1-2 percent, 1-12 h, noise 0.3-0.8 percent, 20 nights) gave no detection. Held-
out, scored once after every threshold was fixed (seeds 21-26, 200 stars, 20 nights; they were not
used for any choice): transit true positives 30 to 46, false positives 6 to 6, eclipses reported
as transits 13 to 13, pulsations reported as transits 0 to 0, 1.26 times the CPU time (1.24 times
the wall clock). The five strong planets on pulsating hosts that the guard lost: three now have
every event bounded or touched (seed 21 star 85, seed 22 star 6: 9 bounded and 1 touched of 10,
seed 24 star 37), seed 21 star 83 has one of
its two events bounded, and seed 23 star 139 is still lost (its detrended box, at 3.177 d the
planet's 12.731 d over four, has S/N 9.7 but local S/N 7.4, below ``trend_snr_threshold``).

Limits: one sinusoid per star (a multi-mode pulsator is not removed); sinusoids slower than 1
cycle/day are not touched (a planet on such a host stays rejected); the sinusoid is a global one,
so an amplitude that changes over the record is only partly removed; ``trend_snr_threshold`` costs
a planet on a pulsating star whose box reaches S/N 7-9; a pulsator with one real transit can borrow
a 1 h noise dip as the second event that ``min_events`` asks for, so the candidate has one true and
one spurious event (dev seed 44 star 44: P 5.562 d, S/N 13.3, a bounded transit and a touched noise
dip; about 1 in 120 such stars on 240 synthetic pulsators with one 3 h transit, never a detection
without the real transit; on ``main`` these stars give nothing).

Second pass
-----------
``second_pass`` (**off by default**) searches a star once more when its best box is rejected by
any gate except the S/N gate (local S/N, events, support, pulsation guard, shape guard); a box too
weak for the S/N gate means no other box on the star is strong either. Without it a star's
strongest box decides the star: an eclipsing binary at S/N 361 rejected by the guard hid a planet
of S/N 39 (seed7-200s-20n star 138), and a one-night dip that wins the search hides what repeats.

Measured, and why it is opt-in. On the stored 20-night runs the second pass recovers seed 7
star 138 (a planet on an eclipsing binary, both transits bounded; the reported period is the
sub-harmonic 1.145 d = 8.012/7 d, and the period never reaches a :class:`Detection`) and adds
one false positive on seed 42 (star 94, a flare star); the 3-night rows are unchanged. On the
held-out seeds 21-26 (200 stars, 20 nights) it added no true transit, 3 false positives and
1 eclipse mistaken for a transit, at 1.22 times the single-pass time, so it is off. The first
pass alone, on the same six seeds, scored 30 transit true positives, 6 false positives, 13
eclipses reported as transits and no pulsator reported as a transit (before the pulsation
detrending and the shape guard); 5 of the 14 strong
planets there sit on pulsating hosts and are rejected by the pulsation guard, which the
re-search does not recover (its mask has nothing to take out of a sinusoid).

The re-search leaves out the *dip* around the rejected box, not only the box. A box is at most
6 h wide and the eclipse of a binary can last 11 h; masking the box's windows and one duration
each side left the wings of every eclipse, and the re-search found them (star 138: period
2.000 d, S/N 16, rejected by the guard again). So the record is folded on the box's period
(:meth:`BlsDetector._dip_mask`) in phase bins of half the shortest duration, using flux in the
star's own units (nights divided by the star's median night level rather than their own median,
so an eclipse that fills most of a night keeps its depth). A bin is depressed when its mean is
below 1 by ``support_snr`` standard errors, and a *dip* is a run of bins depressed by
``tail_sigma`` standard errors containing such a bin (the flare-tail growth of step 3, run
downwards in phase). The mask is:

* the primary: the box's bins and every dip touching them;
* the secondary: a dip centred half a period from the primary that reaches ``snr_threshold``
  on its own and lasts at least 0.7 of the primary (a circular binary's two eclipses last
  equally long). A shorter dip there is not masked: on star 138 the box sits on half the
  binary's period, so half a period away is an arbitrary phase, and masking a planet's
  transit there cost the recovery; with a weak secondary the box sits on the binary's own
  period and, unmasked, the secondary's residue replaced the planet as the second candidate;
* each run widened by one box duration on each side.

The masked samples count as not observed: they leave the search, the statistics, the guard and
the event list, and each night is divided by the median of what is left. The result is a
candidate in its own right and must pass every gate; when the first box was rejected by the
pulsation guard or the shape guard it is also refused (``rejected == "alias"``) if its period is
the first period or half or twice it to 2 percent, where the residue of a binary or a pulsator
repeats (the period of a box refused for lack of events is arbitrary and no alias is applied). One
re-search at most: a rejected second candidate ends the star. The outcome is logged at INFO
(``bls star 138: re-search found P 1.145 d, S/N 38.6 (local 29.3): accepted``).

Limits
------
* Only the best box per star is reported, and with ``second_pass`` one more after a rejection; a
  third signal on a star is out of scope. The strongest peak at a clearly different period that
  is not a small-integer ratio of the best period is logged at INFO (``skywatch.detectors.bls``)
  as an alternative. Without the second pass a star whose strongest box is not a planet (an
  eclipsing binary; a pulsator unless its pulsation is subtracted, see above) is rejected whole,
  so a planet behind it is lost too; with
  ``second_pass=True`` the planet is found unless its events sit inside the masked dip, the
  binary's eclipses fill most of every night (the re-search then has no clean night), or the
  re-search's best box is not the planet (a flare star's residue passes the gates on
  seed42-200s-20n star 94; on held-out seed 26 star 167 a binary's secondary eclipse, clipped by
  the night edges, is narrower than 0.7 of the primary and its residue is accepted). Of the four
  re-search acceptances over ten 20-night runs one was a planet. The period of a recovered box
  can be a sub-multiple of the planet's (the other windows fall in the day gaps) while its
  events are the planet's.
* A transit longer than the 6 h grid maximum is fit by the closest box; a transit that
  covers more than half of a night makes that night's first-pass median the in-transit
  level (the out-of-box renormalisation repairs the depth, not the search).
* Two single events of different planets six days apart look like one period.
* Output spans are half-open: ``t_end = t_last + cadence_s``.
* Flare stars still make claims: 17 stars of the 55 that claimed on the 64 stored development runs
  (8 of them true positives, the other 9 with 2 to 5 events at clean S/N 7.3 to 10.9, two of them
  on planet-free runs, seeds 58 and 82). They are flares the clip never seeded (an oracle that excludes 6 e-foldings
  of every true flare leaves none), and the gate sees only flare-sized runs: pf-seed82 star 184
  (0.5 h box, 2 events, local S/N 9.1, clean 8.5) has residue S/N 7.0 of which 0.1 is a seeded flare.
  The bar of 7.2 is kept for the one star the gate does not hold (see Flare tails). The gate costs
  a planet whose events follow a flare by less than 3 hours (3 of 537 development true
  positives);
  on a star with a flare every night the clean S/N is about 60 percent of the local one (a planted
  planet: 22 against 37); most windows are not flares' (a hot pixel's plateau opens one, and a
  cosmic ray whose run reaches 10 samples, 5 percent of them; see What opens a window), so a
  marginal box on a night with a window can fall under 7.0 on the weight of the reference it
  loses, with no flare residue (issue #48); the window is a fixed 3 hours and not scaled to
  each flare.
* The sinusoid-against-box ratio has margin on the pulsator side (0.97 and above) and less than
  it looks on the planet side: the bar is not a floor under the planets. Two real planets read
  0.73 (seed 38 star 168; seed 23 star 114) and the planet hosts of never-seen seeds at most 0.52.
  A Monte Carlo of the simulator's planet recipe at periods of 0.5 to 1.5 d (320 draws, 20
  nights) lost 1 of 279 planets that ``main`` accepts: P 0.562 d, 4.84 h, ratio 0.847; another
  accepted planet read 0.797 (P 0.75 d, 4.5 h). That is roughly 1 planet in 5000 of the
  population. The ratio depends on how the period sits against the night (a planted 3 h planet at
  0.51 d reads 0.54 at one epoch; a planted 0.6 d, 3.5 h planet of 1.2 percent in 0.5 percent
  noise reads 0.826 and is refused, the same at 4 h reads 0.74 and is accepted); a short-period
  planet with a long box is the one that approaches the bar.
  A pulsator weaker than the S/N bar never reaches the guard at all.
* Fast pulsators (15 to 20 cycles/day) that the detrending subtracts leave a box of 2 events
  after the subtraction in 2 of 2 cases measured (seed 13 star 130; seed 112 star 62, ratio 0.10):
  tracked in issue #44.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from skywatch.detectors import register
from skywatch.detectors.nights import split_nights
from skywatch.types import Detection, LightCurve

log = logging.getLogger(__name__)

DAY_S = 86400.0
HOUR_S = 3600.0
DEFAULT_DURATIONS_H = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MAD_SCALE = 1.4826
_GUARD_CORE_WIDTH = 0.8  # the pulsation guard subtracts the box over 0.8 x its duration
_GUARD_MASK_WIDTH = 1.5  # ... and leaves out the rest of the 1.5 x window
_SECONDARY_WIDTH = 0.7  # a secondary dip must last this share of the primary's length
_ALIAS_TOLERANCE = 0.02  # a re-search period within this fraction of P, P/2 or 2P is an alias
_DIP_MIN_BINS = 8  # the dip fold has at least this many phase bins, whatever the period
_DIP_CORE_MIN_BINS = 0.5  # half-width floor (in bins) of the box core in the dip fold
_RENORMALISE_MIN_SAMPLES = 5  # a night of fewer samples is not divided by its median again
_TREND_SLOW_F_MIN = 0.1  # cycles/day: the slowest sinusoid tried when asking if one is slower
_TREND_SLOW_STEP = 0.01  # ... and the step between the frequencies tried
_TREND_PEAKS = 6  # Fourier peaks placed by least squares, the best of which is the sinusoid
_TREND_REFINE_STEPS = 14  # golden-section steps that place a pulsation's frequency
_SHAPE_BOX_STEPS = 15  # box widths tried by the shape test: a geometric ladder of this many
_SHAPE_TRIANGLE_STEPS = 23  # ... and triangle widths, from the shortest box to 4x the longest
_SHAPE_TRIANGLE_SPAN = 4.0
_SHAPE_SHIFT_H = 0.5  # both shapes are centred within this many hours of the box ...
_SHAPE_SHIFT_STEP = 0.1  # ... in steps of this fraction of their width
_SHAPE_MIN_SAMPLES = 10  # fewer samples on the box's nights than this: no verdict
_CHUNK_ELEMENTS = 1_000_000  # periods per chunk = this // n samples (keeps arrays cache-sized)


@dataclass(frozen=True)
class _Clean:
    """One star's quality-clean samples: times, night-median-normalised flux, night index."""

    t: np.ndarray
    flux: np.ndarray
    night_of: np.ndarray
    level: np.ndarray  # the night median each sample's flux was divided by


@dataclass(frozen=True)
class _Arrays:
    """What one search works on: the prepared samples and the observed ones (masks applied)."""

    t: np.ndarray
    flux: np.ndarray
    night: np.ndarray
    level: np.ndarray
    t_obs: np.ndarray
    night_obs: np.ndarray
    flare_windows: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    # [start, end] (s) of the spans after each flare-sized masked run (see :func:`flare_windows`)
    flare_fraction: float = 0.0  # share of the star's clean samples masked as positive outliers


def positive_outliers(x: np.ndarray, clip_sigma: float, tail_sigma: float) -> np.ndarray:
    """Mask of positive outliers in one night of flux normalised to ~1, tails included.

    ``s = 1.4826 * MAD(x)``. Seeds are samples above ``1 + clip_sigma * s``. A flare's decay
    stays above the noise long after its peak has crossed the seed threshold, and the
    mean the box statistic works with is moved by that mass, so each seed is grown into
    the contiguous run where the 5-sample running mean stays above
    ``1 + tail_sigma * s / sqrt(5)`` (``tail_sigma`` standard errors of that mean).
    Nothing negative is ever masked.
    """
    scale = _MAD_SCALE * float(np.median(np.abs(x - np.median(x))))
    if not scale > 0:
        return np.zeros(len(x), dtype=bool)
    seed = x > 1.0 + clip_sigma * scale
    if not seed.any():
        return seed
    width = 5
    pad = np.concatenate((np.full(width // 2, 1.0), x, np.full(width // 2, 1.0)))
    mean = np.convolve(pad, np.ones(width) / width, mode="valid")
    cand = seed | (mean > 1.0 + tail_sigma * scale / math.sqrt(width))
    run = np.cumsum(np.diff(np.concatenate(([False], cand))) & cand)
    has_seed = np.bincount(run[cand], weights=seed[cand]) > 0
    return cand & has_seed[run]


def flare_windows(
    t: np.ndarray, night: np.ndarray, kept: np.ndarray, min_run: int, window_s: float
) -> np.ndarray:
    """``(m, 2)`` array of ``[start, end]`` (seconds): the span of each flare-sized masked run.

    `t` and `night` are the star's clean samples and `kept` says which of them survived the
    one-sided clip of :func:`positive_outliers`. A *flare-sized* run is `min_run` or more
    consecutive masked samples within a night (a cosmic ray masks 5, see ``flare_min_run``). The
    span runs from the first masked sample to `window_s` after the last, where the flare's decay
    left what the clip's tail growth stops at (a running mean within about half a sigma of the
    night) and the rest of the exponential is still above the noise for a few more e-foldings.
    """
    out = []
    for k in np.unique(night):
        idx = np.flatnonzero(night == k)
        masked = ~kept[idx]
        if not masked.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], masked.astype(np.int8), [0]))))
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if b - a >= min_run:
                out.append((float(t[idx[a]]), float(t[idx[b - 1]]) + window_s))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def in_windows(t: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Boolean mask of the samples of `t` inside any ``[start, end]`` of `windows`."""
    hit = np.zeros(len(t), dtype=bool)
    for start, end in windows:
        hit |= (t >= start) & (t <= end)
    return hit


def _night_medians(flux: np.ndarray, night_of: np.ndarray) -> np.ndarray:
    """Per-sample median of the sample's night (1.0 for a night whose median is not positive)."""
    out = np.ones(len(flux))
    for k in np.unique(night_of):
        m = night_of == k
        med = float(np.median(flux[m]))
        if med > 0:
            out[m] = med
    return out


def _divide_by_night_median(flux: np.ndarray, night_of: np.ndarray) -> np.ndarray:
    out = np.array(flux, dtype=np.float64)
    for k in range(int(night_of[-1]) + 1):
        m = night_of == k
        med = float(np.median(out[m]))
        out[m] = out[m] / med if med > 0 else 1.0
    return out


@dataclass(frozen=True)
class Box:
    """The best box found for one star (all times in seconds on the global axis)."""

    period_s: float
    epoch_s: float  # mid-time of one event, in [0, period_s)
    duration_s: float  # the evaluated width: round(duration / bin) * bin
    depth: float
    n_in: int
    n_total: int
    power: float  # depth**2 * n_in * n_out / n_total

    def snr(self, sigma: float) -> float:
        """Matched-filter S/N of the box: ``depth * sqrt(n_in) / sigma``."""
        return self.depth * math.sqrt(self.n_in) / sigma


@dataclass(frozen=True)
class Candidate:
    """A star's best box and what became of it.

    `rejected` is None for a reportable candidate, else the first gate that failed:
    ``"snr"``, ``"local snr"``, ``"min_events"``, ``"support"``, ``"pulsation guard"``,
    ``"shape"`` or ``"flare tails"`` (``"alias"`` for a re-search that found the first box's
    period again).
    `events` are the predicted in-night windows ``(t_start, t_end, night)`` (empty once
    rejected); `excess` is the guard's strongest residual-sinusoid power (None if not
    computed); `alternative` is ``(period_s, power / best power)`` of the strongest
    unrelated peak; `mask` the windows a re-search leaves out (only when asked for, and
    only for a rejection other than ``"snr"``); `trend` is ``(cycles/day, amplitude)`` of the
    sinusoid that was subtracted before this box was found (None when none was); `shape` is
    the shape test's statistic (:meth:`BlsDetector._shape_excess`; None when it was not run);
    `sine_ratio` is the chi-square gain of the best sinusoid over the box's with a free level
    for every night (:meth:`BlsDetector._night_sine_ratio`; None when it was not computed);
    `snr_clean` is ``snr_local`` with the samples inside and within ``flare_window_h`` after each
    flare-sized masked run left out of the contrast (:func:`flare_windows`; equal to
    ``snr_local`` when there is no such run, NaN when no night keeps both box and reference).
    """

    box: Box
    sigma: float
    snr: float
    events: list[tuple[int, int, int]]
    rejected: str | None
    excess: float | None
    alternative: tuple[float, float] | None
    snr_local: float = 0.0
    supported_nights: int = 0
    mask: tuple[Box, ...] = ()
    trend: tuple[float, float] | None = None
    shape: float | None = None
    sine_ratio: float | None = None
    snr_clean: float = math.nan


def frequency_grid(
    baseline_s: float,
    min_period_s: float,
    max_period_s: float,
    min_duration_s: float,
    oversample: float,
) -> np.ndarray:
    """Trial periods in seconds, ascending, uniform in log(frequency)."""
    step = (min_duration_s / baseline_s) / oversample
    n = math.ceil(math.log(max_period_s / min_period_s) / step) + 1
    return np.minimum(min_period_s * np.exp(step * np.arange(n)), max_period_s)


def presample(
    t: np.ndarray, flux: np.ndarray, bucket_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge samples falling in the same `bucket_s` time bucket: (mean t, flux sum, count).

    Sorted `t` in, sorted out. The fold only ever needs sums and counts per phase bin, so a
    bucket narrower than the phase bin loses at most half a bucket of phase resolution
    while cutting the per-trial-period cost by the bucket size.
    """
    bucket = np.floor(t / bucket_s).astype(np.int64)
    first = np.concatenate(([0], np.flatnonzero(np.diff(bucket)) + 1))
    count = np.diff(np.concatenate((first, [len(t)]))).astype(np.float64)
    return np.add.reduceat(t, first) / count, np.add.reduceat(flux, first), count


def search_box(
    t: np.ndarray,
    flux_sum: np.ndarray,
    count: np.ndarray,
    periods_s: np.ndarray,
    durations_s: np.ndarray,
    bin_s: float,
) -> tuple[Box | None, np.ndarray]:
    """Best box (largest ``depth**2 * n_in * n_out / N``) over the period/duration/phase grid.

    Each point is a time `t` (s), the `flux_sum` of `count` merged samples (`count` of
    ones and `flux_sum` the flux itself for unmerged samples), flux normalised to ~1.
    Returns ``(box, power_by_period)``; `box` is None if no dip exists.
    ``power_by_period`` is the best power at each trial period (0 where no dip), used to
    find alternatives.
    """
    t = np.asarray(t, dtype=np.float64)
    m = len(t)
    nper = len(periods_s)
    power_by_period = np.zeros(nper)
    if m < 2 or nper == 0:
        return None, power_by_period
    total = float(flux_sum.sum())
    n = float(count.sum())
    # Per-period bin count nb (bin width P / nb <= bin_s); arrays are padded to the chunk's
    # widest period. Starts at or beyond nb repeat the wrapped boxes, so need no masking.
    nb_all = np.ceil(periods_s / bin_s).astype(np.int64)
    width = periods_s / nb_all
    chunk = max(1, _CHUNK_ELEMENTS // m)
    best_power = 0.0
    best: tuple[float, int, int, int, float, int] | None = None  # P, bin, w, nb, depth, n_in
    for lo in range(0, nper, chunk):
        hi = min(lo + chunk, nper)
        c = hi - lo
        P = periods_s[lo:hi]
        nb = nb_all[lo:hi]
        bw = width[lo:hi]
        nbmax = int(nb.max())
        x = t[None, :] * (1.0 / P)[:, None]
        x -= np.floor(x)
        x *= nb[:, None]
        b = x.astype(np.int64)
        np.minimum(b, (nb - 1)[:, None], out=b)
        b += (np.arange(c, dtype=np.int64) * nbmax)[:, None]
        flat = b.ravel()
        cnt = np.bincount(flat, weights=np.tile(count, c), minlength=c * nbmax).reshape(c, nbmax)
        sm = np.bincount(flat, weights=np.tile(flux_sum, c), minlength=c * nbmax).reshape(c, nbmax)
        w_all = np.maximum(1, np.rint(durations_s[:, None] / bw[None, :]).astype(np.int64))
        ext = np.arange(nbmax + int(w_all.max()))[None, :] % nb[:, None]
        length = ext.shape[1] + 1
        cs = np.zeros((c, length))
        cc = np.zeros((c, length))
        np.cumsum(np.take_along_axis(sm, ext, axis=1), axis=1, out=cs[:, 1:])
        np.cumsum(np.take_along_axis(cnt, ext, axis=1), axis=1, out=cc[:, 1:])
        csf, ccf = cs.ravel(), cc.ravel()
        base = (np.arange(c, dtype=np.int64) * length)[:, None] + np.arange(nbmax)[None, :]
        cs0, cc0 = csf[base], ccf[base]
        rows = np.arange(c)
        for k in range(len(durations_s)):
            end = base + w_all[k][:, None]
            n_in = ccf[end] - cc0
            # depth = (n_in * total - n * s_in) / (n_in * n_out): u > 0 is a dip
            u = n_in * total - n * (csf[end] - cs0)
            np.maximum(u, 0.0, out=u)
            n_out = n - n_in
            n_in *= n_out
            np.maximum(n_in, 1.0, out=n_in)
            power = u * u / (n * n_in)
            j = power.argmax(axis=1)
            pk = power[rows, j]
            np.maximum(power_by_period[lo:hi], pk, out=power_by_period[lo:hi])
            i = int(pk.argmax())
            if pk[i] > best_power:
                best_power = float(pk[i])
                jj = int(j[i])
                n_in_i = float(ccf[end[i, jj]] - cc0[i, jj])
                s_in_i = float(csf[end[i, jj]] - cs0[i, jj])
                depth = (total - s_in_i) / (n - n_in_i) - s_in_i / n_in_i
                best = (
                    float(P[i]),
                    jj % int(nb[i]),
                    int(w_all[k][i]),
                    int(nb[i]),
                    depth,
                    round(n_in_i),
                )
    if best is None:
        return None, power_by_period
    period, j, w, nb, depth, n_in = best
    bw_best = period / nb
    box = Box(
        period_s=period,
        epoch_s=((j + w / 2.0) * bw_best) % period,
        duration_s=w * bw_best,
        depth=depth,
        n_in=n_in,
        n_total=round(n),
        power=best_power,
    )
    return box, power_by_period


def refine(t: np.ndarray, flux: np.ndarray, box: Box) -> Box:
    """`box` with depth and n_in recomputed on the samples themselves (no bucket or bin).

    A sample is in the box when ``|phase - epoch| <= duration / 2`` (the same rule the
    event windows use); depth is the out-of-box mean minus the in-box mean.
    """
    inside = in_box(t, box)
    n_in = int(inside.sum())
    if n_in == 0 or n_in == len(t):
        return replace(box, depth=0.0, n_in=n_in, n_total=len(t))
    depth = float(flux[~inside].mean() - flux[inside].mean())
    return replace(box, depth=depth, n_in=n_in, n_total=len(t))


def night_contrasts(
    t: np.ndarray, flux: np.ndarray, night: np.ndarray, box: Box
) -> tuple[np.ndarray, np.ndarray]:
    """Per-night in-box contrast against the same night's out-of-box samples.

    Returns ``(depth_k, weight_k)`` over the nights holding both in-box and out-of-box
    samples: ``depth_k = mean_out - mean_in`` within the night and
    ``weight_k = n_in * n_out / (n_in + n_out)``, so ``depth_k * sqrt(weight_k) / sigma`` is
    that night's matched-filter S/N. No night level is assumed or estimated.
    """
    inside = in_box(t, box)
    k = np.unique(night, return_inverse=True)[1]
    nk = np.bincount(k)
    n_in = np.bincount(k, weights=inside)
    n_out = nk - n_in
    s_in = np.bincount(k, weights=np.where(inside, flux, 0.0))
    s_out = np.bincount(k, weights=np.where(inside, 0.0, flux))
    ok = (n_in > 0) & (n_out > 0)
    depth = s_out[ok] / n_out[ok] - s_in[ok] / n_in[ok]
    return depth, n_in[ok] * n_out[ok] / nk[ok]


def _renormalise_out_of_box(
    t: np.ndarray, flux: np.ndarray, night: np.ndarray, box: Box, min_out: int = 5
) -> np.ndarray:
    """Divide each night by the median of its out-of-box samples.

    The first-pass night median includes any transit in the night; a long event pulls it
    below the out-of-transit level, which would lift that whole night's out-of-box flux.
    """
    out = ~in_box(t, box)
    flux = flux.copy()
    for k in np.unique(night):
        m = night == k
        o = m & out
        if o.sum() >= min_out:
            med = float(np.median(flux[o]))
            if med > 0:
                flux[m] /= med
    return flux


def _circular_runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """``(start, length)`` of each run of True in `flag`, which wraps around its end."""
    n = len(flag)
    if flag.all():
        return [(0, n)]
    if not flag.any():
        return []
    first = int(np.flatnonzero(~flag)[0])
    rolled = np.roll(flag, -first)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rolled.astype(np.int8), [0]))))
    return [((a + first) % n, int(e - a)) for a, e in zip(edges[::2], edges[1::2], strict=True)]


def _is_alias(period_s: float, first_s: float) -> bool:
    """True when `period_s` is `first_s`, half of it or twice it (to ``_ALIAS_TOLERANCE``)."""
    return any(abs(period_s / first_s * q - 1.0) < _ALIAS_TOLERANCE for q in (1.0, 0.5, 2.0))


def _renormalise_nights(
    flux: np.ndarray, night: np.ndarray, min_n: int = _RENORMALISE_MIN_SAMPLES
) -> np.ndarray:
    """Divide each night by its own median again (nights of fewer than `min_n` are left)."""
    flux = flux.copy()
    for k in np.unique(night):
        m = night == k
        if m.sum() >= min_n:
            med = float(np.median(flux[m]))
            if med > 0:
                flux[m] /= med
    return flux


def _demean_by_night(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """`x` minus the mean of its own night (the fit's per-night intercept); `k` numbers nights."""
    return x - (np.bincount(k, weights=x) / np.bincount(k))[k]


def _sine_fit(
    t_d: np.ndarray, x: np.ndarray, k: np.ndarray, f_per_d: float
) -> tuple[float, float, float]:
    """Least-squares ``a sin + b cos`` at `f_per_d` cycles/day: ``(a, b, chi-square gain)``.

    `t_d` is in days, `x` the zero-mean flux, `k` the night number of each sample; the basis and
    `x` are both demeaned by night, so
    each night carries its own free intercept (the nights were divided by their median, which
    is not the mean of a pulsation that has not completed a cycle in the night).
    """
    arg = 2.0 * np.pi * f_per_d * t_d
    s = _demean_by_night(np.sin(arg), k)
    c = _demean_by_night(np.cos(arg), k)
    ss, cc, sc = float(s @ s), float(c @ c), float(s @ c)
    xs, xc = float(s @ x), float(c @ x)
    det = ss * cc - sc * sc
    if not det > 0:
        return 0.0, 0.0, 0.0
    a = (xs * cc - xc * sc) / det
    b = (xc * ss - xs * sc) / det
    return a, b, a * xs + b * xc


def _refine_frequency(
    t_d: np.ndarray, x: np.ndarray, k: np.ndarray, f0: float, half_width: float
) -> float:
    """Frequency in ``f0 +- half_width`` (cycles/day) maximising :func:`_sine_fit`'s gain.

    A golden-section search: the gain is unimodal within one zero-padded Fourier bin of the
    peak.
    """
    lo, hi = f0 - half_width, f0 + half_width
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    gc = _sine_fit(t_d, x, k, c)[2]
    gd = _sine_fit(t_d, x, k, d)[2]
    for _ in range(_TREND_REFINE_STEPS):
        if gc > gd:
            hi, d, gd = d, c, gc
            c = hi - inv_phi * (hi - lo)
            gc = _sine_fit(t_d, x, k, c)[2]
        else:
            lo, c, gc = c, d, gd
            d = lo + inv_phi * (hi - lo)
            gd = _sine_fit(t_d, x, k, d)[2]
    return 0.5 * (lo + hi)


def _top_peaks(f: np.ndarray, power: np.ndarray, n: int, min_sep: float) -> list[int]:
    """Indices of the `n` highest local maxima of `power` that are at least `min_sep` apart."""
    interior = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    if len(interior) == 0:
        interior = np.array([int(power.argmax())])
    chosen: list[int] = []
    for i in interior[np.argsort(power[interior])[::-1]]:
        if all(abs(f[i] - f[j]) >= min_sep for j in chosen):
            chosen.append(int(i))
            if len(chosen) == n:
                break
    return chosen


def in_box(t: np.ndarray, box: Box) -> np.ndarray:
    """Boolean mask of samples of `t` inside the box's predicted windows."""
    ph = np.mod(np.asarray(t, dtype=np.float64) - box.epoch_s + box.period_s / 2.0, box.period_s)
    return np.abs(ph - box.period_s / 2.0) <= box.duration_s / 2.0


@register
class BlsDetector:
    """Box least squares over all nights of each star; see the module docstring."""

    name = "bls"

    def __init__(
        self,
        min_period_d: float = 0.5,
        max_period_d: float | None = None,
        durations_h: tuple[float, ...] = DEFAULT_DURATIONS_H,
        oversample: float = 3.0,
        snr_threshold: float = 7.0,
        min_events: int = 2,
        clip_sigma: float = 4.0,
        tail_sigma: float = 1.0,
        support_snr: float = 2.0,
        guard_fap: float | None = 1e-3,
        guard_ratio: float = 1.0,
        guard_f_min_per_d: float = 1.0,
        guard_f_max_per_d: float = 30.0,
        night_gap_cadences: int = 100,
        common_mode_min_stars: int = 10,
        second_pass: bool = False,
        detrend_pulsation: bool = True,
        trend_min_ratio: float = 3.0,
        trend_snr_threshold: float = 9.0,
        shape_threshold: float | None = 1.5,
        guard_night_ratio: float | None = 0.81,
        flare_fraction: float = 0.03,
        flare_snr_threshold: float = 7.2,
        flare_window_h: float | None = 3.0,
        flare_min_run: int = 10,
    ) -> None:
        if not min_period_d > 0:
            raise ValueError(f"min_period_d must be > 0, got {min_period_d}")
        if max_period_d is not None and not max_period_d > min_period_d:
            raise ValueError(
                f"max_period_d must be > min_period_d ({min_period_d}), got {max_period_d}"
            )
        if len(durations_h) == 0 or not all(d > 0 for d in durations_h):
            raise ValueError(f"durations_h must be non-empty and positive, got {durations_h}")
        if not oversample > 0:
            raise ValueError(f"oversample must be > 0, got {oversample}")
        if not snr_threshold > 0:
            raise ValueError(f"snr_threshold must be > 0, got {snr_threshold}")
        if min_events < 1:
            raise ValueError(f"min_events must be >= 1, got {min_events}")
        if not clip_sigma > 0:
            raise ValueError(f"clip_sigma must be > 0, got {clip_sigma}")
        if tail_sigma < 0:
            raise ValueError(f"tail_sigma must be >= 0, got {tail_sigma}")
        if guard_fap is not None and not 0 < guard_fap < 1:
            raise ValueError(f"guard_fap must be in (0, 1) or None, got {guard_fap}")
        if not guard_ratio > 0:
            raise ValueError(f"guard_ratio must be > 0, got {guard_ratio}")
        if not 0 < guard_f_min_per_d < guard_f_max_per_d:
            raise ValueError(
                "guard frequencies must satisfy 0 < min < max, got "
                f"{guard_f_min_per_d}, {guard_f_max_per_d}"
            )
        if not trend_min_ratio >= 1:
            raise ValueError(f"trend_min_ratio must be >= 1, got {trend_min_ratio}")
        if not trend_snr_threshold > 0:
            raise ValueError(f"trend_snr_threshold must be > 0, got {trend_snr_threshold}")
        if shape_threshold is not None and not shape_threshold > 0:
            raise ValueError(f"shape_threshold must be > 0 or None, got {shape_threshold}")
        if guard_night_ratio is not None and not guard_night_ratio > 0:
            raise ValueError(f"guard_night_ratio must be > 0 or None, got {guard_night_ratio}")
        if not 0 < flare_fraction <= 1:
            raise ValueError(f"flare_fraction must be in (0, 1], got {flare_fraction}")
        if not flare_snr_threshold > 0:
            raise ValueError(f"flare_snr_threshold must be > 0, got {flare_snr_threshold}")
        if flare_window_h is not None and not flare_window_h > 0:
            raise ValueError(f"flare_window_h must be > 0 or None, got {flare_window_h}")
        if flare_min_run < 1:
            raise ValueError(f"flare_min_run must be >= 1, got {flare_min_run}")
        self.min_period_d = min_period_d
        self.max_period_d = max_period_d
        self.durations_h = tuple(sorted(durations_h))
        self.oversample = oversample
        self.snr_threshold = snr_threshold
        self.min_events = min_events
        self.clip_sigma = clip_sigma
        self.tail_sigma = tail_sigma
        self.support_snr = support_snr
        self.guard_fap = guard_fap
        self.guard_ratio = guard_ratio
        self.guard_f_min_per_d = guard_f_min_per_d
        self.guard_f_max_per_d = guard_f_max_per_d
        self.night_gap_cadences = night_gap_cadences
        self.common_mode_min_stars = common_mode_min_stars
        self.second_pass = second_pass
        self.detrend_pulsation = detrend_pulsation
        self.trend_min_ratio = trend_min_ratio
        self.trend_snr_threshold = trend_snr_threshold
        self.shape_threshold = shape_threshold
        self.guard_night_ratio = guard_night_ratio
        self.flare_fraction = flare_fraction
        self.flare_snr_threshold = flare_snr_threshold
        self.flare_window_h = flare_window_h
        self.flare_min_run = flare_min_run

    # -- preprocessing -----------------------------------------------------------------

    def _clean(self, curve: LightCurve, cadence_s: int) -> _Clean | None:
        """Drop flagged/non-finite samples and divide each night by its median."""
        flux_all = np.asarray(curve.flux, dtype=np.float64)
        keep = ((np.asarray(curve.quality) & 1) == 0) & np.isfinite(flux_all)
        t = np.asarray(curve.t)[keep].astype(np.float64)
        flux = flux_all[keep]
        if len(t) < 4:
            return None
        night_of = np.empty(len(t), dtype=np.int64)
        for k, sl in enumerate(split_nights(t, cadence_s, self.night_gap_cadences)):
            night_of[sl] = k
        return _Clean(
            t, _divide_by_night_median(flux, night_of), night_of, _night_medians(flux, night_of)
        )

    def _common_mode(self, cleaned: Mapping[int, _Clean]) -> tuple[np.ndarray, np.ndarray] | None:
        """(time grid, per-time median over stars of the night-normalised flux), or None.

        Field-wide offsets the quality flag does not cover (a tracking drift holds its new
        level after the flagged span; a cloud's wings) hit every star alike and would
        otherwise be the deepest "transit" on every star. Times seen by fewer than
        `common_mode_min_stars` stars get no correction.
        """
        if self.common_mode_min_stars <= 0 or len(cleaned) < self.common_mode_min_stars:
            return None
        grid = np.unique(np.concatenate([c.t for c in cleaned.values()]))
        mat = np.full((len(cleaned), len(grid)), np.nan)
        for row, c in enumerate(cleaned.values()):
            mat[row, np.searchsorted(grid, c.t)] = c.flux
        counts = np.sum(~np.isnan(mat), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            cm = np.nanmedian(mat, axis=0)
        cm[counts < self.common_mode_min_stars] = 1.0
        return grid, cm

    def _prepare(
        self, c: _Clean, common_mode: tuple[np.ndarray, np.ndarray] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(t, flux, night, level) used by the search: common mode removed, renormalised, clipped.

        `level` is the factor that took each sample's flux from raw units to `flux`.
        """
        flux = c.flux
        level = c.level
        if common_mode is not None:
            grid, cm = common_mode
            x = flux / cm[np.searchsorted(grid, c.t)]
            flux = _divide_by_night_median(x, c.night_of)
            level = level * _night_medians(x, c.night_of)
        ok = np.ones(len(flux), dtype=bool)
        for k in range(int(c.night_of[-1]) + 1):
            m = c.night_of == k
            ok[m] = ~positive_outliers(flux[m], self.clip_sigma, self.tail_sigma)
        return c.t[ok], flux[ok], c.night_of[ok], level[ok]

    # -- guard ---------------------------------------------------------------------------

    def _band_power(
        self,
        t: np.ndarray,
        x: np.ndarray,
        sigma: float,
        cadence_s: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """(frequencies in cycles/day, normalised Fourier power, false-alarm level, bin step s).

        `x` (zero-mean flux) is summed into ``max(cadence_s, 300)`` s bins on one regular grid
        (empty bins are zero) and Fourier transformed, zero-padded fourfold, and the guard band
        is kept. For white noise the normalised power ``|X(f)|**2 / (n sigma**2)`` of each
        independent frequency is exponentially distributed with unit mean, so over ``M =
        (f_max - f_min) * baseline`` independent frequencies the level exceeded by chance with
        probability ``guard_fap`` per star is ``ln(M / guard_fap)``. A sinusoid of amplitude A
        over n samples has power ``n A**2 / (4 sigma**2)`` and gains ``2 * power`` in
        chi-square. `t` and `x` hold only the samples that take part (the others are simply
        absent). None when the band holds no frequency.
        """
        dt = float(max(cadence_s, 300))
        n_bins = int(np.floor((t[-1] - t[0]) / dt)) + 1
        bins = np.floor((t - t[0]) / dt).astype(np.int64)
        sums = np.bincount(bins, weights=x, minlength=n_bins)
        n_fft = 1 << math.ceil(math.log2(4 * n_bins))
        spec = np.fft.rfft(sums, n_fft)
        f_per_d = np.arange(len(spec)) * (DAY_S / (n_fft * dt))
        f_hi = min(self.guard_f_max_per_d, 0.5 * DAY_S / dt)
        band = (f_per_d >= self.guard_f_min_per_d) & (f_per_d <= f_hi)
        if not band.any():
            return None
        power = np.abs(spec[band]) ** 2 / (len(t) * sigma**2)
        independent = max(1.0, (f_hi - self.guard_f_min_per_d) * (t[-1] - t[0]) / DAY_S)
        assert self.guard_fap is not None
        return f_per_d[band], power, math.log(independent / self.guard_fap), DAY_S / (n_fft * dt)

    def _peak_power(
        self, t: np.ndarray, x: np.ndarray, sigma: float, cadence_s: int
    ) -> tuple[float, float]:
        """(strongest normalised Fourier power of `x` in the guard band, its false-alarm level).

        See :meth:`_band_power` for the normalisation and the level.
        """
        spectrum = self._band_power(t, x, sigma, cadence_s)
        if spectrum is None:
            return 0.0, math.inf
        return float(spectrum[1].max()), spectrum[2]

    def _pulsating(
        self, t: np.ndarray, flux: np.ndarray, sigma: float, cadence_s: int, box: Box
    ) -> tuple[bool, float]:
        """(is it a pulsator, strongest residual-sinusoid power). Two tests, either rejects.

        * Residual: subtract the box over the inner 0.8 of its duration (add its depth back
          to those samples), leave out the rest of a window 1.5 times as wide (a deep
          transit's ramps and the box's edge error), and
          look for any sinusoid in what is left; a transit leaves white noise, a pulsator
          its pulsation. Rejected when the power exceeds the false-alarm level.
        * Box versus sine: the chi-square gain of the best sinusoid anywhere in the band,
          fitted to the whole record, is compared with the box's. A box of duty cycle at
          most one half is never matched by one sinusoid (it keeps at most 81% of a square
          wave's variance, 54% at a quarter duty), a sinusoid is matched better by a sine
          than by its best box (the box keeps at most 81% of its variance, the same overlap
          read the other way; the margin is thin at some periods, the residual test holds the
          rest); rejected when
          ``2 * power >= guard_ratio * depth**2 * n_in * n_out / (n * sigma**2)``.
        """
        core = in_box(t, replace(box, duration_s=_GUARD_CORE_WIDTH * box.duration_s))
        ring = in_box(t, replace(box, duration_s=_GUARD_MASK_WIDTH * box.duration_s)) & ~core
        keep = ~ring
        outside = keep & ~core
        if not outside.any():  # nothing is left to hold the box against
            return True, math.inf
        depth = float(flux[outside].mean() - flux[core].mean()) if core.any() else 0.0
        resid = flux - flux[outside].mean() + depth * core
        r_power, r_crit = self._peak_power(t[keep], resid[keep], sigma, cadence_s)
        n, n_in = len(t), box.n_in
        box_chi2 = box.depth**2 * n_in * (n - n_in) / (n * sigma**2)
        s_power, _ = self._peak_power(t, flux - 1.0, sigma, cadence_s)
        return (r_power >= r_crit or 2.0 * s_power >= self.guard_ratio * box_chi2), r_power

    def _shape_excess(
        self, t: np.ndarray, flux: np.ndarray, night: np.ndarray, sigma: float, box: Box
    ) -> float:
        """How much better a triangle fits the box's events than any box does (0 if no verdict).

        An eclipse is V-shaped, a transit has a flat bottom, and a box fit to a V sits on its core
        (the grid stops at 6 h, a binary's eclipse can last 13 h). The samples of the nights that
        hold an event are put at their offset from the nearest event centre (`x`). Two families of
        templates are fitted to the dip ``1 - flux`` with a free depth and a free level for every
        night (the nights were divided by their own medians, which an event distorts): boxes of
        a geometric ladder of widths from the shortest to the longest duration, and triangles
        (``max(0, 1 - 2|x| / w)``, base ``w``) from the shortest duration to
        ``_SHAPE_TRIANGLE_SPAN`` times the longest, never wider than the period. Both are tried at
        centres within ``_SHAPE_SHIFT_H`` of the box in steps of ``_SHAPE_SHIFT_STEP`` of their
        width, so that a V is not preferred only because it forgives a centre the phase bins
        (15 minutes) and the period grid placed badly: a 0.8 h transit fit with a centre a
        quarter hour off read as a V. With ``g`` the largest chi-square gain
        (``sigma**2`` times it, in flux units) of each family the result is ``(g_triangle -
        g_box) / (sigma * sqrt(g_box))``: about ``(chi2_triangle - chi2_box) / S/N``, which
        does not grow with the signal. A box-shaped dip makes the box win and the result negative
        (the least squares gain of a triangle on a box is at most 0.89 of the box's); a V makes it
        positive. Zero when the box's nights hold fewer than ``_SHAPE_MIN_SAMPLES`` samples or no
        dip fits either family.
        """
        period = box.period_s
        cycle = np.rint((t - box.epoch_s) / period)
        x = (t - box.epoch_s) - cycle * period
        events = np.unique(night[np.abs(x) <= box.duration_s / 2.0])
        nights = np.isin(night, events)
        if int(nights.sum()) < _SHAPE_MIN_SAMPLES:
            return 0.0
        x = x[nights]
        dip = 1.0 - flux[nights]
        k = np.unique(night[nights], return_inverse=True)[1]
        count = np.bincount(k).astype(np.float64)

        def terms(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Per night: ``centred template @ dip`` and ``centred template @ centred template``."""
            centred = template - (np.bincount(k, weights=template) / count)[k]
            return np.bincount(k, weights=centred * dip), np.bincount(k, weights=centred**2)

        reach = _SHAPE_SHIFT_H * HOUR_S

        def best_gain(widths: np.ndarray, triangle: bool) -> float:
            """Largest chi-square gain over the widths, each night centred where it fits best."""
            best = 0.0
            for w in widths:
                num = []
                den = []
                for c in np.arange(-reach, reach + 1e-6, min(reach, _SHAPE_SHIFT_STEP * w)):
                    d = np.abs(x - c)
                    template = np.clip(1.0 - 2.0 * d / w, 0.0, None) if triangle else d <= w / 2.0
                    n_c, d_c = terms(template.astype(np.float64))
                    num.append(n_c)
                    den.append(d_c)
                num_a, den_a = np.array(num), np.array(den)
                score = np.where(
                    (den_a > 0) & (num_a > 0), num_a / np.sqrt(np.maximum(den_a, 1e-300)), 0.0
                )
                pick = score.argmax(axis=0)
                cols = np.arange(len(pick))
                total, norm = float(num_a[pick, cols].sum()), float(den_a[pick, cols].sum())
                if total > 0 and norm > 0:
                    best = max(best, total * total / norm)
            return best

        lo, hi = self.durations_h[0] * HOUR_S, self.durations_h[-1] * HOUR_S
        wide = np.geomspace(lo, _SHAPE_TRIANGLE_SPAN * hi, _SHAPE_TRIANGLE_STEPS)
        g_box = best_gain(np.geomspace(lo, hi, _SHAPE_BOX_STEPS), triangle=False)
        g_triangle = best_gain(wide[wide <= period], triangle=True)
        if not g_box > 0:
            return 0.0
        return (g_triangle - g_box) / (sigma * math.sqrt(g_box))

    def _dip_mask(self, t: np.ndarray, flux: np.ndarray, box: Box, sigma: float) -> tuple[Box, ...]:
        """Windows (boxes at the best period) that cover the box's dip and its secondary.

        A box is at most 6 h wide and an eclipse of a binary can last twice that, so the
        windows of the box alone leave the wings of every eclipse in the record. `flux` is in
        the star's own units (each night divided by the star's median night level, not by its
        own median, so an eclipse that fills a night keeps its depth). It is folded on the
        box's period in phase bins of half the shortest duration; a bin is *depressed* when
        its mean is below 1 by ``support_snr`` standard errors, and a dip is a run of bins
        depressed by ``tail_sigma`` standard errors that holds at least one such bin (the tail
        growth of :func:`positive_outliers`, downwards, in phase). The primary is the box's
        own bins and every dip that touches them. The secondary is a dip centred half a
        period from the primary that reaches ``snr_threshold`` on its own and lasts at least
        ``_SECONDARY_WIDTH`` of the primary (the two eclipses of a circular binary last
        equally long; a planet that happens to sit there is short, and masking it would hide
        it). Each window is as wide as its dip plus one box duration each side.
        """
        period = box.period_s
        nb = max(_DIP_MIN_BINS, math.ceil(period / (self.durations_h[0] * HOUR_S / 2.0)))
        width = period / nb
        # phase counted from the box centre, which then sits in the middle of the fold
        b = np.minimum((np.mod(t - box.epoch_s + period / 2.0, period) / width).astype(int), nb - 1)
        n_b = np.bincount(b, minlength=nb).astype(np.float64)
        deficit = np.bincount(b, weights=1.0 - flux, minlength=nb)
        z = deficit / (sigma * np.sqrt(np.maximum(n_b, 1.0)))
        runs = []
        for start, length in _circular_runs(z >= self.tail_sigma):
            idx = (start + np.arange(length)) % nb
            if (z[idx] >= self.support_snr).any():
                runs.append((idx, length))
        core = np.abs(np.arange(nb) + 0.5 - nb / 2.0) <= max(
            box.duration_s / width / 2.0, _DIP_CORE_MIN_BINS
        )
        primary = core.copy()
        primary_len = int(core.sum())
        for idx, length in runs:
            if core[idx].any():
                primary[idx] = True
                primary_len = max(primary_len, length)
        half = np.roll(primary, nb // 2)
        keep = primary.copy()
        for idx, length in runs:
            if primary[idx].any() or not half[idx].any() or length < _SECONDARY_WIDTH * primary_len:
                continue
            if deficit[idx].sum() / (sigma * math.sqrt(n_b[idx].sum())) >= self.snr_threshold:
                keep[idx] = True
        return tuple(
            replace(
                box,
                epoch_s=(box.epoch_s + (start + length / 2.0 - nb / 2.0) * width) % period,
                duration_s=length * width + 2.0 * box.duration_s,
            )
            for start, length in _circular_runs(keep)
        )

    # -- main --------------------------------------------------------------------------

    def _search(
        self,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
        masked: Sequence[Box] = (),
        plan_mask: bool = False,
    ) -> Candidate | None:
        """The star's best box with its S/N and, if it is not reportable, why not.

        Samples inside the predicted windows of any box in `masked` are left out of the
        search, the statistics, the guard and the event list (they are treated as not
        observed) and each night is divided by the median of what is left. With `plan_mask`
        a rejected candidate carries the windows a re-search should leave out
        (:meth:`_dip_mask`). A candidate that is not reportable is searched once more with the
        star's pulsation subtracted (:meth:`_detrended`), and replaced if that one stands.
        """
        t, flux, night, level = self._prepare(c, common_mode)
        flare_fraction = 1.0 - len(t) / len(c.t)
        windows = np.zeros((0, 2))
        if self.flare_window_h is not None:
            windows = flare_windows(
                c.t, c.night_of, np.isin(c.t, t), self.flare_min_run, self.flare_window_h * HOUR_S
            )
        t_obs, night_obs = c.t, c.night_of
        if masked:
            drop = np.zeros(len(t), dtype=bool)
            drop_obs = np.zeros(len(t_obs), dtype=bool)
            for m in masked:
                drop |= in_box(t, m)
                drop_obs |= in_box(t_obs, m)
            t, flux, night, level = t[~drop], flux[~drop], night[~drop], level[~drop]
            t_obs, night_obs = t_obs[~drop_obs], night_obs[~drop_obs]
            flux = _renormalise_nights(flux, night)
        arrays = _Arrays(t, flux, night, level, t_obs, night_obs, windows, flare_fraction)
        cand = self._search_arrays(arrays, c, cadence_s, plan_mask)
        if cand is None or cand.rejected != "pulsation guard" or not self._can_detrend():
            return cand
        again = self._detrended(arrays, c, cadence_s, plan_mask)
        return cand if again is None else again

    def _can_detrend(self) -> bool:
        return self.detrend_pulsation and self.guard_fap is not None

    def _white_sigma(self, flux: np.ndarray, night: np.ndarray) -> float:
        """Scatter of the white noise: ``1.4826 * MAD`` of the within-night first differences
        over ``sqrt(2)``. Unlike the scatter about the median it does not contain a slow
        pulsation, whose amplitude would otherwise hide the pulsation it comes from."""
        same = night[1:] == night[:-1]
        d = np.diff(flux)[same]
        if len(d) < 2:
            return 0.0
        return _MAD_SCALE * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)

    def _pulsation_trend(
        self, a: _Arrays, cadence_s: int, box: Box | None = None
    ) -> tuple[np.ndarray, tuple[float, float, float]] | None:
        """The star's pulsation at the samples of `a`, and ``(cycles/day, amplitude, ratio)``.

        The best sinusoid in the guard band with a free intercept for every night: the six
        strongest peaks of the Fourier power (:meth:`_band_power`) are each placed to
        a fraction of a zero-padded bin by :func:`_refine_frequency`, amplitude and phase are
        fitted by least squares, and the largest chi-square gain wins. The sinusoid counts only
        when its power (``gain / (2 sigma**2)``, ``sigma`` the white-noise scatter) is at least
        ``trend_min_ratio`` times the guard's false-alarm level (``ratio`` is that multiple);
        None otherwise. The samples of `box` and a window 1.5 times as wide are left out of the
        fit, so that a transit is not fitted by the sinusoid; with a `box` the sinusoid is also
        refused when a slower one fits better (:meth:`_slower_sinusoid_wins`).
        """
        keep = np.ones(len(a.t), dtype=bool)
        if box is not None:
            keep = ~in_box(a.t, replace(box, duration_s=_GUARD_MASK_WIDTH * box.duration_s))
        if int(keep.sum()) < 4:
            return None
        sigma = self._white_sigma(a.flux[keep], a.night[keep])
        if not sigma > 0:
            return None
        td = (a.t - float(a.t[0])) / DAY_S
        k = np.unique(a.night[keep], return_inverse=True)[1]
        x = _demean_by_night(a.flux[keep] - 1.0, k)
        fit = self._strongest_sinusoid(a.t[keep], td[keep], x, k, sigma, cadence_s, float(td[-1]))
        if fit is None:
            return None
        gain, f, sin_a, cos_a, level = fit
        ratio = gain / (2.0 * sigma**2) / level
        if ratio < self.trend_min_ratio:
            return None
        if box is not None and self._slower_sinusoid_wins(td[keep], x, k, gain):
            return None
        arg = 2.0 * np.pi * f * td
        trend = sin_a * np.sin(arg) + cos_a * np.cos(arg)
        return trend, (f, math.hypot(sin_a, cos_a), ratio)

    def _strongest_sinusoid(
        self,
        t: np.ndarray,
        td: np.ndarray,
        x: np.ndarray,
        k: np.ndarray,
        sigma: float,
        cadence_s: int,
        span_d: float,
    ) -> tuple[float, float, float, float, float] | None:
        """``(chi-square gain, cycles/day, sin, cos, false-alarm level)`` of the best sinusoid.

        `t` (s) and `td` (days) are the samples, `x` their zero-mean flux with each night's mean
        removed, `k` the night number, `sigma` the white-noise scatter the Fourier power is
        normalised with and `span_d` the length of the whole record in days (peaks closer than
        its inverse are one peak). The power of :meth:`_band_power` only ranks the peaks: it is
        approximate (it ignores the nights' intercepts, the cross term between a frequency and
        its mirror, and the gaps inside nights), and a sinusoid and its alias one cycle/day away
        are close in it; the few strongest peaks are placed by least squares and the one with the
        largest chi-square gain, with a free intercept for every night, wins. None when the
        guard band holds no frequency.
        """
        spectrum = self._band_power(t, x, sigma, cadence_s)
        if spectrum is None:
            return None
        f_band, power, level, df = spectrum
        best: tuple[float, float, float, float] | None = None  # gain, f, sin, cos
        for i in _top_peaks(f_band, power, _TREND_PEAKS, 1.0 / max(1.0, span_d)):
            f = _refine_frequency(td, x, k, float(f_band[i]), df)
            sin_a, cos_a, gain = _sine_fit(td, x, k, f)
            if best is None or gain > best[0]:
                best = (gain, f, sin_a, cos_a)
        assert best is not None
        return (*best, level)

    def _night_sine_ratio(
        self, t: np.ndarray, flux: np.ndarray, night: np.ndarray, box: Box, cadence_s: int
    ) -> float:
        """Chi-square gain of the best sinusoid over the box's, both with a free level per night.

        The nights were divided by their own medians, which absorb part of a sinusoid of a few
        hours in a night of eight, so the pulsation guard's plain Fourier power (no night level,
        a scatter that contains the pulsation) understates a pulsator against the box. Here both
        are fitted to the whole record the way the nights make them comparable: the box's gain is
        ``(sum w_k d_k)**2 / sum w_k`` over the nights (:func:`night_contrasts`, the square of
        ``snr_local * sigma``) and the sinusoid's is the best in the guard band
        (:meth:`_strongest_sinusoid`). The scatter cancels. A box train keeps at most 81 percent of
        a sinusoid's variance and a sinusoid at most 81 percent of a square wave's, so the ratio
        is at most about 0.81 for a planet and at least about 1.23 for a pulsator whose best box
        is its fold. 0 when the scatter or the box leaves nothing to compare.
        """
        depth_k, weight_k = night_contrasts(t, flux, night, box)
        if len(weight_k) == 0 or not float(weight_k.sum()) > 0:
            return 0.0
        box_gain = float(depth_k @ weight_k) ** 2 / float(weight_k.sum())
        sigma = self._white_sigma(flux, night)
        if not (box_gain > 0 and sigma > 0):
            return 0.0
        k = np.unique(night, return_inverse=True)[1]
        x = _demean_by_night(flux - 1.0, k)
        fit = self._strongest_sinusoid(
            t,
            (t - float(t[0])) / DAY_S,
            x,
            k,
            sigma,
            cadence_s,
            (float(t[-1]) - float(t[0])) / DAY_S,
        )
        return 0.0 if fit is None else fit[0] / box_gain

    def _slower_sinusoid_wins(
        self, td: np.ndarray, x: np.ndarray, k: np.ndarray, gain: float
    ) -> bool:
        """True when a sinusoid slower than the guard band fits better than the one at `gain`.

        Nights are one day apart and every night is divided by its own median, so a sinusoid
        of period ``P`` days and one of ``1 / (1 / P + 1)`` differ only by what the 8 hours of
        a night show of them: a 3 day sinusoid is seen as a ramp in every night and folds into
        a box at 1.33 cycles/day, inside the guard band, whose subtraction leaves the ramp's
        other alias as a dip every night. Frequencies from ``_TREND_SLOW_F_MIN`` to the band's
        lower edge are tried (with the same free intercept for every night); when the best of
        them has the larger chi-square gain the star's variability is slower than what this
        method removes, and it is left to the guard.
        """
        slow = np.arange(_TREND_SLOW_F_MIN, self.guard_f_min_per_d, _TREND_SLOW_STEP)
        return any(_sine_fit(td, x, k, float(f))[2] > gain for f in slow)

    def _detrended(
        self, a: _Arrays, c: _Clean, cadence_s: int, plan_mask: bool
    ) -> Candidate | None:
        """The candidate found once the star's pulsation is subtracted, if it is reportable.

        Subtract-then-search, with the sinusoid refitted around the box it uncovers:

        1. Fit the sinusoid to the whole record, subtract it, search.
        2. When that search finds a box worth a look (S/N at the gate), fit the sinusoid again
           with that box's samples left out and search once more. A transit that is in the
           record when the sinusoid is fitted pulls the fit towards itself (a 1 percent, 3 h
           transit left a residual that failed the guard), and a box that does not belong to
           the star (the fold of the pulsation) must not decide which samples are left out the
           first time. Only the second fit is used for the answer when there is one.

        The result must pass every gate (``snr`` and ``snr_local`` at ``trend_snr_threshold``,
        a second look at a star whose first look failed); None when no sinusoid stands out or
        the result is rejected, and the caller keeps the original candidate then.
        """
        fit = self._pulsation_trend(a, cadence_s)
        if fit is None:
            return None
        flux = _renormalise_nights(a.flux - fit[0], a.night)
        cand = self._search_arrays(replace(a, flux=flux), c, cadence_s, plan_mask)
        if cand is not None and cand.rejected != "snr":
            fit = self._pulsation_trend(a, cadence_s, cand.box)
            if fit is None:
                return None
            flux = _renormalise_nights(a.flux - fit[0], a.night)
            cand = self._search_arrays(replace(a, flux=flux), c, cadence_s, plan_mask)
        if cand is None or cand.rejected is not None:
            return None
        if min(cand.snr, cand.snr_local) < self.trend_snr_threshold:
            return None
        return replace(cand, trend=fit[1][:2])

    def _search_arrays(
        self, a: _Arrays, c: _Clean, cadence_s: int, plan_mask: bool
    ) -> Candidate | None:
        t, flux, night, level, t_obs, night_obs = (
            a.t,
            a.flux,
            a.night,
            a.level,
            a.t_obs,
            a.night_obs,
        )
        if len(t) < 4:
            return None
        baseline = float(c.t[-1] - c.t[0]) + cadence_s
        min_p = self.min_period_d * DAY_S
        max_p = self.max_period_d * DAY_S if self.max_period_d is not None else baseline / 2.0
        if not max_p > min_p:
            return None
        durations = np.array(self.durations_h, dtype=np.float64) * HOUR_S
        periods = frequency_grid(baseline, min_p, max_p, float(durations[0]), self.oversample)
        bin_s = float(durations[0]) / 2.0
        tm, sm, cm = presample(t, flux, max(float(cadence_s), bin_s / 3.0))
        box, power_by_period = search_box(tm, sm, cm, periods, durations, bin_s)
        if box is None:
            return None
        star_flux = flux * level / float(np.median(level[np.unique(night, return_index=True)[1]]))
        flux = _renormalise_out_of_box(t, flux, night, box)
        box = refine(t, flux, box)
        resid = flux[~in_box(t, box)]
        if not box.depth > 0 or len(resid) < 2:
            return None
        sigma = _MAD_SCALE * float(np.median(np.abs(resid - np.median(resid))))
        if not sigma > 0:
            return None
        snr = box.snr(sigma)
        alt = self._alternative(box, power_by_period, periods)
        depth_k, weight_k = night_contrasts(t, flux, night, box)
        z_k = depth_k * np.sqrt(weight_k) / sigma
        w_sum = float(weight_k.sum())
        snr_local = (
            float(depth_k @ weight_k / (sigma * math.sqrt(w_sum))) if w_sum > 0 else math.nan
        )
        supported = int((z_k >= self.support_snr).sum())
        snr_clean = snr_local
        after = in_windows(t, a.flare_windows)
        if after.any():  # the same box with the samples a flare may still lift left out
            depth_c, weight_c = night_contrasts(t[~after], flux[~after], night[~after], box)
            w_c = float(weight_c.sum())
            snr_clean = (
                float(depth_c @ weight_c / (sigma * math.sqrt(w_c))) if w_c > 0 else math.nan
            )

        def reject(
            why: str,
            excess: float | None = None,
            shape: float | None = None,
            sine_ratio: float | None = None,
        ) -> Candidate:
            mask = ()
            if plan_mask and why != "snr":
                mask = self._dip_mask(t, star_flux, box, sigma)
            return Candidate(
                box,
                sigma,
                snr,
                [],
                why,
                excess,
                alt,
                snr_local,
                supported,
                mask,
                shape=shape,
                sine_ratio=sine_ratio,
                snr_clean=snr_clean,
            )

        local_bar = self.snr_threshold
        if a.flare_fraction >= self.flare_fraction:  # a flare star: its tails make red noise
            local_bar = max(local_bar, self.flare_snr_threshold)
        if snr < self.snr_threshold:
            return reject("snr")
        if not snr_local >= local_bar:  # also an undefined local S/N (no night has both)
            return reject("local snr")
        events = self._events(t_obs, night_obs, box, cadence_s)
        if len({n for _, _, n in events}) < self.min_events:
            return reject("min_events")
        if supported < self.min_events:
            return reject("support")
        r = None
        ratio = None
        if self.guard_fap is not None:
            pulsating, r = self._pulsating(t, flux, sigma, cadence_s, box)
            if not pulsating and self.guard_night_ratio is not None:
                ratio = self._night_sine_ratio(t, flux, night, box, cadence_s)
                pulsating = ratio >= self.guard_night_ratio
            if pulsating:
                return reject("pulsation guard", r, sine_ratio=ratio)
        shape = None
        if self.shape_threshold is not None:
            shape = self._shape_excess(t, flux, night, sigma, box)
            if shape >= self.shape_threshold:
                return reject("shape", r, shape)
        if not snr_clean >= self.snr_threshold:  # also a box whose every event follows a flare
            return reject("flare tails", r, shape, ratio)
        return Candidate(
            box,
            sigma,
            snr,
            events,
            None,
            r,
            alt,
            snr_local,
            supported,
            shape=shape,
            sine_ratio=ratio,
            snr_clean=snr_clean,
        )

    def search(self, curve: LightCurve, cadence_s: int) -> Candidate | None:
        """Search one light curve on its own (no common-mode correction); None if unusable."""
        c = self._clean(curve, cadence_s)
        return None if c is None else self._search(c, None, cadence_s)

    def _star(
        self,
        star_id: int,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
    ) -> list[Detection]:
        cand = self._search(c, common_mode, cadence_s, plan_mask=self.second_pass)
        if cand is None:
            return []
        if cand.alternative is not None:
            log.info(
                "bls star %s: alternative period %.3f d (best %.3f d), power ratio %.2f",
                star_id,
                cand.alternative[0] / DAY_S,
                cand.box.period_s / DAY_S,
                cand.alternative[1],
            )
        if cand.rejected is not None:
            if cand.rejected != "snr":
                log.info(
                    "bls star %s: S/N %.1f (local %.1f) rejected: %s, sine power %s",
                    star_id,
                    cand.snr,
                    cand.snr_local,
                    cand.rejected,
                    cand.excess,
                )
            if not self.second_pass or not cand.mask:
                return []
            cand = self._re_search(star_id, c, common_mode, cadence_s, cand)
            if cand is None:
                return []
        if cand.trend is not None:
            log.info(
                "bls star %s: pulsation %.3f cycles/day, amplitude %.4f, subtracted; "
                "P %.3f d, S/N %.1f (local %.1f)",
                star_id,
                cand.trend[0],
                cand.trend[1],
                cand.box.period_s / DAY_S,
                cand.snr,
                cand.snr_local,
            )
        return [
            Detection(
                detector=self.name,
                det_id=f"{self.name}-s{star_id}-{k}",
                star_id=star_id,
                t_start=t0,
                t_end=t1,
                score=cand.snr,
                kind_guess="transit",
            )
            for k, (t0, t1, _) in enumerate(cand.events)
        ]

    def _re_search(
        self,
        star_id: int,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
        first: Candidate,
    ) -> Candidate | None:
        """The one re-search of a star whose best box was rejected (``first.mask`` is set).

        The record is searched again without the windows of ``first.mask``; the result is the
        second candidate, or None when the search found nothing or it is rejected. It is
        a candidate in its own right: every gate applies. When the first box was rejected by
        the pulsation guard its period belongs to a periodic signal (a binary, a pulsator),
        and the candidate is also refused as an ``"alias"`` if its period is that period or
        half or twice it, where the residue of what was masked repeats; the period of a box
        rejected for lack of events means nothing, so no alias is refused then. The outcome
        is logged at INFO.
        """
        second = self._search(c, common_mode, cadence_s, masked=first.mask)
        if second is None:
            log.info("bls star %s: re-search found no box", star_id)
            return None
        periodic = first.rejected in ("pulsation guard", "shape")
        if (
            periodic
            and second.rejected is None
            and _is_alias(second.box.period_s, first.box.period_s)
        ):
            second = replace(second, rejected="alias", events=[])
        log.info(
            "bls star %s: re-search found P %.3f d, S/N %.1f (local %.1f): %s",
            star_id,
            second.box.period_s / DAY_S,
            second.snr,
            second.snr_local,
            second.rejected or "accepted",
        )
        return second if second.rejected is None else None

    @staticmethod
    def _events(
        t_obs: np.ndarray, night_of: np.ndarray, box: Box, cadence_s: int
    ) -> list[tuple[int, int, int]]:
        """(t_start, t_end, night) of each predicted event covering at least one sample."""
        half = box.duration_s / 2.0
        n_lo = math.floor((t_obs[0] - half - box.epoch_s) / box.period_s)
        n_hi = math.ceil((t_obs[-1] + half - box.epoch_s) / box.period_s)
        out: list[tuple[int, int, int]] = []
        for n in range(n_lo, n_hi + 1):
            mid = box.epoch_s + n * box.period_s
            a = int(np.searchsorted(t_obs, mid - half, side="left"))
            b = int(np.searchsorted(t_obs, mid + half, side="right"))
            if b > a:
                out.append((int(t_obs[a]), int(t_obs[b - 1]) + cadence_s, int(night_of[a])))
        return out

    @staticmethod
    def _alternative(
        box: Box, power: np.ndarray, periods: np.ndarray
    ) -> tuple[float, float] | None:
        """(period_s, power / best power) of the strongest peak that is not the best period
        or a small-integer ratio of it, or None."""
        ratio = periods / box.period_s
        bad = np.zeros(len(periods), dtype=bool)
        for q in (1, 2, 3, 4):
            for p in range(1, 5):
                bad |= np.abs(ratio * q / p - 1.0) < 0.02
        masked = np.where(bad, 0.0, power)
        i = int(masked.argmax())
        if masked[i] <= 0:
            return None
        return float(periods[i]), float(masked[i] / box.power)

    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]:
        cleaned = {}
        for star_id in sorted(curves):
            c = self._clean(curves[star_id], cadence_s)
            if c is not None:
                cleaned[star_id] = c
        common_mode = self._common_mode(cleaned)
        out: list[Detection] = []
        for star_id, c in cleaned.items():
            out.extend(self._star(star_id, c, common_mode, cadence_s))
        return out
