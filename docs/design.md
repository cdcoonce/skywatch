# skywatch — design

_Status: v0, 2026-10-10. This document is the contract workers build against. Change it by PR, and say why in `docs/log/`._

## One paragraph

skywatch is a synthetic sky survey. A small array of telescopes watches a
field of stars night after night and returns one light curve per star: flux
against time, with noise. Into those light curves the simulator injects two
kinds of thing. **Science** events are real astrophysics: planet transits,
stellar flares, eclipsing binaries, slow pulsations. **Glitches** are the
instrument and the weather: cosmic-ray hits, cloud passes, tracking drift, hot
pixels, dropouts. Every injected thing is recorded as ground truth. Detectors
then read the light curves cold and emit candidate events. The scorer matches
candidates to truth and reports, per detector and per kind, what was caught,
what was missed, and what was mistaken for what. The product is the scoreboard
and the world behind it, not any single detector.

## Why this shape

The predecessor (Synthetic-Signal-Observatory, Dec 2025) flagged 3-sigma hits
on Gaussian noise. Nothing was there to detect, so no detector could be wrong,
so nothing could be learned. skywatch inverts that: the world has structure,
the structure is labelled, and the only interesting output is a measured
comparison. The rolling z-score from the old project is kept as the baseline
that is expected to lose.

Two rules follow from this and are not negotiable:

1. **Truth is written at injection time, never inferred afterwards.** A label
   exists because the simulator placed the event, not because something looked
   anomalous later.
2. **Detectors never see truth.** They receive light curves and nothing else.
   Scoring is the only place truth and detections meet.

## Vocabulary

| Term | Meaning |
| --- | --- |
| **Field** | The set of stars under observation. Fixed for a survey. |
| **Star** | One target. Has a quiescent flux, a noise level, and a variability class. |
| **Night** | One contiguous observing run at a fixed cadence (default 60 s, 8 h). |
| **Survey** | A sequence of nights over the same field. |
| **Observation** | One (star, time) sample: flux, flux_err, quality flag. |
| **Light curve** | All observations of one star in a survey, time-ordered. |
| **Event** | Something injected into one or more light curves. `origin` is `science` or `instrument`. |
| **Truth label** | The record of an injected event: which stars, which time span, which kind, which parameters. |
| **Detection** | A detector's claim: star, time span, score, optional kind guess. |
| **Match** | A detection paired with a truth label by overlap under the matching rule. |

## World model

### Stars

Each star has `star_id`, `flux0` (quiescent flux, photons per cadence, log-uniform
over roughly three decades), `noise_floor` (fractional, for scintillation and
systematics), and a `variability` class:

- `constant`: nothing but noise.
- `sinusoidal`: low-amplitude pulsation, period of hours, amplitude 0.1 to 2 %.
- `eclipsing_binary`: periodic primary and secondary eclipses; the eclipses are
  science events and get truth labels per eclipse.
- `flare_star`: elevated flare rate; flares still get individual labels.

Planets are properties of stars: zero or more, each with period, epoch, depth
(0.01 to 3 %), duration (0.5 to 6 h). Every transit that falls inside a night is
a science event with its own label.

### Noise

Per observation: `flux = truth_flux * (1 + eps_scint) + eps_photon`, where
`eps_photon ~ N(0, sqrt(truth_flux))` and `eps_scint ~ N(0, noise_floor)`.
`flux_err` reports the modelled sigma. This is deliberately simple; it can be
made nastier later (red noise) without changing any interface.

### Science events (origin = science)

| kind | shape | scope | parameters |
| --- | --- | --- | --- |
| `transit` | box with linear ingress/egress | one star | depth, duration, t_mid |
| `flare` | instant rise, exponential decay | one star | amplitude, decay_s, t_peak |
| `eclipse` | V-shaped dip | one star | depth, duration, t_mid, primary/secondary |
| `pulsation` | sinusoid, continuous | one star | period, amplitude (labelled once per night) |

### Glitches (origin = instrument)

| kind | shape | scope | parameters |
| --- | --- | --- | --- |
| `cosmic_ray` | single-sample positive spike | one star, one cadence | amplitude |
| `cloud` | multiplicative dimming, smooth edges | **all stars**, shared span | depth, duration, t_mid |
| `tracking_drift` | slow multiplicative ramp | **all stars**, shared span | slope, duration |
| `hot_pixel` | flux stuck at a constant high value | one star, span | value, duration |
| `dropout` | cadences missing (no observation rows) | all stars or one star | duration |

The field-wide glitches (`cloud`, `tracking_drift`) are the point of having a
field rather than one star: a detector that looks at one light curve at a time
cannot tell a cloud from a transit, and the scoreboard should show that.

### Determinism

One integer `seed` fully determines a survey. Use `numpy.random.SeedSequence(seed)`
and `spawn` one child stream per concern (field generation, per-star noise, per
kind of event). Adding a new event kind must not change the draws of existing
kinds; that is what the spawned streams buy. Timestamps are integers of seconds
since the survey epoch.

## Injector contract

Nights are built as a `Scene`, injectors mutate it, and the survey turns the
scene into light curves plus truth. This is what lets science, glitches and the
survey loop be built independently.

```python
@dataclass
class Scene:
    t: np.ndarray          # int64 seconds since survey epoch, one shared grid per night
    flux: np.ndarray       # float64 (n_stars, n_t): true flux before noise; starts at flux0
    mask: np.ndarray       # bool (n_stars, n_t): True = observed. Dropouts clear it.
    stars: list[Star]      # row i of flux/mask is stars[i]
    night_index: int
    cadence_s: int

class Injector(Protocol):
    kind: str
    def inject(self, scene: Scene, rng: np.random.Generator) -> list[TruthLabel]: ...
```

Rules:

- Injectors mutate `scene.flux` and `scene.mask` in place and return the truth
  labels for what they placed. They never add noise; the survey does that last.
- Order per night: science injectors first, then instrument injectors. A cloud
  multiplies everything under it, transits included; a hot pixel overwrites; a
  dropout masks. Truth labels record the injected span regardless of what later
  injectors did on top.
- Deterministic science (transits from `planets`, eclipses and pulsations from
  `Star.params`) is derived from the field, so it is reproducible across nights
  by construction. Stochastic science (flares) and all glitches are Poisson
  per night with rates from `SurveyConfig`.
- `Star.params` is a small dict holding class-specific parameters:
  `eclipsing_binary` -> `period_s, epoch_s, depth_primary, depth_secondary, duration_s`;
  `sinusoidal` -> `period_s, amplitude, phase`; `flare_star` -> `flare_rate_per_day`.
  `constant` has an empty dict.
- **Footprint.** Every truth label carries `params["footprint_samples"]`: the number of
  samples the event actually changed, counted by the injector from the scene itself (flux
  and mask before against after, restricted to samples observed at injection time), never
  from span arithmetic. For a one-star label it counts that star's samples; for a field-wide
  label it counts grid columns where any star changed. Dropouts count the samples they
  newly masked; a drift counts its persistent tail, not just the ramp. `0` means the event
  fell between samples (a flare peaking after the night's last sample, a sub-cadence
  dropout at a coarse cadence, a span clipped to a sliver) and no detector could ever see
  it. The label is still written, because truth records what was placed; the scorer
  excludes it. A later injector can still hide samples this one changed (a dropout over a
  cosmic ray); that is not counted. The same holds for hot pixels: `hot_pixel` overwrites flux and runs
  before `dropout`, so a cosmic ray, flare or transit sample under a hot pixel keeps its footprint
  count while its signal is erased; that label is then an unavoidable miss.
- Each injector receives its own `rng` spawned from the survey seed, keyed by
  `(night_index, kind)`, so adding or removing one kind does not perturb another.
- The survey applies noise after all injectors: `flux_obs = flux * (1 + N(0, noise_floor)) + N(0, sqrt(flux))`
  and `flux_err = sqrt(flux + (noise_floor * flux) ** 2)`. Masked samples are
  dropped, so a light curve's `t` is the night grid minus dropouts.
- `quality` is a bitmask the survey sets from what it knows about the
  instrument (bit 0 = inside a field-wide glitch span). Detectors may read it
  but it is not truth; it is what a real pipeline would also have.

## Data model

All tables are plain columnar records, written as Parquet and queried through
DuckDB. No ORM, no pandas requirement in the core.

```
stars        star_id, flux0, noise_floor, variability, params(json), ra, dec
planets      planet_id, star_id, period_s, epoch_s, depth, duration_s
observations star_id, t, flux, flux_err, quality      (t = int seconds)
truth        label_id, kind, origin, star_id|NULL(field-wide), t_start, t_end, params(json)
detections   detector, det_id, star_id, t_start, t_end, score, kind_guess|NULL
matches      detector, det_id, label_id|NULL, iou
scores       detector, kind, tp, fp, fn, precision, recall, f1, ..., unobservable
```

One **run** is one directory: `runs/<run_id>/` holding these files plus
`manifest.json` (seed, config, versions, timestamps).

## Detector interface

```python
class Detector(Protocol):
    name: str
    def detect(self, curves: Mapping[int, LightCurve], cadence_s: int) -> list[Detection]: ...
```

A `LightCurve` is three aligned numpy arrays (`t`, `flux`, `flux_err`) plus
`quality`. Detectors receive the whole field at once so that field-aware
detectors are possible; single-star detectors simply loop.

Planned detectors, in order:

1. `rolling_z` — the predecessor's method. Per star, rolling mean and std over a
   window, flag |z| above a threshold. Baseline. The window is per night; the
   first `warmup` (5) samples of a night are blind, and samples `warmup` to `window`
   use the samples seen so far, with z scaled by `sqrt(i / window)`.
   `warmup <= 0` or `warmup >= window` switches the growing window off (the first
   `window` samples of a night are then blind), so `--window 5` or smaller switches it
   off too. A NaN in the window, the sample or its `flux_err` gives z = 0 and never flags.
2. `median_residual` — subtract a running median, flag residual outliers. Should
   beat rolling_z on flares and cosmic rays.
3. `field_common_mode` — estimate the field-wide multiplicative trend from the
   ensemble, flag spans where it departs from 1. The only detector that can
   name a cloud.
4. `bls` — box least squares for periodic transits across nights (built; scored on
   the 20-night surveys, where it should be the only thing that finds shallow
   transits, and expected to report nothing real on 3-night ones). Per star it
   divides each night by its median (after dividing out the field's common mode),
   masks positive outliers with their flare tails, folds on periods from 0.5 d to
   half the baseline (uniform in log-frequency) with boxes of 0.5 to 6 h, and ranks
   boxes by `depth^2 n_in n_out / N`; the best box is a candidate at
   `snr = depth sqrt(n_in) / sigma >= 7` (also required against a per-night local
   contrast, with two nights each showing the dip), reported as one `transit`
   detection per predicted in-night event with the S/N as score. A pulsation guard
   rejects the star when a sinusoid fits the record at least as well as the box or
   still stands out after the box is subtracted (so eclipsing binaries are rejected
   too). One candidate per star, plus one re-search when the best box is rejected by
   a gate other than S/N (`second_pass`, **off by default**): the dip around the
   rejected box is masked, a binary's secondary included, and the star is searched
   again, the result facing every gate on its own (this is how a planet on an
   eclipsing binary is found). Measured: on the stored 20-night runs it recovers
   seed 7 star 138, a planet on an eclipsing binary (reported at the sub-harmonic
   period 1.145 d = 8.012/7 d; the period never reaches a `Detection`), and adds one
   false positive on seed 42; on held-out seeds 21-26 it added no true transits,
   3 false positives and 1 eclipse mistaken for a transit, which is why it is off.
   The first pass alone scored 30 transit true positives, 6 false positives, 13
   eclipses reported as transits and no pulsator reported as a transit on those
   seeds, and 5 of the 14 strong planets there sit on pulsating hosts and were
   rejected by the pulsation guard. `detrend_pulsation` (**on by default**) searches
   such a star again with its pulsation subtracted: the best sinusoid in 1-30
   cycles/day with a free intercept per night, subtracted, a search, then the
   sinusoid refitted with that box's samples left out and one more search (a
   subtract-then-search whose refit is the joint fit's sinusoid at the box found);
   the result faces every gate, with S/N and local S/N at 9, and a star that is
   rejected again keeps its original rejection. Measured on held-out seeds 21-26
   (scored once, after every threshold was fixed on 24 development runs): transit
   true positives 30 to 46, false positives 6 to 6, eclipses reported as transits
   13 to 13, no pulsator reported as a transit, 1.26 times the CPU time; of the five
   planets on pulsating hosts three now have every event bounded or touched (one of
   them 9 bounded and 1 touched of 10), one has one of two events bounded and one
   (seed 23 star 139: S/N 9.7 but local S/N 7.4, under the bar of 9) is still lost.
   One sinusoid per star, and a sinusoid slower than the guard band is left to the
   guard. A pulsator with one real transit can borrow a noise dip as the second
   event `min_events` asks for (dev seed 44 star 44; about 1 in 120 such stars on 240
   synthetic cases, and nothing on `main`). A detrended box can be made to repeat (issue #44, `trend_repeat_snr`,
   **off by default**): its S/N over the nights other than its strongest must reach 5. The fast
   pulsators the issue named (seed 112 star 62 at 20.65 cycles/day, seed 13 star 130 at 15.71)
   were not subtraction residue: on 5,254 pulsators the power left at the fundamental and at
   twice it is a median 0.2 of the false-alarm level in every band from 2 to 25 cycles/day, and
   the three false boxes among the 67 development and design boxes accepted after a subtraction
   each rest on one short night with a flare (S/N without that night 2.2, 3.6 and 4.4; the 52
   planets among them read 6.1 or more, a development-only gap). With the test on, it also
   refuses 11 boxes whose one real transit shared a box with a borrowed dip: on the 174
   standard development runs transit true positives 1248 to 1242, pulsations reported as
   transits 10 to 0; on held-out seeds 151-156 (scored once) 117 to 116 (a pulsator hosting a
   planet that transits once, seed 154 star 37) and 1 to 0.
   It ships off because the review of PR #50 found two things the development data hid.
   First, the transits it costs are lost to the system: the union of `bls` and `single_transit`
   loses transit true positives 151 to 140 on the 11 development and design runs where it fires,
   125 to 124 held-out, 176 to 175 and 100 to 95 on never-seen seeds (380-391, 392-395), for
   pulsations reported as transits 25 to 3, 1 to 0, 1 to 0 and 11 to 7: 28 false detections for
   18 real transits, and `single_transit` finds 0 of the 18 because it cannot see a pulsating host
   (issue #51). Second, the bar's gap did not hold unseen: a real two-transit planet read 3.57
   (refused), a false box 5.50 and a mixed one 5.73 (both kept); seed 21 star 85, a planet PR #38
   recovered, sits 1.9 above the bar. The default is flipped after #51, on seeds nobody has read.

   The common mode of a field of pulsators (issue #57). `bls` divides every star by the per-time
   median over the stars, and the median of a field of pulsators is a smooth wobble that the
   division writes into every star: on quiet nights its 2 hour block scatter is 0.034 percent on the
   all-sinusoidal runs 830 and 831 against 0.021 to 0.022 percent on standard runs. That wobble, not
   the repeat test's one-night or borrowed-dip case, is what made 10 pulsation->transit claims
   (twice the planet-free rule) on puls-pf-seed830: star 133's 8-event box survives subtracting the
   injected sinusoid, reads 7.59 in the repeat test (which is off) and shows in the other 179 stars
   of the run as a median local S/N of 1.22 where the same box at four other epochs reads -0.18 to
   0.63. When more than half the stars carry a significant sinusoid (`common_mode_pulsators` = 0.5;
   standard runs 0.115 to 0.28, all-sinusoidal 0.875 to 0.98) the median is taken a second time over the
   stars with their own sinusoid subtracted; below that it is the plain one (`single_transit` takes the
   same common mode since issue #59). This is a fix for the planet-free rule (at most 5 `bls` detections on an all-sinusoidal
   planet-free run), not a planet gain. `main` breaks the rule on 3 of the 34 all-sinusoidal planet-free
   runs read so far (seeds 371, 830 and held-out 472); the clean held-out evidence is seed 472 (14 to 2).
   Development (100 runs), `main` to the branch, `bls` alone: all-sinusoidal planet-free detections 21 to
   8 (seeds 830 and 811 go, 371 keeps 4 of 7, two small boxes appear). The 28 all-sinusoidal runs with
   planets gained planets 186 to 199 (transit labels bounded 724 to 781, pulsation->transit 55 to 46), but
   that gain did not replicate on never-seen seeds. Never-seen 474-489 (16 runs, the review of PR #58):
   union planets 56 to 55 (within the one-planet allowance), `bls` planets 40 to 39; on the 480-483 runs
   with planets `bls` planets 23 to 22; 474-479 (standard field) `bls` is byte-identical to `main` (`cmp`
   on all six); 484-489 (all-sinusoidal, planet-free) are 0 detections on `main` and on the branch, so they
   do not test the rule. Held-out, scored once (460-473): planet-free detections 0, 0, 14, 0 to 0, 0, 2,
   0, all-sinusoidal with planets planets 34 to 37, transit labels bounded 120 to 130, standard identical.
   Seeds 460-465 are not clean for the share gate (the share was added after reading them); the gate's
   clean evidence is 466-473 and the 16 never-seen runs, all on one side of the cliff: the share range
   0.28 to 0.875 has never been exercised, so 0.5 is a majority argument and not a measured cliff. The
   first variant took the second median in every field and raised standard-field fp on 460-465 from 3 to
   10 (two borderline flare-star boxes), having lowered it on the development runs 66 to 56: a standard
   field's effect is of either sign, so it is left as it was.
   The grid, statistic, guard, detrending and
   second pass are in the `skywatch/detectors/bls.py` docstring.

   The last gate is a shape test (issue #36). `main` reported 40 eclipse detections as
   transits on the 27 development runs (8 of 428 binaries, 5 of them with every reported event
   the same eclipse, so an odd-even or half-phase test had nothing to compare); the pulsation
   guard was far from rejecting them (twice the sine power over the box's chi-square 0.09 to
   0.38, residual power over the false-alarm level 0.37 to 0.90, against 1). Nights holding a
   box event are fitted with the best box and the best triangle (a V; a free level per night,
   each night centred within 0.5 h), and the candidate is rejected when `(g_triangle - g_box) /
   (sigma sqrt(g_box))`, about the chi-square advantage per unit of S/N, reaches
   `shape_threshold` = 1.5 (`Candidate.shape`, `rejected == "shape"`). Chosen on the
   development runs and planted cases (172 planted planets reaching the gate: highest 1.04;
   66 development planets: highest 0.93). Development, branch against `main`: transit true
   positives 202 to 202, false positives 56 to 56, eclipses reported as transits 40 to 3 (the
   3-night runs). Held-out seeds 21-26, scored once (the seeds on which #36 found the leak):
   transit true positives 46 to 46, false positives 6 to 6, eclipses reported as transits 13
   to 0, no pulsator reported as a transit; `single_transit` unchanged on 10 runs. Never-seen
   seeds 61-66 (reviewed after the numbers above were fixed): true positives 69 to 69, false
   positives 18 to 18, eclipses reported as transits 26 to 5; the planet hosts read at most 0.12,
   and the remaining leak is seed 64 star 195, a 7.04 h eclipse in 8 h nights, statistic 0.70.
   A V that fills the nights is not told from a box (seed 7's 3-night binary, a 12.7 h eclipse);
   seed 42's 3-night binary has a 2.4 h eclipse that does not fill them: its two event nights
   read 3.72 and 4.11 fitted alone and -1.29 jointly (explained below, issue #41). The margins
   are thin on the binary side (eclipses at 1.7 and 1.8, and 0.70 on the unseen seeds; a planet
   at 0.93 on development, 0.12 on the unseen seeds).

   Eclipses the joint reading missed (issue #41). Of the 38 binaries that reach the shape gate on
   78 development runs (the runs above, plus design seeds 280-291 at 20 nights and 280-299 at 3
   nights), 9 read under 1.5: one of them has an eclipse longer than the night (seed 7's 12.7 h in
   8 h nights), so "wider than the night" is not what leaks (3 of the 4 binaries with eclipses of
   a night or more are rejected). Two things hide the rest. The fit gives every event one depth,
   and the box at half the eclipse's period (22 of the 38) alternates primary and secondary or
   lands on a night-edge wing: seed 42's 3-night binary has a whole 14 percent secondary on
   night 0 and the last hour of a 50 percent primary on night 2, the box needs depths 0.087 and
   0.107 (one amplitude holds them), the triangle 0.133 and 0.72 (one cannot), so the box wins by
   64 in chi-square. And the ratio divides the chi-square advantage by the S/N: seed 64 star 195
   (S/N 19.2) is a V that wins by 23.9, about 5 sigma, and reads 0.70. The gate now also reads the
   fits with a depth for each night (the statistic is the larger of the two readings) and rejects
   when the chi-square by which the best V beats the best box with a depth for each night reaches
   `shape_chi2` = 14 (`Candidate.shape_chi2`; the highest planet is 9.1, the lowest binary it
   rescues 18.7, and 6 of 39 682 planted planets exceed 16). The chi-square is in units of the
   larger of the star's scatter and the box's residual scatter on the event nights, so noise
   that is not white raises the bar. Nothing the old reading rejected is let through. Development
   (71 planet runs), branch against `main`: transit true positives 448 to 448, false positives 50
   to 50, eclipses reported as transits 41 to 10 (all 10 on 3-night runs, from five binaries at
   S/N 8.1 to 12.8 with 2 or 3 event nights: the shape cannot be read there; their chi-square is
   at most 7.9 on development, but the never-seen 3-night runs leak one at 13.5, and the highest
   planet is 9.1 on development and 8.5 never seen, so the bar of 14 has a gap open both ways, and
   a bar of 10 would have caught the 13.5 but was not taken, as that would tune on seeds no longer
   held out), pulsations 4 to 4,
   flares 7 to 7; planet-free 41-44 and 41-43 unchanged except seed 42's 3-night run, 2 to 0.
   Held-out 201-206, scored twice (first with the variant in which a brightening night adds
   nothing, the `pick = 0` fix, which was then reverted; again with the final code; `shape_chi2` =
   14 and the noise floor were in 9141d90 before both scorings and did not change after): 40 to 40,
   7 to 7, 0 to 0, 0 to 0, 3 to 3 both times; no binary reached the gate on them, so they show no
   cost and cannot show the gain. Never-seen 300-311, also scored twice (the variant, which decided
   the `pick = 0` question, and the final code): 135 to 135, 15 to 14, eclipses 2 to 0. Never seen,
   reviewed afterwards: 320-331 (20 nights, 12 runs) and 340-369 (3 nights, 30 runs), 42 runs in
   all, the branch is byte-identical to `main` (transit true positives 173 = 173, false positives
   13 = 13, eclipses reported as transits 6 = 6): no gain and no cost. No binary reached the gate
   on 320-331 (41 stars did, 39 planet hosts at most 8.5 in chi-square and 2 flare stars); 15
   binaries did on the 3-night runs and three leak in both versions: seed 349 star 32 (chi-square
   3.8), seed 360 star 51 (6.2) and seed 361 star 78 (13.5, under the bar of 14 by 0.5); the other
   12 are rejected by `main`'s reading already. So the benefit is measured on development (seed
   288 star 48, 20 detections; seed 64 star 195, 5) and on one never-seen binary (seed 309 star
   57): the binaries this change rescues are rare. The `pick = 0` quirk of the one-amplitude fit (a night that
   brightens where the dip should be subtracts the first centre's score) was measured and kept:
   making it add nothing moved 14 of 223 statistics on development without crossing 1.5, and on
   the never-seen seeds took a planet from 0.82 to 1.503. `single_transit` is unchanged
   (detections byte-identical on 84 runs: the 78 development ones and the 6 held-out ones): it calls the one-amplitude reading on a single night.
   The method, table and limits are in the `skywatch/detectors/bls.py` docstring and
   `scripts/bls_wide_eclipse_audit.py`.

   Planet-free stars (issue #39). On a run with no planets every transit claim is false, and
   `bls` had never been held to the rule `single_transit` was set by (at most 5 detections on
   each 20-night planet-free run). At `main` the planet-free runs of seeds 41 to 44 gave 0, 4, 2
   and 10 detections, and eight design runs (seeds 51 to 58) 2, 2, 3, 12, 2, 2, 6 and 23. Over the
   41 development and design runs, stars without a planet made claims at these rates: flare stars
   41 of 763, sinusoidal 8 of 920, constant 2 of 4986, eclipsing binaries 3 of 558. Two gates, chosen
   on those runs. (1) A third pulsation test: the chi-square gain of the best sinusoid (1-30
   cycles/day, a free level for every night) over the box's gain with a free level for every night
   (`snr_local * sigma` squared), rejected at 0.81, the most a sinusoid can keep of a box train; the
   seven pulsators accepted at the first pass read 0.97 to 1.40 (the old box-versus-sine test 0.53
   to 0.99), planet hosts at most 0.75 (0.73 for the two real planets that read highest, seed 38
   star 168 and seed 23 star 114; never-seen planet hosts at most 0.52; the bar is not a floor, a
   Monte Carlo of the simulator's planet recipe at P 0.5 to 1.5 d lost 1 of 279 accepted planets,
   P 0.562 d, 4.84 h, ratio 0.847, roughly 1 in 5000 planets). (2) A star with 3
   percent or more of its samples masked as flares must reach a local S/N of 7.2: flare tails are
   red noise the scatter does not contain, 32 of the 41 flare-star claims are on such stars and no
   candidate of another class is; 7.2 is the lowest on a 0.1 grid that puts every 20-night
   planet-free run of the development and design seeds at 5 or fewer, and the lowest local S/N of a
   transit true positive on a flare star is 7.28. The flare bar sits inside the distribution of false
   claims, not in a gap of it (11 claims at 7.0 to 7.19 rejected, 7 at 7.2 to 7.4 survive; pf-seed44
   meets the rule because star 126 reads 7.195), so flare stars are not solved by a bar (issue #43,
   next paragraph; the fast pulsators of issue #44 were one short night with a flare, refused by the
   repeat test of a detrended box when it is on, above). Development, `main` to the branch (33 runs):
   transit true positives 271 to 271, false positives 74 to 58, eclipses reported as transits 8 to
   8, pulsations reported as transits 81 to 3 (29 to 0 on seeds 61-66), planet-free 20-night runs
   0, 4, 2, 10 to 0, 4, 2, 0. Held-out, scored once (seeds 71-76): transit true positives 33 to 33,
   false positives 8 to 8, eclipses reported as transits 5 to 5, pulsations reported as transits
   12 to 0. **The planet-free bar is not met on the held-out planet-free seeds 81-84**: 0, 6, 0, 0
   detections, at `main` as on the branch (two flare stars on seed 82, local S/N 7.5 and 9.1);
   not re-tuned. `single_transit` is identical span for span on 33 runs; the time of `bls` is
   unchanged. The audit is `scripts/bls_planet_free_audit.py`; the gates are in the
   `skywatch/detectors/bls.py` docstring.

   Flare tails (issue #43). What a claim on a flare star sits on was measured on the development
   runs (9 planet-free claims, 36 more without a planet match, 10 true positives on flare hosts; the
   table is in the issue's comment): the box is a stretch free of flare residue (mean 0.01 sigma) and
   the rest of its event nights carries 0.35 sigma of it (median 0.01 sigma), the tails the one-sided
   clip stops short of. The residue's own S/N is 77 percent of the local S/N in the middle of the
   claims (38 to 108 percent; 3 percent for the true positives, 0 to 74), 42 percent of it from flares
   the clip seeded and the rest from ones it never did. The box does not sit on a decay (31 of 45 have
   a smaller share of their samples within 1 h after a clipped sample than the rest of the star, 18 of
   45 within 3 h) and is not asymmetric (the levels before and after it differ by 0.10 sigma, the
   shape statistic is negative in 39 of 45), so neither a veto on events after a flare nor an
   asymmetry test describes them. The fix is a last gate: the local S/N of the same box with the
   samples inside and within 3 hours after each flare-sized masked run (10 or more consecutive clipped
   samples) left out of both sides of the contrast (`Candidate.snr_clean`) must reach 7.0; 3 hours is
   6 e-foldings of the longest decay the world makes. Development, `main` to the branch (64 stored
   runs): transit true positives 350 to 347, false positives (all kinds) 81 to 50, eclipses reported as
   transits 15 to 15, pulsations 3 to 3, flares reported as transits 32 to 6, detections 546 to 442,
   claims on flare stars 55 stars to 17; 18 more runs (91-100, 111-114, planet-free 101-104): true
   positives 187 to 187, false positives 21 to 15, flares 5 to 0. Planet-free 20-night runs, detections
   per run: seeds 41-44 0, 4, 2, 0 to 0, 0, 2, 0; 51-58 2, 2, 0, 0, 2, 2, 3, 5 to 0, 0, 0, 0, 0, 2, 0, 2;
   81-84 0, 6, 0, 0 to 0, 2, 0, 0; 101-104 0, 2, 3, 3 to 0, 0, 0, 3. Held-out, scored once (seeds
   131-136, planet-free 141-144): true positives 29 to 29, false positives 7 to 3, eclipses 0 to 0,
   pulsations 0 to 0, flares 5 to 0, detections 48 to 35; planet-free 2, 0, 0, 0 to 0, 0, 0, 0. Three
   development true positives are lost: seed 12 star 32 and seed 62 star 38 to flare windows over
   the box (flare stars whose box overlaps one planet event and is mostly residue), and seed 61
   star 56 (a constant star at local S/N 7.04) to a window opened by an 11-sample cosmic-ray run,
   not a flare. All three, and never-seen seed 261 star 29 (lost to a cosmic-ray run removing 152
   reference samples), are overlap credit at periods unrelated to the planet's (the box's period
   is 5.00, 0.515, 1.69 and 0.239 times the planet's), not planets found; the other 8 flare-host
   planets keep every grade.
   The flare bar of 7.2 stays: without it seed 96 star 10 (local S/N 7.13, clean 7.93) is claimed
   again. 9 of the 45 flare claims remain, flares the clip never seeded (an oracle that excludes 6
   e-foldings of every true flare leaves none). `single_transit` is identical span for span on 27
   runs; the time of `bls` is unchanged.

   What opens a window (issue #48; an opt-in mode, off by default). PR #47's definition, ten
   consecutive clipped samples, opens a window for every run, and the runs are mostly not flares. The 50 development 20-night runs (10,000
   stars) hold 23,338 flare-sized runs on 7,901 stars (79 percent), and on 5,500 of those stars none
   of the runs has a flare label under it: 42.4 percent of the runs have only a flare under them,
   36.9 percent only a hot pixel (a flat plateau), 15.9 percent only a cosmic ray (a spike widened by
   the clip and, by chance, 10 samples), 4.8 percent a mixture or none of the three
   (`scripts/bls_flare_run_audit.py`). A flare is `A exp(-(t - t_peak) / tau)` with `tau` 120 to
   1,800 s, so in the decay mode (`flare_tau_s=(60.0, 3600.0)`; the default `None` is PR #47's
   definition) a run opens a window only if the exponential fitted from its highest sample
   (`decay_fit`, in units of the night's scatter) has a decay time in 60 to 3,600 s (`flare_tau_s`:
   the world's range with a factor of 2 either side for the fit's scatter) and a residual of at most
   3 per degree of freedom (`flare_fit_chi2`). The fitted decay time of the flare runs is 128 s (1st
   percentile), 218 (5th), 898 (median), 1,681 (95th), 2,198 (99th) and their residual 0.79 per
   degree of freedom (median; 95th percentile 1.61); a cosmic ray's decay time is 15 s (a quarter
   of a cadence) at the median, a hot pixel's plateau has a residual of 648 (median; 5th percentile
   5.4). 9,746 of 9,888 flare-only runs (98.6 percent) still open a window, 42 of 8,613 hot-pixel-only
   and 14 of 3,712 cosmic-ray-only runs do. Stars carrying a window: 7,901 to 2,440 of 10,000. The
   decay time alone lets 5,849 hot-pixel runs open one, "the peak is in the first four samples" alone
   drops 26 percent of the flare runs and lets 761 hot-pixel and 1,292 cosmic-ray runs open one.
   The window length and the minimum run are unchanged.

   Measured, `main` (with PR #47's gate) to the decay mode. Development, 20 planet runs (seeds 7, 42,
   21-26, 61-66, 71-76): transit true positives 157 to 157, false positives (all kinds) 18 to 19,
   eclipses reported as transits 5 to 5, pulsations 0 to 0, flares 1 to 1, detections 188 to 190; 20
   planet-free runs (41-44, 51-58, 81-84, 101-104): false positives 6 to 6, flares 2 to 2,
   detections 11 to 11, per run unchanged (at most 3). The detection pair added is seed 62 star 25,
   a constant star whose two-event box (S/N 11.0, local 9.4, matched to a tracking-drift label) a
   hot pixel's 15-sample plateau had refused by removing its reference (clean S/N 2.2). Design check
   (12, 261, 440-445 planets): true positives 57 to 58, false positives 2 to 3, detections 65 to 67;
   planet-free 446-449: 0 to 0. Held-out, scored once (planets 400-405, planet-free 410-413):
   nothing changed (32 true positives, 3 false positives, 1 flare reported as a transit, 37
   detections; 0 detections on each planet-free run); stars carrying a window 1,647 to 525 of 2,000.
   Seed 261 star 29 (PR #47's never-seen loss) is found again, with one false positive event beside
   it; the three development losses stay lost, to flares (61/56's clean S/N 5.26 against 3.30,
   with the cosmic-ray window gone). `single_transit` is byte-identical on four runs.

   Why it ships off (review of PR #52). Never seen (planets 600-611, planet-free 620-623, 200 stars
   and 20 nights, simulated after the review began), `main` to the decay mode: transit true
   positives 69 to 70, false positives 23 to 25, flares reported as transits 3 to 4, detections
   107 to 112; 0 detections on each planet-free run on both. Nothing is removed in any run. The one
   true positive added (seed 609 star 150) is a box of the wrong period (5.14 d against the
   planet's 7.65 d) whose event a hot pixel's window had refused: overlap credit. The rest are
   false: seed 601 star 131 is a flare star (4.5 flares a day, no planet) claimed on three events
   (local S/N 7.75, clean 2.71 to 7.62), because a hot-pixel run that ended three minutes before a
   flare opened a window of three hours that covered the flare's tail inside the box, and `main`
   refused the box as "flare tails" by that accident. Seed 62 star 25 and the second event of
   seed 261 star 29 on the development runs are the same: the windows PR #47 opens after hot pixels
   refuse false boxes by accident, and narrowing them to real flares adds claims. So the mode is
   kept, measured and opt-in, and the default stays PR #47's definition (`bls` is byte-identical
   to `main` with it, `cmp` of `detections/bls.parquet` on seeds 42, 62 and 400).

   Limits. The 98.7 percent of flare-only runs that open a window (7,694 never-seen runs) is
   circular: the band (60 to 3,600 s) and the bar (3) come from the generator's own flare law, so
   it shows that the code reads the world's flares, not that the shape test would read real ones.
   A flare whose run the fit refuses (under a hot pixel) opens none: 229 of 3,525 never-seen runs
   with a flare label under them (6.5 percent). A lead, not tried: 23 of 28 hot-pixel-only runs
   that open a window have 5 samples or fewer from the peak to the run's end (at most 3 degrees of
   freedom, where nearly any exponential fits), and a minimum fitted length of 8 samples would have
   closed 26 of the 28 hot-pixel openings at a cost of 63 of 6,113 flare-only openings (1
   percent).

   Never seen (planets 261-266 and planet-free 271-274, 200 stars and 20 nights, simulated after
   the constants were fixed; the review of PR #47), `main` to the branch:

   | runs | transit tp | fp (all kinds) | flare->transit | detections | per run |
   |---|---|---|---|---|---|
   | planets 261-266 | 78 to 77 | 5 to 1 | 0 to 0 | 87 to 80 | 14, 3, 30, 15, 7, 18 to 10, 0, 30, 15, 7, 18 |
   | planet-free 271-274 | 0 to 0 | 7 to 2 | 8 to 4 | 25 to 12 | 11, 6, 6, 2 to 5, 3, 2, 2 |

   Eclipses reported as transits (3 to 3) and pulsations (1 to 1) are unchanged. The planet-free
   rule (at most 5 detections a run) is met, 271 exactly at the limit (star 13, a flare star at
   local S/N 9.43, clean 9.04).
5. `single_transit` — a one-event box search for the planets that transit once (built;
   scored on the single-event labels of the stored runs). Per star it uses `bls`'s
   preprocessing, then searches every night on its own for one box (0.5 to 6 h, at every
   start sample, including boxes cut by the night's start or end, so a transit against a
   night edge is searchable) against the same night's out-of-box samples and reports the best
   box per night at an S/N of 8 or more, the matched-filter S/N of that contrast, one
   `transit` detection per night. The threshold is the lowest at which every detection on a
   planet-free run (each one is a false transit claim) numbers at most 5 on each of four
   planet-free 20-night runs: 10, 2, 1, 3 detections at 6; 8, 2, 1, 2 at 7; 4, 0, 1, 2 at 8
   (the scorer's fp, a detection overlapping no truth label, is 0 on all of them from 6 to 10).
   The issue's `depth sqrt(n_in) / sigma` cannot meet that rule at any threshold from 7 to 10,
   so the contrast form stays.
   Guards reuse `bls`'s pulsation guard: a star whose strongest box fails it as a one-event
   box is searched with `bls`, and when that periodic best box is itself a guard rejection (a
   pulsator or an eclipsing binary) the binary's eclipse windows are masked and the star is
   searched again; a star still failing the guard is dropped. Measured at 8: on the stored
   development runs 4 of the 6 single-event labels at S/N 10 or more are bounded, so the bar
   of 5 is **not met** (it held at 6 only: the misses at 8 are a planet on an eclipsing
   binary, against an eclipse the mask removes, and a planet whose detector S/N is 6.63
   against a true 14.3), 0 false positives, eclipse->transit 2 on seed 7 and 0 on seed 42,
   pulsation->transit 1 on seed 42 (3 nights); on the held-out seeds 21 to 26, 9 of 13 are
   bounded (69 percent, as at 6), 0 false positives, eclipse->transit 0 to 7 per run (0, 0,
   3, 2, 0, 7), pulsation->transit 0. A star with two strong single events, or a planet with
   other events beside the one searched, is read as a sinusoid and dropped; so are the
   sinusoidal and flare hosts (held-out seed 22 star 81, seed 25 stars 39 and 52), and a dip
   that fills most of a night leaves no box with 30 samples outside it (seed 25 star 121:
   56 observed samples, 44 inside the label). Guard (b) tests `bls`'s rejection reason
   (`!= "pulsation guard"`), and `bls` now detrends pulsating hosts: with both merged, four
   stars flip from guard-rejected to accepted inside this detector's `bls` searches and guard
   (b) drops them instead of masking them; the detections are identical span for span on the
   4 development and 6 held-out runs, the planet-free all-detection counts are unchanged
   (4, 0, 1, 2), and this detector's CPU is about 2.2 times higher on a 20-night run (seed 26:
   15.8 s to 35.0 s). A shape test (#45) refuses V-shaped dips: each reported box's night
   goes through `bls`'s shape statistic (`_shape_excess`, a triangle against a box, read as a
   one-event box with the night's own scatter), and a star with any box at 1.0 or more is not
   reported, its other nights being read as eclipse residue. The threshold sits above the
   planets of 80 development runs (142 planet boxes, highest 0.73) and below the median of the
   eclipse boxes (1.29). Edge-truncated boxes get the same test and the same threshold: a
   transit cut by the night's edge is a step and an eclipse cut by it a ramp, and the test
   cannot call a short wing (none of the 78 planet boxes at an edge is refused; 66 of 111
   eclipse boxes at one are refused by the night alone, 89 of 119 eclipse boxes in all by the
   star rule). On the 27 development and design runs eclipse->transit goes 44 to 7, transit
   true positives 32 to 31 (a planet on an eclipsing binary, S/N 8.3), single-event planets at
   S/N 10 or more bounded 20 to 20; on the held-out seeds 221-226 (scored once) eclipse->transit
   11 to 6, those planets 11 of 14 to 11 of 14, and on the planet-free 231-234 all detections
   8 to 2; the highest planet statistic there is 0.82 (seed 224 star 101), a margin of 0.18
   against 0.27 on development (`bls`: 0.57). On never-seen seeds 241-246 (planets) and
   251-254 (planet-free), checked in the review of #46, eclipse->transit goes 9 to 1 and 6
   to 0, no planet is lost (transit true positives 11 to 11) and the planets top at 0.05, so
   they do not narrow the margin.
   Pulsating hosts (#51): guard (a) used to refuse a sinusoidal star whole, so a planet that
   transits once on one was found by nothing (PR #50's review: 18 transit labels on 17 hosts that
   `bls` loses with its repeat test on, `single_transit` finding 0). For a star guard (a)
   refuses, the search now runs after `bls`'s sinusoid is subtracted (`_pulsation_trend`,
   refitted once with the best box's window left out), the box must pass guard (a) again on the
   residual and then the shape test; a star that fails that goes on to guard (b) unchanged. No
   threshold was added or changed. On the 199 development runs (149 standard, 38 standard
   planet-free, 8 all-sinusoidal with planets, 4 all-sinusoidal planet-free) 7570 hosts are
   refused: only sinusoidal ones come through (85 of 4324 reported, 101 detections: 86 on a
   transit, 4 flare, 1 hot pixel, 1 cosmic ray, 9 on nothing but the pulsation, S/N 8.2 to 10.7
   on development; no eclipsing binary, flare or constant host reported). Of the 18 labels 16 are bounded (the
   other two: seed 44 star 44, whose raw night reads 7.0 so guard (a) never refuses it, and a
   second label whose residual reads 4.0). Planet-free: all detections on the 38 standard runs
   23 to 27 (at most 4 on a run, as at `main`), on the all-sinusoidal 370-373 0 to 1 in all. On
   the 14 runs that carry the 18 labels the union of `bls` and `single_transit` goes from 274
   to 287 transit labels bounded or touched, fp 3 to 3, pulsation->transit 28 to 32. Held-out,
   scored once (420-425 standard, 426-429 all-sinusoidal with planets, 430-433 all-sinusoidal
   planet-free): union transit labels bounded or touched 154 to 176, fp 3 to 3, eclipse->transit
   5 to 5, pulsation->transit 6 to 8 (the two new claims, S/N 8.8 and 8.3, sit beside a flare in
   the same night), detections 182 to 216, nothing on 430-433. `bls` is unchanged (`cmp` on four
   runs). Never-seen check in the review (700-711 standard, 712-715 all-sinusoidal with planets,
   720-723 all-sinusoidal planet-free): union transit labels bounded or touched 224 to 238, fp 20
   to 20, pulsation->transit 7 to 11, detections 289 to 328, no detection of `main` lost; the four
   new pulsation claims there are 712 star 75 (S/N 182.9, a ramp-down then a hot-pixel plateau 25 s
   after the box), 712 star 79 (9.8, a flare label 28 s after), 721 star 181 (9.1) and 722 star 55
   (10.8) (flare labels 1.0 h after). All six new claims on fresh seeds are flare- or
   hot-pixel-adjacent in the same night, about 0.4 per all-sinusoidal run, not noise dips: on a
   pulsator guard (a) used to refuse the whole star, so such a night never reached the search.
   The acceptance bar "pulsation->transit not higher" is missed on both fresh sets (6 to 8
   held-out, 7 to 11 never-seen); the detector ships on by default on the owner's decision,
   because the cost is one fixable class (the flare-adjacent gate of #54, below) and the gain
   is real finds. The next step after that is the follow-up PR that
   flips PR #50's `trend_repeat_snr` default, measured on its own seeds. PR #50's gate, measured
   in the review beside this detector (`bls` with `trend_repeat_snr=5.0`, 14 held-out runs
   420-433 and never-seen 700-715; 30 runs): the union of `bls` and `single_transit` goes from
   405 to 404 transit labels bounded (414 to 412 bounded or touched), pulsation->transit 19 to 10,
   detections 544 to 522, fp 23 to 23; 2 of 414 transit labels are lost (puls-seed715
   `transit-n2-s28-0`, puls-seed428 `transit-n7-s156-0`, both found only by `bls` with the gate
   off), not zero. That is the basis for the flip PR.
   Flares on a pulsator's residual (#54): the six pulsation->transit claims above are the
   flares and hot pixels that the pulsation hid from the first clip (its scale is the raw
   night's, inflated by the sinusoid). 12 of the 15 pulsation claims `main` makes on the
   detrended path across 60 development runs sit inside, or up to 1.05 h before, a run of 10 to 85
   samples that `bls`'s clip flags on the residual; the gate applies that clip to the residual
   and refuses a box whose S/N, measured without the flagged samples, is under the detector's
   own threshold (8). No distance, window or bar was added (`clip_residual`, on by default). It
   removes 13 development detections (12 pulsation claims, 1 transit: seed395 star 114, one
   label of a planet, found by `single_transit` alone) and added none on 88 runs (it is not
   structural: the gate runs before guard (a)): pulsation->transit
   15 to 3, union transit labels bounded or touched 768 to 767, planets 253 to 252. Held-out,
   scored once (820-825 standard, 826-829 all-sinusoidal with planets, 830-833 all-sinusoidal
   planet-free; 14 runs): `single_transit` pulsation->transit 4 to 0, detections 42 to 38 (all four
   removed are pulsation claims), no transit label lost; union with `bls`: transit labels bounded
   or touched 203 to 203, planets with a transit event bounded or touched 62 to 62, fp 7 to 7,
   pulsation->transit 20 to 16 (all `bls`'s), detections 253 to 249. Five of the six known
   claims are no longer reported (seed424 star 41, 427 star 103, 712 star 79, 721 star 181 and 722
   star 55); seed712 star 75 stays: a hot-pixel plateau covers most of a 96-sample night, is
   the median, and the box is the unaffected stretch, so no clip sees anything above the level
   (two more such cases on development, none held-out). The held-out bar is not as slack as the
   development one: seed826 star 162 reads 7.58 without its flagged samples, so a lower bar would
   have kept it. Never-seen seeds 500-513 (14 runs, checked in the review): union planets 62 to
   62, pulsation->transit 11 to 8, fp 3 to 3; `single_transit` pulsation->transit 5 to 2,
   detections 47 to 44. `bls` is unchanged (`cmp` on four runs).

   Short nights (issue #57). A night with fewer than two hours of samples (`min_night_h` = 2) is not
   searched: a dropout splits an 8 h night, and a hot pixel (up to an hour in the world) can cover
   most of the fragment, so the plateau is the median and the unaffected stretch reads as a 35 to 66
   percent dip that no clip sees (seed 712 star 75, seed 375 star 80, seed 376 star 37). On the 100
   development runs 89 of 2,122 nights are that short, `main` made 10 detections on them and none was a
   transit; detections 316 to 305, fp 1 to 1, pulsation->transit 5 to 1, no transit label lost.
   Held-out (460-473, scored once): detections 49 to 44, transit labels bounded 42 to 41 (puls-seed466
   star 14), fp 1 to 0, pulsation->transit 2 to 0.
   The bar acts in every search of a star, and the placement was compared on the development and
   held-out runs (the never-seen 474-489 cannot tell the variants apart): inside the
   pulsation-subtracted search it keeps seeds 375/80, 376/37, 712/75 and 466/104 out; inside the
   eclipse-mask re-search it loses one transit (seed 466 star 14: the mask leaves its night at 94 samples)
   and keeps four false boxes out (460/51, 601/64, 510/54 and the hot pixel 804/78) while one false box
   gets in (800/73), so it stays.
   Common mode (issue #59). `single_transit` passes the cadence to `bls`'s `_common_mode`, so on a
   field where more than half the stars carry a significant sinusoid it divides by the median over the
   pulsation-free stars, as `bls` does (issue #57); on a standard field it is the plain median, byte for
   byte. The lead of PR #58's review (480-483, planets 19 to 22) held on 36 development runs where every
   star is sinusoidal and `planet_rate` is 0.5: `single_transit` transit labels bounded 204 to 253, planets
   179 to 217, fp 0 to 0, pulsation->transit 1 to 0, detections 207 to 255; union with `bls` (unchanged)
   planets 355 to 376, bounded 1079 to 1099, fp 0 to 0, pulsation->transit 55 to 54, four runs losing
   planets (393, 480, 715, 826: 5 in all against 26 gained). 34 planet-free runs: `single_transit`
   detections 1 to 0, the largest run of the union 4 detections on both sides. Standard 420-425:
   `cmp`-identical. Held-out, scored once (520-533), `main` against the branch: standard 520-525
   `single_transit` identical on all six (`cmp`), `bls` identical (`cmp` on 520, 521, 526 and 528);
   all-sinusoidal with planets 526-529 `single_transit` bounded 25 to 30, planets 21 to 24, fp 0 to 0,
   detections 25 to 30, union planets 32 to 33, bounded 75 to 76, fp 0 to 0, pulsation->transit 9 to 9,
   detections 106 to 111; planet-free 530-533 no detection from either detector before or after. The
   held-out gain is one planet in the union, against 21 on development: `single_transit` alone gained 2
   planets on seed 526 (3 to 5) and 1 on seed 529 (9 to 10), and the union moved only on 529 (11 to 12;
   526 stays at 7 planets, which `bls` already had).
   Never-seen 540-547 (every star sinusoidal, `planet_rate` 0.5, `scripts/bls_detrend_audit.py --simulate`,
   `main` against the branch, `bls` unchanged): `single_transit` planets 47 to 60, fp 0 to 0,
   pulsation->transit 1 to 0; union planets 85 to 91, bounded 253 to 259, fp 0 to 0, pulsation->transit
   10 to 9. The gain mechanism: 7 of the 8 union planets gained and 20 of the 24 `single_transit` label
   gains are strong boxes (S/N 13 to 59) that the residual pulsation guard refused on `main`, because the
   plain median's field-wide wobble survives each star's own sinusoid subtraction and reads as "still
   pulsating"; the median over the freed stars has none. The loss mechanism: each loss (393, 480, 715
   twice, 826; 543 star 148, 544 star 134) is the same box on the same samples with a statistic across a
   bar, S/N across 8 (four times on development, twice here) and once the shape statistic across 1.0
   (826 star 68, 0.866 to 1.02), caused by the field-wide difference between the two medians over the
   box (0.0002 to 0.0009 of flux), not by the host star. Paired over 85 detections both sides have, the
   S/N change is -0.12 +- 0.07, not significant. The three regimes are consistent within the noise of
   about three changes per run: union planets per run +0.58 on development, +0.75 never-seen, +0.25
   held-out.
   The statistic, threshold, guards and limits are in the `skywatch/detectors/single_transit.py`
   docstring.
6. Later: matched flare filter, a learned model, an LLM-judge for comparison.

## Scoring

- A detection is **eligible** for a truth label when they are on the same star (or
  the label is field-wide) and their time spans overlap at all, or the detection
  is within one cadence of a single-sample label.
- An eligible pair is graded **bounded** when span IoU >= 0.3 (or the single-cadence
  tolerance holds) and **touched** when the spans overlap but IoU < 0.3.
- Matching is greedy by descending detection score (ties by `det_id`). A detection
  takes the unclaimed label with a bounded grade (highest IoU), else the unclaimed
  label with a touched grade (highest IoU). Star-level labels match at most one
  detection.
- **Edge tolerance** (`edge_tolerance_cadences`, default 1; 0 turns it off). After the
  pass above, a detection still unmatched that overlaps no label but lies within that
  many cadences of one's edge on the same star (or a field-wide label) is matched to it
  by a second pass over the leftovers, so overlap always beats edge. The distance is
  between the half-open spans: `det.t_start - label.t_end` after a label,
  `label.t_start - det.t_end` before it, and the detection is eligible when
  `0 <= distance < k * cadence_s` (0 = abutting, the first sample after or last sample
  before the label). A detection that overlaps any label on its star (claimed or not)
  is not edge-eligible; the overlap pass has already settled it (see fragments below).
  Among eligible labels any still-unclaimed one is preferred over an already-claimed
  one, even a nearer claimed one; within the chosen group the nearest edge wins, then the lowest `label_id`. The
  grade is **touched** if the label is still unclaimed, and **corroborating** if this
  detector already claimed it: the end of a hot-pixel plateau or the decay of a flare crossing the
  threshold is evidence of an event the detector already counted, not a false alarm.
  Bounded stays IoU-based and is never affected.
- A **field-wide** label (`star_id` None) accepts many detections from one detector:
  the first is its TP; each later eligible detection is **corroborating** and
  counts as neither TP nor FP. Only detections eligible for nothing (overlap, single-cadence
  tolerance or edge tolerance) are FP.
- **Fragments.** A detection that overlaps an already-claimed star-level label, finds
  no unclaimed label to take (the free label always wins), and so loses the one-to-one
  race for the claimed one, is **corroborating** for that label, graded the
  same as an overlapping duplicate on a field-wide label and as an edge detection on a
  claimed label. Which single detection earns the tp is unchanged (greedy by score, then
  the best label by IoU), so `tp_bounded`, `tp_touched` and recall do not move. A
  fragment that overlaps several claimed labels corroborates the one with the highest
  IoU (ties by `label_id`); a claimed field-wide label it is eligible for takes
  precedence, as before. Why: precision measures whether a detection points at something
  real, and the flare's decay re-crossing the threshold after the onset was found points
  at the flare; booking it a false alarm made a detector that merges fragments look more
  precise than one that reports them, regardless of whether either sees the event. It is
  not extra credit either: a fragment is neither tp nor fp, adds nothing to recall, and
  is omitted from the confusion table. `false_alarm` keeps meaning "no truth here": only
  a detection that matches, corroborates or abuts nothing is FP.
- **Unobservable labels.** A label with `footprint_samples == 0` is dropped before
  matching: it is never matched, never an FN, and not in recall's denominator. A detection
  over one is treated as if the label were absent, so it is matched to something else if
  it can be and is otherwise an FP (a detector that flags a spot where nothing was
  injected is wrong, whatever the truth file says about that spot). The count is reported
  per detector and kind as `unobservable` (the `unobs` column) so the exclusion is visible
  rather than silent, and the confusion table has no `missed` row for such a label. A label
  with no `footprint_samples` key (a run simulated before #5) is treated as observable, and
  `score` prints a one-line warning to stderr saying how many labels lack it.
- Per detector and per `kind`: `tp_bounded`, `tp_touched`, `corroborating`, `fp`,
  `fn`, `unobservable`. `tp = tp_bounded + tp_touched`; precision = tp / (tp + fp); recall =
  tp / (tp + fn); F1 from those; `bounded_rate = tp_bounded / tp` (how well the
  detector spans what it finds). FP is counted against the kind the detector
  guessed if it guessed, else against `unknown`.
- A **confusion** table (truth kind by detected-as) is reported alongside, with
  bounded and touched matches and `missed` rows, so that "flagged a cloud as a
  transit" is visible, not folded into FP. Corroborating detections are omitted.
- Pulsations are continuous and are scored only on a "did you notice this star
  varies" basis in v0 (one label per star per night).

Why: on the first run, 281 of 312 "false positives" overlapped a real injected
event, because IoU >= 0.3 made an onset hit on a 47-minute flare both an FN and an
FP, and one-label-one-detection booked 216 of 217 stars that flagged the same cloud
as false alarms. The scoreboard must separate "did not see it" from "saw it but
spanned it wrong" and "everyone saw it", so the scorer changed, not the detector.

## Interfaces that exist from day one

```
skywatch simulate --seed 42 --stars 200 --nights 3 --out runs/
skywatch detect   --run runs/<id> --detector rolling_z
skywatch score    --run runs/<id>
skywatch report   --run runs/<id>          # text table now, site later
skywatch planets  --run runs/<id> [--min-events N] [--min-snr X]
```

`planets` is the truth-side yield tool, like `score` and `report`; detectors never import
it. It prints one row per injected planet from the stored tables: the number of in-night
`transit` labels (`events`, clipped labels included) and the stacked matched-filter S/N a
periodic search could reach. Per event, S/N = `depth * sqrt(sum(profile(t_i)^2)) /
sigma_tot`, summed over the star's observed samples inside the label's span, where
`profile` is the injector's box with linear ramps (`ramp = 0.1 * duration_s`) and
`sigma_tot = sqrt(noise_floor^2 + 1/flux0)` is the per-sample relative noise. A label with
`footprint_samples == 0` is an event with S/N 0. `snr_stack` is the quadrature sum over a
planet's events, `snr_max` the largest one. Instrument-flagged samples (`quality` bit 0)
count like any other, so the S/N reads as an upper bound for a detector that drops them. The filters hide rows only; the summary line
always counts the whole run. It is how a periodic detector's acceptance targets are named
from the run itself rather than estimated.

## Non-goals for v0

No real catalogs, no real survey data, no deployment, no dashboard. The
dashboard comes after the first measured finding exists. Hosting is undecided
and the engine must not assume one (hence Parquet + JSON manifest, not a live
server).

## Open questions (decided by the agent, logged in docs/log/)

- Whether `flux` should be in magnitudes for display. Engine stays in flux.
- Night-to-night continuity of glitches (does a hot pixel persist?). v0: no.
- Red noise. v0: no.
