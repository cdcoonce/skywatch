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
   guard. A pulsator with one real transit can borrow a 1 h noise dip as the second
   event `min_events` asks for (dev seed 44 star 44; about 1 in 120 such stars on 240
   synthetic cases, and nothing on `main`). The grid, statistic, guard, detrending and
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
   read 3.72 and 4.11 fitted alone and -1.29 jointly, cause not established. The margins are thin
   on the binary side (eclipses at 1.7 and 1.8, and 0.70 on the unseen seeds; a planet at 0.93 on
   development, 0.12 on the unseen seeds).

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
   next paragraph; fast pulsators the detrending leaves a 2-event box on: issue #44). Development, `main` to the branch (33 runs):
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
   development true positives are lost, all of them events that follow a flare (seed 12 star 32 and
   seed 62 star 38, flare stars whose box overlaps one planet event and is mostly residue; seed 61
   star 56, a constant star at local S/N 7.04); the other 8 flare-host planets keep every grade.
   The flare bar of 7.2 stays: without it seed 96 star 10 (local S/N 7.13, clean 7.93) is claimed
   again. 9 of the 45 flare claims remain, flares the clip never seeded (an oracle that excludes 6
   e-foldings of every true flare leaves none). `single_transit` is identical span for span on 27
   runs; the time of `bls` is unchanged.
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
   they do not narrow the margin. The statistic, threshold, guards and limits are
   in the `skywatch/detectors/single_transit.py` docstring.
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
