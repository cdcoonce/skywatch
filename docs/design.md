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
scores       detector, kind, tp, fp, fn, precision, recall, f1
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
   window, flag |z| above a threshold. Baseline.
2. `median_residual` — subtract a running median, flag residual outliers. Should
   beat rolling_z on flares and cosmic rays.
3. `field_common_mode` — estimate the field-wide multiplicative trend from the
   ensemble, flag spans where it departs from 1. The only detector that can
   name a cloud.
4. `bls` — box least squares for periodic transits across nights. Should be the
   only thing that finds shallow transits.
5. Later: matched flare filter, a learned model, an LLM-judge for comparison.

## Scoring

- A detection and a truth label **match** when they are on the same star (or
  the label is field-wide) and their time spans overlap with IoU >= 0.3, or the
  detection is within one cadence of a single-sample label.
- Matching is greedy by descending detection score; one label matches at most
  one detection, one detection at most one label.
- Per detector and per `kind`: TP, FP, FN, precision, recall, F1. FP is counted
  against the kind the detector guessed if it guessed, else against `unknown`.
- A **confusion** table (truth kind by detected-as) is reported alongside, so
  that "flagged a cloud as a transit" is visible, not folded into FP.
- Pulsations are continuous and are scored only on a "did you notice this star
  varies" basis in v0 (one label per star per night).

## Interfaces that exist from day one

```
skywatch simulate --seed 42 --stars 200 --nights 3 --out runs/
skywatch detect   --run runs/<id> --detector rolling_z
skywatch score    --run runs/<id>
skywatch report   --run runs/<id>          # text table now, site later
```

## Non-goals for v0

No real catalogs, no real survey data, no deployment, no dashboard. The
dashboard comes after the first measured finding exists. Hosting is undecided
and the engine must not assume one (hence Parquet + JSON manifest, not a live
server).

## Open questions (decided by the agent, logged in docs/log/)

- Whether `flux` should be in magnitudes for display. Engine stays in flux.
- Night-to-night continuity of glitches (does a hot pixel persist?). v0: no.
- Red noise. v0: no.
