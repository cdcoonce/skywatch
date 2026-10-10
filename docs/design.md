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
  cosmic ray); that is not counted.
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
  is not edge-eligible and stays unmatched, as before. Among eligible labels any
  still-unclaimed one is preferred over an already-claimed one, even a nearer claimed
  one; within the chosen group the nearest edge wins, then the lowest `label_id`. The
  grade is **touched** if the label is still unclaimed, and **corroborating** if this
  detector already claimed it: the end of a hot-pixel plateau or the decay of a flare crossing the
  threshold is evidence of an event the detector already counted, not a false alarm.
  Bounded stays IoU-based and is never affected.
- A **field-wide** label (`star_id` None) accepts many detections from one detector:
  the first is its TP; each later eligible detection is **corroborating** and
  counts as neither TP nor FP. Only detections eligible for nothing (overlap, single-cadence
  tolerance or edge tolerance) are FP.
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
