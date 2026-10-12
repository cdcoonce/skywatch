"""One-event box search per night: the single-transit detector.

Most planets a survey catches transit once, and a transit at the edge of a night has flux on
one side only, so neither a periodic search (``bls`` needs two events) nor a detector that
compares a dip with its surroundings on both sides can see it. This detector searches each
night of each star on its own for the one box-shaped dip that best explains it and reports it
when its S/N reaches ``snr_threshold`` (``kind_guess="transit"``, ``score`` = the S/N).

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

Limits
------
* A second strong event stays in the residual of the guard: a star with two single events of
  S/N 10 or more each is read as a sinusoid and not searched (the residual power of a dip grows
  with its S/N squared; at S/N 9 it is at the level of the false-alarm limit). A periodic
  planet's other events do the same.
* A planet whose transit overlaps or abuts the masked eclipses of its binary host is lost with
  them (seed7-200s-20n star 99: the planet's transit starts 20 minutes after an eclipse ends and
  the mask, 21 h of every 44 h cycle, covers it). Other eclipse residue gets through: the
  eclipse->transit confusion on the held-out 20-night runs is 0 to 7 per run at the default
  (12 over six runs), not zero.
* A star the guards reject is lost with its planet when the star is a sinusoidal or flare
  variable: held-out seed22 star 81 and seed25 stars 39 and 52 are found at S/N 9.5 to 15.6
  with ``guards=False`` and dropped with them on. A planet-only host whose guard ratio
  (see Guards) crosses 1 is the same loss.
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
``_common_mode``, ``_prepare``, ``_pulsating``, ``_search`` and the fields of ``Box`` and
``Candidate.mask``; a rename in ``bls`` breaks it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from skywatch.detectors import register
from skywatch.detectors.bls import (
    _MAD_SCALE,
    BlsDetector,
    Box,
    _Clean,
    _renormalise_nights,
    in_box,
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
        self.snr_threshold = snr_threshold
        self.durations_h = tuple(sorted(durations_h))
        self.min_in = min_in
        self.min_out = min_out
        self.contrast = contrast
        self.guards = guards
        self.mask_pad_h = mask_pad_h
        self._bls = BlsDetector()

    # -- search ----------------------------------------------------------------------------

    def _boxes(
        self, t: np.ndarray, flux: np.ndarray, night: np.ndarray, cadence_s: int
    ) -> list[NightBox]:
        """The best box of each night whose S/N reaches the threshold (night order)."""
        durations = tuple(h * HOUR_S for h in self.durations_h)
        out: list[NightBox] = []
        for k in np.unique(night):
            idx = np.flatnonzero(night == k)
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

    def _star(
        self,
        star_id: int,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
    ) -> list[Detection]:
        t, flux, night, _ = self._bls._prepare(c, common_mode)
        if len(t) < self.min_in + self.min_out:
            return []
        boxes = self._boxes(t, flux, night, cadence_s)
        if not boxes:
            return []
        if self.guards and self._pulsating(t, flux, night, boxes, cadence_s):
            cand = self._bls._search(c, common_mode, cadence_s, plan_mask=True)
            if cand is None or cand.rejected != "pulsation guard" or not cand.mask:
                log.info("single_transit star %s: pulsation guard, not searched", star_id)
                return []
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
                return []
            log.info(
                "single_transit star %s: eclipses masked (%d windows)", star_id, len(cand.mask)
            )
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
