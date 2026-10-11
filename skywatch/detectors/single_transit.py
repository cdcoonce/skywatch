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
tails. Masked samples are still *observed*: they count when the reported span is cut, so a
flare on a transit does not move it.

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
same night: on four planet-free 20-night runs it gives 14 to 40 false alarms at S/N 7 and 0 to 6
at 10 where the contrast form gives none at either. The best box per night is reported when its S/N reaches the threshold; its :class:`Detection` spans the
observed samples the box covers (half-open: ``t_end = t_last + cadence_s``).

Threshold
---------
``snr_threshold = 6``: the lowest level at which the false-alarm count, a detection that overlaps
no truth label of any kind (the scorer's fp), is at most 5 on each of four planet-free 200-star
20-night runs (``planet_rate=0``, seeds 41 to 44): 0, 0, 0 and 0 at 6; 3, 3, 1 and 6 at 5; 0 on
every run at 7 to 10. Detections on those runs that sit on a real eclipse, flare, cloud or
pulsation are not false alarms in that sense but are not transits either: at 6 they number 10, 2,
1 and 3, and the eclipse and pulsation ones are the guards' residue.

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
  eclipse->transit confusion is a few per 20-night run, not zero.
* Only the best box per night is reported: two dips in one night are one detection.
* A dip that fills most of a night leaves little reference (``min_out`` 30) and a night
  median taken inside the dip; the S/N is then the lower, contrast, one.
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
        snr_threshold: float = 6.0,
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
        t_obs = c.t
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
            drop_obs = np.zeros(len(t_obs), dtype=bool)
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
                drop_obs |= in_box(t_obs, m)
            t, flux, night = t[~drop], flux[~drop], night[~drop]
            t_obs = t_obs[~drop_obs]
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
            seen = t_obs[(t_obs >= b.t_start) & (t_obs <= b.t_end)]
            out.append(
                Detection(
                    detector=self.name,
                    det_id=f"{self.name}-s{star_id}-{k}",
                    star_id=star_id,
                    t_start=int(seen[0]),
                    t_end=int(seen[-1]) + cadence_s,
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
