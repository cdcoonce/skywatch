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
For each trial period the samples are folded into phase bins no wider than half the
shortest duration (``P / ceil(P / bin_s)``, ``bin_s = min_duration / 2``); samples are first
merged into ``bin_s / 3`` time buckets (300 s: a 5x saving at 60 s cadence, at most a
sixth of a phase bin of timing error), binned sums and counts are accumulated with ``np.bincount`` for a chunk of
periods at once, and every duration is a box sum over consecutive bins (wrapping through the
phase origin) read from a cumulative sum. A box is ``round(duration / bin_width)`` bins wide,
so the evaluated width differs from the nominal duration by at most half a bin; the reported
event windows use the evaluated width. The phase is stepped by one bin. The winning box is
then re-measured on the samples themselves (:func:`refine`).

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
  box is small against its nights and is lower otherwise.
* The predicted windows cover observed samples on at least ``min_events`` (2) distinct
  nights.
* At least ``min_events`` nights each show the dip at ``support_snr`` (2.0) on their own, so
  one bad night cannot make a candidate.
* The pulsation guard passes (below).

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

Limits
------
* Only the best box per star is reported. The strongest peak at a clearly different period
  that is not a small-integer ratio of the best period is logged at INFO
  (``skywatch.detectors.bls``) as an alternative; a second planet is out of scope. A star
  whose strongest box is not a planet (an eclipsing binary, a pulsator) is rejected whole,
  so a planet behind it is lost too.
* A transit longer than the 6 h grid maximum is fit by the closest box; a transit that
  covers more than half of a night makes that night's first-pass median the in-transit
  level (the out-of-box renormalisation repairs the depth, not the search).
* Two single events of different planets six days apart look like one period.
* Output spans are half-open: ``t_end = t_last + cadence_s``.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace

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
_CHUNK_ELEMENTS = 1_000_000  # periods per chunk = this // n samples (keeps arrays cache-sized)


@dataclass(frozen=True)
class _Clean:
    """One star's quality-clean samples: times, night-median-normalised flux, night index."""

    t: np.ndarray
    flux: np.ndarray
    night_of: np.ndarray


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
    ``"snr"``, ``"local snr"``, ``"min_events"``, ``"support"`` or ``"pulsation guard"``.
    `events` are the predicted in-night windows ``(t_start, t_end, night)`` (empty once
    rejected); `excess` is the guard's strongest residual-sinusoid power (None if not
    computed); `alternative` is ``(period_s, power / best power)`` of the strongest
    unrelated peak.
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
        return _Clean(t, _divide_by_night_median(flux, night_of), night_of)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, flux, night) used by the search: common mode removed, renormalised, clipped."""
        flux = c.flux
        if common_mode is not None:
            grid, cm = common_mode
            flux = _divide_by_night_median(flux / cm[np.searchsorted(grid, c.t)], c.night_of)
        ok = np.ones(len(flux), dtype=bool)
        for k in range(int(c.night_of[-1]) + 1):
            m = c.night_of == k
            ok[m] = ~positive_outliers(flux[m], self.clip_sigma, self.tail_sigma)
        return c.t[ok], flux[ok], c.night_of[ok]

    # -- guard ---------------------------------------------------------------------------

    def _peak_power(
        self, t: np.ndarray, x: np.ndarray, sigma: float, cadence_s: int
    ) -> tuple[float, float]:
        """(strongest normalised Fourier power of `x` in the guard band, its false-alarm level).

        `x` (zero-mean flux) is summed into ``max(cadence_s, 300)`` s bins on one regular grid
        (empty bins are zero) and Fourier transformed, zero-padded fourfold. For white noise
        the normalised power ``|X(f)|**2 / (n sigma**2)`` of each independent frequency is
        exponentially distributed with unit mean, so over ``M = (f_max - f_min) * baseline``
        independent frequencies the level exceeded by chance with probability ``guard_fap``
        per star is ``ln(M / guard_fap)``. A sinusoid of amplitude A over n samples has
        power ``n A**2 / (4 sigma**2)`` and gains ``2 * power`` in chi-square. `t` and `x`
        hold only the samples that take part (the others are simply absent).
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
            return 0.0, math.inf
        power = float((np.abs(spec[band]) ** 2).max() / (len(t) * sigma**2))
        independent = max(1.0, (f_hi - self.guard_f_min_per_d) * (t[-1] - t[0]) / DAY_S)
        assert self.guard_fap is not None
        return power, math.log(independent / self.guard_fap)

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
        depth = float(flux[outside].mean() - flux[core].mean()) if core.any() else 0.0
        resid = flux - flux[outside].mean() + depth * core
        r_power, r_crit = self._peak_power(t[keep], resid[keep], sigma, cadence_s)
        n, n_in = len(t), box.n_in
        box_chi2 = box.depth**2 * n_in * (n - n_in) / (n * sigma**2)
        s_power, _ = self._peak_power(t, flux - 1.0, sigma, cadence_s)
        return (r_power >= r_crit or 2.0 * s_power >= self.guard_ratio * box_chi2), r_power

    # -- main --------------------------------------------------------------------------

    def _search(
        self,
        c: _Clean,
        common_mode: tuple[np.ndarray, np.ndarray] | None,
        cadence_s: int,
    ) -> Candidate | None:
        """The star's best box with its S/N and, if it is not reportable, why not."""
        t, flux, night = self._prepare(c, common_mode)
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
        snr_local = float(depth_k @ weight_k / (sigma * math.sqrt(weight_k.sum())))
        supported = int((z_k >= self.support_snr).sum())

        def reject(why: str, excess: float | None = None) -> Candidate:
            return Candidate(box, sigma, snr, [], why, excess, alt, snr_local, supported)

        if snr < self.snr_threshold:
            return reject("snr")
        if snr_local < self.snr_threshold:
            return reject("local snr")
        events = self._events(c.t, c.night_of, box, cadence_s)
        if len({n for _, _, n in events}) < self.min_events:
            return reject("min_events")
        if supported < self.min_events:
            return reject("support")
        r = None
        if self.guard_fap is not None:
            pulsating, r = self._pulsating(t, flux, sigma, cadence_s, box)
            if pulsating:
                return reject("pulsation guard", r)
        return Candidate(box, sigma, snr, events, None, r, alt, snr_local, supported)

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
        cand = self._search(c, common_mode, cadence_s)
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
            return []
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
