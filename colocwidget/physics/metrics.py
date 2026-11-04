"""Colocalization coefficients, each with what it actually measures.

Two families, which answer different questions and are often confused:

* **Correlation** -- do the intensities rise and fall together?
  Pearson's r (PCC), Spearman's rank rho (SRCC), Li's intensity correlation
  quotient (ICQ), van Steensel's cross-correlation function (CCF).
* **Co-occurrence** -- what fraction of one signal lies where the other is?
  Manders' M1 and M2 (thresholded, as Costes et al. define them), Manders'
  overlap coefficient (MOC) and its split k1, k2.

Plus an **object-based** measure that works on spots rather than pixels (nearest-
neighbour distance between local maxima, against the chance level), and the
**ground truth**, computed from the labels themselves rather than an image.

Formulas follow Manders et al., J. Microsc. 169, 375 (1993); Costes et al.,
Biophys. J. 86, 3993 (2004); Li et al., J. Neurosci. 24, 4070 (2004); van Steensel
et al., J. Cell Sci. 109, 787 (1996); Dunn, Kamocka & McDonald, Am. J. Physiol.
Cell Physiol. 300, C723 (2011); Adler & Parmryd, Cytometry A 77, 733 (2010); and
Fiji's Coloc 2, whose thresholded Manders' definition is used here.

Everything here is plain numpy and takes 1D arrays of pixel values (already
masked to the region of interest) unless it says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

THRESHOLD_MODES = ("Costes", "Otsu", "fixed fraction", "none")


# --- correlation ------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """``r = sum (a - a_)(b - b_) / sqrt(sum (a - a_)^2 sum (b - b_)^2)``."""
    if a.size < 3:
        return float("nan")
    da = a - a.mean()
    db = b - b.mean()
    den = math.sqrt(float(da @ da) * float(db @ db))
    return float(da @ db) / den if den > 0 else float("nan")


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(x.size)
    ranks[order] = np.arange(x.size, dtype=float)
    # Tie correction: give every member of a run of equal values the run's mean rank.
    starts = np.r_[0, np.nonzero(np.diff(xs))[0] + 1]
    ends = np.r_[starts[1:], x.size]
    if starts.size < x.size:
        mean_rank = (starts + ends - 1) / 2.0
        ranks[order] = np.repeat(mean_rank, ends - starts)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson's r of the ranks: monotone, not just linear, association."""
    return pearson(_ranks(a), _ranks(b))


def icq(a: np.ndarray, b: np.ndarray) -> float:
    """Li's ICQ: fraction of pixels whose deviations share a sign, minus 0.5.

    +0.5 dependent, 0 random, -0.5 segregated.
    """
    prod = (a - a.mean()) * (b - b.mean())
    if prod.size == 0:
        return float("nan")
    return float((prod > 0).mean() - 0.5)


def nmdp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized mean deviation product per pixel, in [-1, 1].

    ``(a - a_)(b - b_) / ((max a - a_)(max b - b_))``: the pixels Pearson's r is
    summed from, so a map of it shows *where* the correlation comes from.
    (Jaskolski, Mulle & Manzoni, J. Neurosci. Methods 146, 42, 2005.)
    """
    da, db = a - a.mean(), b - b.mean()
    den = float(da.max() * db.max())
    return da * db / den if den > 0 else np.zeros_like(a)


def cross_correlation(a2d: np.ndarray, b2d: np.ndarray, mask: np.ndarray, max_shift_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Van Steensel's CCF: Pearson's r as channel 2 is shifted along x.

    A true colocalization peaks at zero shift with a width set by the PSF; a
    systematic offset (chromatic shift) moves the peak; exclusion gives a dip.
    """
    shifts = np.arange(-max_shift_px, max_shift_px + 1)
    out = np.full(shifts.size, np.nan)
    w = a2d.shape[1]
    for i, s in enumerate(shifts):
        if abs(s) >= w - 2:
            continue
        if s >= 0:
            aa, bb, mm = a2d[:, : w - s], b2d[:, s:], mask[:, : w - s] & mask[:, s:]
        else:
            aa, bb, mm = a2d[:, -s:], b2d[:, : w + s], mask[:, -s:] & mask[:, : w + s]
        out[i] = pearson(aa[mm], bb[mm])
    return shifts, out


# --- co-occurrence ----------------------------------------------------------------


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Manders' overlap coefficient and its split.

    ``MOC = sum ab / sqrt(sum a^2 sum b^2)``, ``k1 = sum ab / sum a^2``,
    ``k2 = sum ab / sum b^2``. Not mean-subtracted, so it depends on background
    and on the ratio of the two intensities, unlike Pearson's r.
    """
    ab = float(a @ b)
    aa = float(a @ a)
    bb = float(b @ b)
    moc = ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")
    return moc, (ab / aa if aa > 0 else float("nan")), (ab / bb if bb > 0 else float("nan"))


def manders(a: np.ndarray, b: np.ndarray, ta: float, tb: float) -> tuple[float, float]:
    """Thresholded Manders' coefficients (Coloc 2's tM1, tM2).

    ``M1 = sum a_i [a_i > Ta and b_i > Tb] / sum a_i [a_i > Ta]``: the fraction of
    channel 1's signal that lies where channel 2 is present; M2 is the converse.
    """
    above_a = a > ta
    above_b = b > tb
    both = above_a & above_b
    sa = float(a[above_a].sum())
    sb = float(b[above_b].sum())
    m1 = float(a[both].sum()) / sa if sa > 0 else float("nan")
    m2 = float(b[both].sum()) / sb if sb > 0 else float("nan")
    return m1, m2


# --- thresholds -------------------------------------------------------------------


def otsu(x: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: the split that maximizes between-class variance."""
    if x.size == 0:
        return 0.0
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(counts)
    w1 = w0[-1] - w0
    s0 = np.cumsum(counts * centres)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = (w0 * w1 * (m0 - m1) ** 2)[:-1]
    # An empty gap between two populations is a plateau of equal scores: take its middle.
    best = np.nonzero(between >= between.max() * (1 - 1e-9))[0]
    return float(edges[int(best[len(best) // 2]) + 1])


def orthogonal_regression(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Total least squares fit ``b = m a + c``: neither channel is the "x" one."""
    ma, mb = a.mean(), b.mean()
    sxx = float(((a - ma) ** 2).mean())
    syy = float(((b - mb) ** 2).mean())
    sxy = float(((a - ma) * (b - mb)).mean())
    if sxy == 0:
        return 0.0, float(mb)
    m = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4 * sxy * sxy)) / (2 * sxy)
    return m, float(mb - m * ma)


@dataclass
class Thresholds:
    ta: float
    tb: float
    mode: str
    slope: float = float("nan")
    intercept: float = float("nan")
    note: str = ""


def costes_threshold(a: np.ndarray, b: np.ndarray) -> Thresholds:
    """Costes' automatic threshold.

    Fit ``b = m a + c`` (orthogonal regression). Walk a threshold ``T`` down channel
    1 (and ``mT + c`` down channel 2) until the pixels *not* above both thresholds
    have no positive correlation left: everything above is then the correlated
    population. Bisection on T, which works because r_below falls with T.
    """
    if a.size > 40_000:
        # The threshold is a property of the joint distribution, not of every pixel:
        # a fixed subsample estimates it as well and keeps large images interactive.
        pick = np.random.default_rng(0).choice(a.size, 40_000, replace=False)
        a, b = a[pick], b[pick]
    m, c = orthogonal_regression(a, b)
    if not (m > 0 and math.isfinite(m)):
        t = otsu(a), otsu(b)
        return Thresholds(t[0], t[1], "Costes", m, c, "no positive regression slope: Costes undefined, used Otsu")
    lo, hi = float(a.min()), float(a.max())

    def r_below(t: float) -> float:
        below = ~((a > t) & (b > m * t + c))
        if below.sum() < 10:
            return -1.0
        r = pearson(a[below], b[below])
        return r if math.isfinite(r) else -1.0

    if r_below(lo) > 0:
        # Even the lowest threshold leaves correlation below it: everything correlates.
        return Thresholds(lo, m * lo + c, "Costes", m, c)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if r_below(mid) > 0:
            hi = mid
        else:
            lo = mid
    return Thresholds(hi, m * hi + c, "Costes", m, c)


def thresholds(a: np.ndarray, b: np.ndarray, mode: str, fraction: float = 0.2) -> Thresholds:
    if mode == "Costes":
        return costes_threshold(a, b)
    m, c = orthogonal_regression(a, b) if a.size > 2 else (float("nan"), float("nan"))
    if mode == "Otsu":
        return Thresholds(otsu(a), otsu(b), mode, m, c)
    if mode == "fixed fraction":
        return Thresholds(fraction * float(np.percentile(a, 99.9)), fraction * float(np.percentile(b, 99.9)),
                          mode, m, c)
    return Thresholds(0.0, 0.0, "none", m, c)


def costes_significance(a2d: np.ndarray, b2d: np.ndarray, mask: np.ndarray, block_px: int,
                        trials: int = 50, seed: int = 0) -> tuple[float, float, float]:
    """Costes' randomization test.

    Scramble channel 1 in blocks the size of the PSF (so each block keeps its own
    internal, optical correlation) and recompute r. The P-value is the fraction of
    scrambles whose r is *below* the observed one: P > 0.95 means the observed r
    is unlikely to arise from the two patterns by chance.
    Returns (P, mean r of scrambles, sd of scrambles).
    """
    rng = np.random.default_rng(seed)
    k = max(int(block_px), 1)
    h, w = a2d.shape
    nh, nw = h // k, w // k
    if nh * nw < 4:
        return float("nan"), float("nan"), float("nan")
    a = a2d[: nh * k, : nw * k]
    b = b2d[: nh * k, : nw * k]
    mm = mask[: nh * k, : nw * k]
    observed = pearson(a[mm], b[mm])
    blocks = a.reshape(nh, k, nw, k).transpose(0, 2, 1, 3).reshape(nh * nw, k, k)
    rs = np.empty(trials)
    for i in range(trials):
        shuffled = blocks[rng.permutation(nh * nw)].reshape(nh, nw, k, k).transpose(0, 2, 1, 3).reshape(nh * k, nw * k)
        rs[i] = pearson(shuffled[mm], b[mm])
    rs = rs[np.isfinite(rs)]
    if rs.size == 0 or not math.isfinite(observed):
        return float("nan"), float("nan"), float("nan")
    return float((rs < observed).mean()), float(rs.mean()), float(rs.std())


# --- all pixel coefficients at once ----------------------------------------------


@dataclass
class Coefficients:
    pcc: float
    srcc: float
    icq: float
    moc: float
    k1: float
    k2: float
    m1: float
    m2: float
    thresholds: Thresholds
    n_pixels: int

    KEYS = ("pcc", "srcc", "icq", "moc", "k1", "k2", "m1", "m2")
    LABELS = {
        "pcc": "Pearson's r (PCC)",
        "srcc": "Spearman's rho",
        "icq": "Li's ICQ",
        "moc": "Manders' overlap (MOC)",
        "k1": "overlap k1",
        "k2": "overlap k2",
        "m1": "Manders' M1",
        "m2": "Manders' M2",
    }

    def get(self, key: str) -> float:
        return getattr(self, key)


def coefficients(a: np.ndarray, b: np.ndarray, threshold_mode: str = "Costes", fraction: float = 0.2,
                 rank: bool = True, fixed: Thresholds | None = None) -> Coefficients:
    """Every pixel coefficient. ``fixed`` reuses thresholds chosen elsewhere."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if fixed is not None:
        t = fixed
    else:
        t = thresholds(a, b, threshold_mode, fraction) if a.size > 2 else Thresholds(0, 0, threshold_mode)
    m1, m2 = manders(a, b, t.ta, t.tb)
    moc, k1, k2 = overlap(a, b)
    return Coefficients(
        pearson(a, b), spearman(a, b) if rank else float("nan"), icq(a, b), moc, k1, k2, m1, m2, t, a.size,
    )


# --- objects ------------------------------------------------------------------------


def local_maxima(img: np.ndarray, threshold: float, mask: np.ndarray | None = None) -> np.ndarray:
    """(row, col) of pixels above threshold and strictly above their 8 neighbours,
    after a 3x3 binomial smooth that keeps single-pixel noise from counting as spots."""
    p = np.pad(img, 1, mode="edge")
    w = {-1: 1.0, 0: 2.0, 1: 1.0}
    s = sum(w[dy] * w[dx] * p[1 + dy: 1 + dy + img.shape[0], 1 + dx: 1 + dx + img.shape[1]]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 16.0
    q = np.pad(s, 1, mode="constant", constant_values=-np.inf)
    peak = s > threshold
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            peak &= s > q[1 + dy: 1 + dy + s.shape[0], 1 + dx: 1 + dx + s.shape[1]]
    if mask is not None:
        peak &= mask
    return np.argwhere(peak)


@dataclass
class SpotColoc:
    n1: int
    n2: int
    frac1: float  # fraction of ch1 spots with a ch2 spot within d
    frac2: float
    chance1: float  # expected by chance from ch2's spot density alone
    chance2: float
    distance_um: float


def spot_colocalization(a2d: np.ndarray, b2d: np.ndarray, mask: np.ndarray, ta: float, tb: float,
                        pixel_um: float, distance_um: float) -> SpotColoc:
    """Object-based colocalization: nearest-neighbour distance between spot centres.

    Chance level: with ch2 spots scattered at density rho, the probability that one
    lies within d of a given point is ``1 - exp(-rho pi d^2)``.
    """
    s1 = local_maxima(a2d, ta, mask)
    s2 = local_maxima(b2d, tb, mask)
    area = float(mask.sum()) * pixel_um**2
    d_px = distance_um / pixel_um

    def frac(p, q):
        if len(p) == 0:
            return float("nan")
        pad = np.zeros((len(p), 1))
        return fraction_within(np.hstack([p, pad]).astype(float), np.hstack([q, np.zeros((len(q), 1))]).astype(float), d_px)

    def chance(q):
        return 1 - math.exp(-len(q) / area * math.pi * distance_um**2) if area > 0 else float("nan")

    return SpotColoc(len(s1), len(s2), frac(s1, s2), frac(s2, s1), chance(s2), chance(s1), distance_um)


# --- the truth, from the labels -----------------------------------------------------


def fraction_within(p: np.ndarray, q: np.ndarray, d: float) -> float:
    """Fraction of points ``p`` with at least one point of ``q`` within distance d.

    Exact: a cell list of side d, so only the 27 neighbouring cells are searched.
    """
    if len(p) == 0:
        return float("nan")
    if len(q) == 0:
        return 0.0
    lo = np.minimum(p.min(0), q.min(0)) - d
    kp = np.floor((p - lo) / d).astype(np.int64)
    kq = np.floor((q - lo) / d).astype(np.int64)
    dims = np.maximum(kp.max(0), kq.max(0)) + 3

    def key(k):
        return (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]

    order = np.argsort(key(kq), kind="stable")
    qs, keys_q = q[order], key(kq)[order]
    found = np.zeros(len(p), bool)
    for off in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
        todo = np.nonzero(~found)[0]
        if todo.size == 0:
            break
        kk = key(kp[todo] + off)
        start = np.searchsorted(keys_q, kk, "left")
        end = np.searchsorted(keys_q, kk, "right")
        j = 0
        live = end > start
        while live.any():
            idx = todo[live]
            cand = start[live] + j
            hit = ((p[idx] - qs[cand]) ** 2).sum(1) <= d * d
            found[idx[hit]] = True
            j += 1
            live = live & (start + j < end)
            live[live] &= ~found[todo[live]]
    return float(found.mean())


def voxel_pearson(p: np.ndarray, q: np.ndarray, voxel: float, volume_voxels: float) -> float:
    """Pearson's r of the two label densities counted in voxels of side ``voxel``.

    Computed sparsely: only occupied voxels are stored, and the empty majority
    enters through the total count. This is what a perfect, noise-free microscope
    with that voxel size would report.
    """
    if len(p) == 0 or len(q) == 0 or volume_voxels <= 0:
        return float("nan")
    lo = np.minimum(p.min(0), q.min(0))
    kp = np.floor((p - lo) / voxel).astype(np.int64)
    kq = np.floor((q - lo) / voxel).astype(np.int64)
    dims = np.maximum(kp.max(0), kq.max(0)) + 1

    def key(k):
        return (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]

    ua, ca = np.unique(key(kp), return_counts=True)
    ub, cb = np.unique(key(kq), return_counts=True)
    _, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    n = float(volume_voxels)
    sab = float((ca[ia].astype(float) * cb[ib]).sum())
    sa, sb = float(ca.sum()), float(cb.sum())
    saa, sbb = float((ca.astype(float) ** 2).sum()), float((cb.astype(float) ** 2).sum())
    cov = sab - sa * sb / n
    var = (saa - sa * sa / n) * (sbb - sb * sb / n)
    return cov / math.sqrt(var) if var > 0 else float("nan")


@dataclass
class Truth:
    """Colocalization of the molecules themselves, at a stated distance."""

    distance_um: float
    m1: float  # fraction of ch1 labels within d of a ch2 label
    m2: float
    pcc: float  # voxel Pearson's r at voxel size d
    n1: int
    n2: int
    by_compartment: dict[int, tuple[float, float]] = field(default_factory=dict)  # true fractions of each channel

    def compartment_overlap(self) -> tuple[float, float]:
        """Share of each channel in compartments the other channel also occupies."""
        if not self.by_compartment:
            return float("nan"), float("nan")
        s1 = sum(f1 for f1, f2 in self.by_compartment.values() if f2 > 0)
        s2 = sum(f2 for f1, f2 in self.by_compartment.values() if f1 > 0)
        return s1, s2
