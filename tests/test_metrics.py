import math

import numpy as np
import pytest

from colocwidget.physics import metrics as M


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_pearson_limits(rng):
    a = rng.random(5000)
    assert M.pearson(a, a) == pytest.approx(1.0)
    assert M.pearson(a, -a) == pytest.approx(-1.0)
    assert M.pearson(a, 3 * a + 7) == pytest.approx(1.0)  # offset and gain do not matter
    assert abs(M.pearson(a, rng.random(5000))) < 0.05
    assert math.isnan(M.pearson(a, np.ones_like(a)))


def test_spearman_sees_monotone_not_just_linear(rng):
    a = rng.random(2000)
    b = a**5
    assert M.spearman(a, b) == pytest.approx(1.0)
    assert M.pearson(a, b) < 0.9


def test_spearman_ties_get_mean_ranks():
    a = np.array([1.0, 2.0, 2.0, 3.0])
    assert list(M._ranks(a)) == [0.0, 1.5, 1.5, 3.0]


def test_icq_range(rng):
    a = rng.random(4000)
    assert M.icq(a, a) == pytest.approx(0.5)
    assert M.icq(a, -a) == pytest.approx(-0.5)
    assert abs(M.icq(a, rng.random(4000))) < 0.03


def test_manders_overlap_is_not_mean_subtracted(rng):
    a = rng.random(3000)
    b = rng.random(3000)
    moc, k1, k2 = M.overlap(a, b)
    assert moc > 0.7  # unrelated but both everywhere: MOC is high anyway
    assert abs(M.pearson(a, b)) < 0.06
    assert moc == pytest.approx(math.sqrt(k1 * k2))


def test_manders_m1_m2_on_disjoint_and_identical():
    a = np.array([5.0, 5.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 5.0, 5.0])
    assert M.manders(a, b, 0.0, 0.0) == (0.0, 0.0)
    assert M.manders(a, a, 0.0, 0.0) == (1.0, 1.0)
    # Fraction of a's signal where b is present.
    b2 = np.array([1.0, 0.0, 5.0, 5.0])
    m1, m2 = M.manders(a, b2, 0.0, 0.0)
    assert m1 == pytest.approx(0.5)
    assert m2 == pytest.approx(1 / 11)


def test_otsu_splits_two_populations(rng):
    x = np.concatenate([rng.normal(10, 1, 5000), rng.normal(50, 3, 5000)])
    assert 15 < M.otsu(x) < 45


def test_costes_threshold_separates_correlated_from_background(rng):
    n = 20000
    signal = np.zeros(n)
    on = rng.random(n) < 0.2
    signal[on] = rng.uniform(20, 100, on.sum())
    a = signal + rng.normal(0, 2, n)
    b = 0.8 * signal + rng.normal(0, 2, n)
    t = M.costes_threshold(a, b)
    assert t.slope == pytest.approx(0.8, rel=0.1)
    # The threshold sits between the background and the dimmest correlated pixels.
    assert 0 < t.ta < 25
    m1, m2 = M.manders(a, b, t.ta, t.tb)
    assert m1 > 0.9 and m2 > 0.9


def test_costes_falls_back_when_there_is_no_positive_slope(rng):
    a = rng.random(3000)
    t = M.costes_threshold(a, -a)
    assert "Otsu" in t.note


def test_costes_randomization_separates_real_from_chance(rng):
    yy, xx = np.mgrid[:96, :96]
    blobs = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0) for cx, cy in rng.uniform(5, 90, (30, 2)))
    a = blobs + rng.normal(0, 0.02, blobs.shape)
    b = blobs + rng.normal(0, 0.02, blobs.shape)
    mask = np.ones_like(a, bool)
    p, rmean, _ = M.costes_significance(a, b, mask, block_px=5, trials=30)
    assert p > 0.95 and abs(rmean) < 0.3
    other = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0) for cx, cy in rng.uniform(5, 90, (30, 2)))
    p2, _, _ = M.costes_significance(a, other, mask, block_px=5, trials=30)
    assert p2 < 0.95


def test_cross_correlation_peaks_at_the_shift(rng):
    a = rng.random((40, 120))
    a = (a + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 3
    b = np.roll(a, 4, axis=1)
    shifts, r = M.cross_correlation(a, b, np.ones_like(a, bool), 10)
    assert shifts[np.nanargmax(r)] == 4


def test_nmdp_is_bounded(rng):
    a, b = rng.random(1000), rng.random(1000)
    v = M.nmdp(a, b)
    assert v.max() <= 1.0 + 1e-12 and v.min() >= -1.0 - 1e-12


def test_local_maxima_finds_isolated_spots():
    img = np.zeros((40, 40))
    for y, x in ((10, 10), (30, 25)):
        img[y - 1: y + 2, x - 1: x + 2] = 1.0
        img[y, x] = 3.0
    peaks = M.local_maxima(img, 0.2)
    assert {tuple(p) for p in peaks} == {(10, 10), (30, 25)}


def test_spot_colocalization_and_chance():
    img = np.zeros((60, 60))
    img[20, 20] = img[40, 40] = 10.0
    other = np.zeros((60, 60))
    other[21, 20] = 10.0
    s = M.spot_colocalization(img, other, np.ones_like(img, bool), 0.1, 0.1, 0.1, 0.2)
    assert (s.n1, s.n2) == (2, 1)
    assert s.frac1 == pytest.approx(0.5) and s.frac2 == pytest.approx(1.0)
    assert 0 < s.chance1 < 0.01


def test_fraction_within_matches_brute_force(rng):
    p = rng.random((400, 3))
    q = rng.random((300, 3))
    d = 0.07
    brute = (np.sqrt(((p[:, None] - q[None]) ** 2).sum(-1)).min(1) <= d).mean()
    assert M.fraction_within(p, q, d) == pytest.approx(brute)
    assert M.fraction_within(p, p[:0], d) == 0.0


def test_voxel_pearson_matches_a_dense_computation(rng):
    p = rng.random((500, 3)) * [1, 1, 0]
    q = np.concatenate([p[:250] + 0.001, rng.random((250, 3)) * [1, 1, 0]])
    v = 0.05
    # Voxels anchored at the same origin as the sparse version, covering every point.
    lo = np.minimum(p.min(0), q.min(0))
    dense_a = np.histogram2d(p[:, 0] - lo[0], p[:, 1] - lo[1], bins=21, range=[[0, 21 * v], [0, 21 * v]])[0].ravel()
    dense_b = np.histogram2d(q[:, 0] - lo[0], q[:, 1] - lo[1], bins=21, range=[[0, 21 * v], [0, 21 * v]])[0].ravel()
    assert M.voxel_pearson(p, q, v, 21 * 21) == pytest.approx(M.pearson(dense_a, dense_b), abs=1e-9)


def test_coefficients_bundle_and_fixed_thresholds(rng):
    a = rng.random(2000)
    c = M.coefficients(a, a)
    assert c.pcc == pytest.approx(1.0) and c.m1 == pytest.approx(1.0)
    fixed = M.Thresholds(0.5, 0.5, "fixed")
    d = M.coefficients(a, a, fixed=fixed)
    assert d.thresholds is fixed
    for mode in M.THRESHOLD_MODES:
        assert math.isfinite(M.coefficients(a, a * 0.5 + 0.1, mode).m1)
