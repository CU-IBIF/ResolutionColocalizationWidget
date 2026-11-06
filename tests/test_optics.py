# Copyright (c) 2026, Creighton University
# SPDX-License-Identifier: BSD-3-Clause
#
# DISCLAIMER: This software is provided "AS IS", without warranty or guarantee
# of any kind, express or implied, including but not limited to merchantability,
# fitness for a particular purpose, accuracy, and non-infringement. It is NOT FIT
# FOR ANY PURPOSE, STATED OR IMPLIED. It is a simplified educational model, not
# intended for clinical, diagnostic, or regulatory use or for the operation of
# any real instrument; its results may be wrong and must be independently
# verified. In no event shall Creighton University or the contributors be liable
# for any claim, damages, or other liability arising from its use. See LICENSE.

import math

import numpy as np
import pytest

from colocwidget.physics import optics


def _fwhm_of(profile: np.ndarray, dx: float) -> float:
    half = profile.max() / 2
    above = np.nonzero(profile >= half)[0]
    lo, hi = above[0], above[-1]
    # Linear interpolation at both half-maximum crossings.
    left = lo - (profile[lo] - half) / (profile[lo] - profile[lo - 1])
    right = hi + (profile[hi] - half) / (profile[hi] - profile[hi + 1])
    return (right - left) * dx


@pytest.mark.parametrize("fwhm", [0.05, 0.2, 0.5])
def test_the_psf_has_the_fwhm_it_was_asked_for(fwhm):
    dx = 0.005
    h = optics.psf(512, dx, fwhm)
    assert h.sum() == pytest.approx(1.0)
    assert _fwhm_of(h[256], dx) == pytest.approx(fwhm, rel=0.03)


def test_widefield_fwhm_is_the_airy_value():
    assert optics.widefield_fwhm_um(0.52, 1.4) == pytest.approx(0.191, abs=0.001)
    assert optics.lateral_fwhm_um("widefield", 0.52, 1.4) == pytest.approx(0.191, abs=0.001)


def test_modalities_rank_by_resolution():
    w = {m: optics.lateral_fwhm_um(m, 0.52, 1.4, 0.05) for m in optics.MODALITIES}
    assert w["STED"] < w["pixel reassignment (Airyscan-type)"] < w["confocal (1 AU)"] < w["widefield"]


def test_sted_cannot_be_wider_than_the_confocal_spot():
    assert optics.lateral_fwhm_um("STED", 0.52, 1.4, sted_fwhm_um=1.0) < 0.2


def test_the_axial_section_is_realistic_for_1au():
    # ~0.6-0.9 um for a 1.4 NA oil objective at 1 AU.
    assert 0.5 < optics.axial_section_um(0.52, 1.4, 1.518) < 1.0
    assert optics.axial_section_um(0.52, 0.75, 1.0) > optics.axial_section_um(0.52, 1.4, 1.518)


def test_widefield_keeps_every_plane_and_a_pinhole_rejects_them():
    dz = np.array([0.0, 1.0, 5.0])
    assert (optics.plane_weight(dz, False, 0.8) == 1).all()
    w = optics.plane_weight(dz, True, 0.8)
    assert w[0] == 1 and w[1] < 0.2 and w[2] < 0.01


def test_defocus_blur_conserves_photons():
    grid = optics.FourierGrid(64, 0.05)
    h = optics.defocus_otf(grid.radius() ** 2, 2.0, optics.cone_tan(1.4, 1.518))
    assert h[0, 0] == pytest.approx(1.0)  # the zero frequency: total light
    assert h[0, 5] < 0.5


def test_shift_phase_translates_an_image():
    n, dx = 64, 0.05
    img = np.zeros((n, n))
    img[20, 10] = 1.0
    grid = optics.FourierGrid(n, dx)
    out = np.fft.irfft2(np.fft.rfft2(img) * optics.shift_phase(grid, 5 * dx, 3 * dx), s=(n, n))
    assert np.unravel_index(np.argmax(out), out.shape) == (23, 15)


def test_box_integrate_conserves_and_handles_fractional_pixels():
    rng = np.random.default_rng(0)
    img = rng.random((100, 100))
    out = optics.box_integrate(img, 0.02, 0.065)
    assert out.shape == (30, 30)
    # Exactly the photons inside the whole pixels.
    assert out.sum() == pytest.approx(img[:97, :97].sum() + img[97, :97].sum() * 0.5 + img[:97, 97].sum() * 0.5
                                      + img[97, 97] * 0.25, rel=1e-9)
    same = optics.box_integrate(img, 0.02, 0.02)
    assert np.allclose(same, img)


def test_impossible_na_is_refused():
    with pytest.raises(ValueError, match="exceeds"):
        optics.check_na(1.4, 1.0)


def test_nyquist_is_about_lambda_over_4na():
    assert optics.nyquist_pixel_um(optics.widefield_fwhm_um(0.52, 1.4)) == pytest.approx(0.52 / (4 * 1.4), rel=0.01)
    assert math.isfinite(optics.depth_of_field_um(0.52, 1.4, 1.518))
