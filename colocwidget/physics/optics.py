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

"""Resolution: how far apart two things must be before the microscope keeps them apart.

Colocalization is a statement about distance, and every image has a distance below
which it cannot tell "together" from "near". This module sets that distance for each
kind of microscope and turns a 3D distribution of labels into the 2D image the
microscope forms of one focal plane.

Lateral blur
    Every modality is described by the FWHM of its lateral PSF. The transfer
    function is the incoherent OTF of a circular pupil,
    ``(2/pi) [acos(s) - s sqrt(1 - s^2)]``, with ``s = f / f_c`` and the cutoff
    chosen so the PSF has that FWHM: ``f_c = 1.029 / FWHM`` (for widefield,
    ``FWHM = 0.514 lambda / NA`` and ``f_c = 2 NA / lambda`` exactly). Evaluated
    analytically, so it is right at every resolution on one grid.

Axial blur
    Out-of-focus planes are blurred further by the geometric cone of the
    objective, a Gaussian of ``sigma = |dz| tan(theta) / 2`` (the second moment of a
    defocus disc of radius ``|dz| tan theta``), on top of the in-focus OTF.

    * **Widefield** keeps every photon from every plane: a plane 3 um away is
      blurred, not removed. This is why a widefield image of a cell shows the
      apical and basal membranes lying over the nucleus.
    * **Confocal-type** modalities weight a plane at ``dz`` by
      ``1 / (1 + (2 dz / FWHM_z)^2)``: the pinhole's rejection of a uniform plane,
      with ``FWHM_z`` the optical-section thickness for a 1 AU pinhole.

Scalar, paraxial, aberration-free, as in NoiseWidget. All lengths are microns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Refractive indices at ~550 nm. Immersion oil is the ISO 8036 value.
IMMERSION_MEDIA: dict[str, float] = {
    "air": 1.000,
    "water": 1.333,
    "silicone oil": 1.406,
    "glycerol": 1.474,
    "oil": 1.518,
}

# The OTF of a circular pupil, scaled so the PSF has a given FWHM: f_c * FWHM.
CUTOFF_TIMES_FWHM = 2.0 * 0.5145


@dataclass(frozen=True)
class Objective:
    """A generic objective class, not a vendor product."""

    label: str
    magnification: float
    na: float
    medium: str


OBJECTIVES: list[Objective] = [
    Objective("20x / 0.75 air", 20, 0.75, "air"),
    Objective("40x / 0.95 air", 40, 0.95, "air"),
    Objective("40x / 1.30 oil", 40, 1.30, "oil"),
    Objective("60x / 1.20 water", 60, 1.20, "water"),
    Objective("60x / 1.40 oil", 60, 1.40, "oil"),
    Objective("100x / 1.45 oil", 100, 1.45, "oil"),
]


@dataclass(frozen=True)
class Modality:
    """A kind of microscope, by what it does to lateral and axial resolution."""

    name: str
    lateral_factor: float  # FWHM relative to the widefield 0.514 lambda / NA
    sectioning: bool  # rejects out-of-focus light (pinhole) rather than keeping it
    fixed_fwhm: bool  # the lateral FWHM is a lever of its own (STED)
    note: str


MODALITIES: dict[str, Modality] = {
    m.name: m
    for m in (
        Modality(
            "widefield", 1.0, False, False,
            "every plane contributes: out-of-focus light is blurred, never removed",
        ),
        Modality(
            "confocal (1 AU)", 0.90, True, False,
            "a 1 AU pinhole: modest lateral gain, but out-of-focus planes are rejected",
        ),
        Modality(
            "pixel reassignment (Airyscan-type)", 1 / 1.7, True, False,
            "a detector array behind the pinhole: ~1.7x finer laterally, confocal sectioning",
        ),
        Modality(
            "STED", 1.0, True, True,
            "a depletion doughnut narrows the spot to the FWHM you set; confocal sectioning",
        ),
    )
}


def check_na(na: float, n: float) -> None:
    if na <= 0:
        raise ValueError("NA must be positive")
    if na > n:
        raise ValueError(f"NA {na:.2f} exceeds the immersion index {n:.3f}: NA = n sin(theta) <= n")


def widefield_fwhm_um(wavelength_um: float, na: float) -> float:
    """FWHM of the Airy disc, ``0.514 lambda / NA``."""
    return 0.5145 * wavelength_um / na


def rayleigh_um(wavelength_um: float, na: float) -> float:
    """Rayleigh two-point resolution, ``0.61 lambda / NA``."""
    return 0.61 * wavelength_um / na


def lateral_fwhm_um(modality: str, wavelength_um: float, na: float, sted_fwhm_um: float = 0.05) -> float:
    m = MODALITIES[modality]
    if m.fixed_fwhm:
        # STED's spot is set by the depletion power, not the NA -- but it is never
        # wider than the confocal spot it is carved from.
        return min(sted_fwhm_um, widefield_fwhm_um(wavelength_um, na) * 0.9)
    return m.lateral_factor * widefield_fwhm_um(wavelength_um, na)


def cone_tan(na: float, n: float) -> float:
    """``tan(theta)`` of the objective's marginal ray, ``sin theta = NA / n``."""
    check_na(na, n)
    s = min(na / n, 0.999)
    return s / math.sqrt(1.0 - s * s)


def axial_section_um(wavelength_um: float, na: float, n: float, pinhole_au: float = 1.0) -> float:
    """Optical-section FWHM of a pinholed microscope.

    ``sqrt((0.88 lambda / (n - sqrt(n^2 - NA^2)))^2 + (sqrt(2) n PH / NA)^2)``, with
    the pinhole PH back-projected to the sample (1 AU = 1.22 lambda / NA).
    Cole, Jinadasa & Brown, Nat. Protoc. 6, 1929 (2011), after Wilson.
    """
    check_na(na, n)
    diffraction = 0.88 * wavelength_um / (n - math.sqrt(max(n * n - na * na, 0.0)))
    ph = pinhole_au * 1.22 * wavelength_um / na
    geometric = math.sqrt(2.0) * n * ph / na
    return math.hypot(diffraction, geometric)


def depth_of_field_um(wavelength_um: float, na: float, n: float) -> float:
    """Widefield depth of field, ``lambda n / NA^2`` -- but widefield never rejects the rest."""
    return wavelength_um * n / na**2


def nyquist_pixel_um(fwhm_um: float) -> float:
    """Largest pixel that samples the cutoff ``f_c = 1.029 / FWHM``: ``1 / (2 f_c)``."""
    return fwhm_um / (2.0 * CUTOFF_TIMES_FWHM)


def plane_weight(dz_um: np.ndarray | float, sectioning: bool, section_um: float):
    """Fraction of a plane's light at defocus ``dz`` that reaches the image."""
    dz = np.asarray(dz_um, dtype=float)
    if not sectioning:
        return np.ones_like(dz)
    return 1.0 / (1.0 + (2.0 * dz / section_um) ** 2)


# --- transfer functions on an rfft2 grid ------------------------------------------


@dataclass(frozen=True)
class FourierGrid:
    """Frequencies of an ``n x n`` real image at spacing ``dx``, in rfft2 layout."""

    n: int
    dx_um: float

    @property
    def fx(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n, d=self.dx_um)[None, :]

    @property
    def fy(self) -> np.ndarray:
        return np.fft.fftfreq(self.n, d=self.dx_um)[:, None]

    def radius(self) -> np.ndarray:
        return np.hypot(self.fx, self.fy)


def airy_otf(f: np.ndarray, fwhm_um: float) -> np.ndarray:
    """Incoherent circular-pupil OTF whose PSF has the given FWHM."""
    s = np.clip(f * fwhm_um / CUTOFF_TIMES_FWHM, 0.0, 1.0)
    return (2.0 / math.pi) * (np.arccos(s) - s * np.sqrt(1.0 - s * s))


def defocus_otf(f2: np.ndarray, dz_um: float, tan_theta: float) -> np.ndarray:
    """Extra blur of a plane ``dz`` out of focus: Gaussian of ``sigma = |dz| tan(theta) / 2``."""
    sigma = abs(dz_um) * tan_theta / 2.0
    if sigma == 0:
        return np.ones_like(f2)
    return np.exp(-2.0 * math.pi**2 * sigma**2 * f2)


def shift_phase(grid: FourierGrid, dx_um: float, dy_um: float) -> np.ndarray:
    """Multiplying by this translates the image by (dx, dy)."""
    return np.exp(-2j * math.pi * (grid.fx * dx_um + grid.fy * dy_um))


def psf(n: int, dx_um: float, fwhm_um: float) -> np.ndarray:
    """In-focus intensity PSF, centred, unit sum. Used by the tests and the report."""
    grid = FourierGrid(n, dx_um)
    h = np.fft.irfft2(airy_otf(grid.radius(), fwhm_um), s=(n, n))
    h = np.fft.fftshift(np.clip(h, 0.0, None))
    return h / h.sum()


# --- pixels ------------------------------------------------------------------------


def box_integrate(image: np.ndarray, dx_um: float, pixel_um: float) -> np.ndarray:
    """Integrate a finely sampled image over square pixels of any size.

    A pixel counts every photon that lands on it: an area integral, not a resample.
    The cumulative sum of a piecewise-constant image is piecewise linear, so linear
    interpolation of it at the pixel edges is exact. Pixels that would overhang the
    field are dropped. Vectorised: the edges are the same for every row.
    """
    if pixel_um <= 0:
        raise ValueError("pixel size must be positive")
    n = image.shape[0]
    count = int(math.floor(n * dx_um / pixel_um + 1e-9))
    if count < 1:
        raise ValueError(f"a {pixel_um:.3g} um pixel is larger than the {n * dx_um:.3g} um field")
    edges = np.arange(count + 1) * (pixel_um / dx_um)
    i0 = np.minimum(np.floor(edges).astype(int), n - 1)
    t = edges - i0

    def along_last(a: np.ndarray) -> np.ndarray:
        c = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, axis=-1)], axis=-1)
        at = c[..., i0] * (1.0 - t) + c[..., i0 + 1] * t
        return np.diff(at, axis=-1)

    return along_last(along_last(image).T).T
