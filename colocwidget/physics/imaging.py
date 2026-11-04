"""From labels to a two-colour image to colocalization numbers, at three stages.

The same coefficients are computed three times, so each loss is visible on its own:

1. **Truth**: from the labels themselves, at a stated molecular distance.
2. **Noise-free image**: after the optics (resolution, sectioning, registration)
   and the pixels, but before any photon is counted. The difference from the truth
   is what *resolution* does.
3. **Recorded image**: with shot noise, background and read noise, measured over
   repeat frames. The difference from stage 2 is what *noise* does.

The imaging is linear, so each channel's image is a sum over focal planes,

    I(f) = OTF(f; FWHM) * sum_k  D_k(f) G(f; z_k - z_focus) w(z_k - z_focus)

with ``D_k`` the label density in plane k, G the defocus blur and w the plane's
weight (1 in widefield, a Lorentzian for a pinhole). Only the sum over planes
depends on the focus and the sectioning, and only the OTF on the lateral
resolution -- so the planes' FFTs are cached on the scene, the sum on the axial
optics, and a resolution sweep costs one inverse FFT per point.

Two clocks, as in NoiseWidget: the scene and its FFTs (~0.1-0.5 s, cached), and
everything after (milliseconds to tens of milliseconds), so the sliders are live.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from . import optics
from .scenes import COMPARTMENTS, DEFAULT_SCENE, MEMBRANE, NUCLEUS, CYTOSOL, SCENES, ScenePoints, build_scene

GRID_N = 512
ROIS = ("whole field", "cell", "nucleus", "cytosol", "plasma membrane")


@dataclass(frozen=True)
class Levers:
    """Everything a user can turn. Defaults: interleaved puncta under a widefield microscope."""

    # specimen
    scene: str = DEFAULT_SCENE
    scene_params: tuple[tuple[str, float], ...] = ()  # overrides of the scene's defaults
    scene_seed: int = 1
    # optics
    modality: str = "widefield"
    na: float = 1.40
    medium: str = "oil"
    emission_nm_1: float = 520.0
    emission_nm_2: float = 600.0
    sted_fwhm_nm: float = 50.0
    focus_um: float | None = None  # None: the scene's default focal plane
    # registration of channel 2 against channel 1 (chromatic aberration, misalignment)
    shift_x_nm: float = 0.0
    shift_y_nm: float = 0.0
    shift_z_nm: float = 0.0
    # detection
    pixel_nm: float = 65.0
    peak_1: float = 200.0  # photons in the brightest pixels of the noise-free image
    peak_2: float = 200.0
    background: float = 10.0  # photons per pixel, each channel
    read_noise_e: float = 1.5
    bleed_through: float = 0.0  # fraction of channel 1's photons detected in channel 2
    noise: bool = True
    # analysis
    roi: str = "whole field"
    threshold: str = "Costes"
    threshold_fraction: float = 0.2
    manders_form: str = M.MANDERS_FORMS[0]  # denominator: signal above its own threshold, or all signal
    subtract_background: bool = True
    truth_nm: float = 25.0  # molecular distance that counts as "together"
    realizations: int = 10
    noise_seed: int = 0

    def with_(self, **changes) -> "Levers":
        params = changes.pop("params", None)
        out = dataclasses.replace(self, **changes)
        if params:
            merged = dict(out.scene_params)
            merged.update(params)
            out = dataclasses.replace(out, scene_params=tuple(sorted(merged.items())))
        return out

    @property
    def scene_model(self):
        return SCENES[self.scene]

    @property
    def params(self) -> dict[str, float]:
        return self.scene_model.resolve(dict(self.scene_params))

    @property
    def focus(self) -> float:
        return self.scene_model.default_focus(self.params) if self.focus_um is None else self.focus_um

    @property
    def n_medium(self) -> float:
        return optics.IMMERSION_MEDIA[self.medium]

    def wavelength_um(self, channel: int) -> float:
        return (self.emission_nm_1 if channel == 1 else self.emission_nm_2) / 1000.0

    def fwhm_um(self, channel: int) -> float:
        return optics.lateral_fwhm_um(self.modality, self.wavelength_um(channel), self.na, self.sted_fwhm_nm / 1000)

    def section_um(self, channel: int) -> float:
        return optics.axial_section_um(self.wavelength_um(channel), self.na, self.n_medium)


# --- result -------------------------------------------------------------------------


@dataclass
class Stage:
    """One stage's coefficients in the region of interest."""

    name: str
    coeffs: M.Coefficients


@dataclass
class Repeats:
    """The recorded coefficients over repeat frames of the same scene."""

    values: dict[str, np.ndarray]

    def mean(self, key: str) -> float:
        v = self.values[key]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    def sd(self, key: str) -> float:
        v = self.values[key]
        v = v[np.isfinite(v)]
        return float(v.std(ddof=1)) if v.size > 1 else float("nan")


@dataclass
class CompartmentRow:
    name: str
    true_1: float  # fraction of channel 1's labels in the compartment
    true_2: float
    seen_1: float  # fraction of channel 1's recorded signal in its ROI (focal plane)
    seen_2: float
    coeffs: M.Coefficients | None


@dataclass
class Simulation:
    levers: Levers
    scene: ScenePoints
    fwhm_um: tuple[float, float]
    pixel_um: float
    object_slice: tuple[np.ndarray, np.ndarray]  # labels in the focal plane, fine grid
    optical: tuple[np.ndarray, np.ndarray]  # noise-free photons per pixel (after bleed-through)
    expected: tuple[np.ndarray, np.ndarray]  # optical + background
    recorded: tuple[np.ndarray, np.ndarray]  # photons, with noise
    analysed: tuple[np.ndarray, np.ndarray]  # what the coefficients see (background-subtracted, clipped)
    roi_mask: np.ndarray
    roi_name: str
    masks: dict[str, np.ndarray]
    truth: M.Truth
    noise_free: M.Coefficients
    measured: M.Coefficients
    repeats: Repeats
    costes_p: tuple[float, float, float]
    ccf: tuple[np.ndarray, np.ndarray]
    spots: M.SpotColoc
    compartments: list[CompartmentRow]
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.recorded[0].shape

    @property
    def extent_um(self) -> float:
        return self.shape[1] * self.pixel_um


# --- the model ----------------------------------------------------------------------


class Simulator:
    """Holds the caches. ``run(levers)`` is pure in its result."""

    def __init__(self):
        self._scene_key = None
        self._scene: ScenePoints | None = None
        self._planes: list[np.ndarray] = []  # per channel: (nz, n, n//2+1) complex
        self._density: list[np.ndarray] = []  # per channel: (nz, n, n) label counts
        self._grid: optics.FourierGrid | None = None
        self._f = None
        self._f2 = None
        self._axial_key = None
        self._axial: list[np.ndarray] = []
        self._truth_key = None
        self._truth: M.Truth | None = None
        self._sweep_key = None
        self._sweep = None

    # --- caches ---------------------------------------------------------------

    def scene(self, lv: Levers) -> ScenePoints:
        key = (lv.scene, lv.scene_params, lv.scene_seed)
        if key != self._scene_key:
            sp, _ = build_scene(lv.scene, dict(lv.scene_params), lv.scene_seed)
            n, dx = GRID_N, sp.dx_um
            self._grid = optics.FourierGrid(n, dx)
            self._f = self._grid.radius()
            self._f2 = self._f**2
            self._density, self._planes = [], []
            for labels in (sp.ch1, sp.ch2):
                vol = _deposit(labels.xyz, n, dx, sp.z_slices)
                self._density.append(vol)
                self._planes.append(np.fft.rfft2(vol))
            self._scene, self._scene_key = sp, key
            self._axial_key = self._truth_key = self._sweep_key = None
        return self._scene

    def axial_sums(self, lv: Levers) -> list[np.ndarray]:
        """Per channel: the planes, defocused and weighted for this focus, summed."""
        self.scene(lv)
        key = (self._scene_key, lv.modality, lv.na, lv.medium, lv.focus, lv.emission_nm_1, lv.emission_nm_2,
               lv.shift_x_nm, lv.shift_y_nm, lv.shift_z_nm)
        if key != self._axial_key:
            m = optics.MODALITIES[lv.modality]
            sections = [lv.section_um(c) if m.sectioning else None for c in (1, 2)]
            self._axial, self._axial_key = self._plane_sums(lv, sections), key
        return self._axial

    def _plane_sums(self, lv: Levers, sections: list[float | None]) -> list[np.ndarray]:
        """Sum the planes for each channel; a section of None keeps every plane (widefield)."""
        sp = self._scene
        tan = optics.cone_tan(lv.na, lv.n_medium)
        out = []
        for c, planes in enumerate(self._planes, start=1):
            focus = lv.focus + (lv.shift_z_nm / 1000 if c == 2 else 0.0)
            section = sections[c - 1]
            total = np.zeros_like(planes[0])
            for z, plane in zip(sp.z_slices, planes):
                dz = z - focus
                w = float(optics.plane_weight(dz, section is not None, section or 1.0))
                if w < 1e-4:
                    continue
                total += plane * (w * optics.defocus_otf(self._f2, dz, tan))
            if c == 2 and (lv.shift_x_nm or lv.shift_y_nm):
                total *= optics.shift_phase(self._grid, lv.shift_x_nm / 1000, lv.shift_y_nm / 1000)
            out.append(total)
        return out

    def truth(self, lv: Levers) -> M.Truth:
        sp = self.scene(lv)
        key = (self._scene_key, lv.truth_nm)
        if key != self._truth_key:
            d = lv.truth_nm / 1000
            p, q = sp.ch1.xyz, sp.ch2.xyz
            zs = np.concatenate([p[:, 2], q[:, 2]]) if len(p) + len(q) else np.zeros(1)
            layers = max(float(np.ptp(zs)) + d, d) / d
            volume = (sp.field_um / d) ** 2 * layers
            by_comp = {}
            for comp in COMPARTMENTS if sp.cell is not None else ():
                f1 = float((sp.ch1.compartment == comp).mean()) if len(sp.ch1) else 0.0
                f2 = float((sp.ch2.compartment == comp).mean()) if len(sp.ch2) else 0.0
                by_comp[comp] = (f1, f2)
            self._truth = M.Truth(
                d, M.fraction_within(p, q, d), M.fraction_within(q, p, d),
                M.voxel_pearson(p, q, d, volume), len(p), len(q), by_comp,
            )
            self._truth_key = key
        return self._truth

    # --- images ---------------------------------------------------------------

    def optical_pixels(self, lv: Levers, fwhm: tuple[float, float], pixel_um: float,
                       sums: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Noise-free photons per pixel, scaled to the peaks, with bleed-through."""
        sp = self.scene(lv)
        sums = self.axial_sums(lv) if sums is None else sums
        n = GRID_N
        imgs = []
        for c, total in enumerate(sums):
            fine = np.fft.irfft2(total * optics.airy_otf(self._f, fwhm[c]), s=(n, n))
            px = optics.box_integrate(np.clip(fine, 0.0, None), sp.dx_um, pixel_um)
            imgs.append(px)
        out = []
        for c, img in enumerate(imgs):
            ref = float(np.percentile(img, 99.9))
            peak = lv.peak_1 if c == 0 else lv.peak_2
            out.append(img * (peak / ref) if ref > 0 else img)
        if lv.bleed_through > 0:
            out[1] = out[1] + lv.bleed_through * out[0]
        return out[0], out[1]

    def masks(self, sp: ScenePoints, lv: Levers, shape: tuple[int, int], pixel_um: float, fwhm: float) -> dict[str, np.ndarray]:
        """Regions of interest on the pixel grid, drawn at the focal plane."""
        h, w = shape
        out = {"whole field": np.ones(shape, bool)}
        g = sp.cell
        if g is None:
            return out
        y = (np.arange(h)[:, None] + 0.5) * pixel_um
        x = (np.arange(w)[None, :] + 0.5) * pixel_um
        z = min(max(lv.focus, 0.0), g.h - 1e-3)
        a, b = g.cell_section(z)
        if a <= 0:
            return out
        rho = np.hypot((x - g.centre[0]) / a, (y - g.centre[1]) / b)
        cell = rho <= 1.0
        rim_um = max(0.4, 1.5 * fwhm)
        rim = cell & (rho > 1.0 - rim_um / min(a, b))
        na_, nb = g.nucleus_section(z)
        if na_ > 0:
            nuc = np.hypot((x - g.nucleus_centre[0]) / na_, (y - g.nucleus_centre[1]) / nb) <= 1.0
            nuc &= ~rim
        else:
            nuc = np.zeros(shape, bool)
        out["cell"] = cell
        out["nucleus"] = nuc
        out["cytosol"] = cell & ~rim & ~nuc
        out["plasma membrane"] = rim
        return out

    def record(self, lv: Levers, optical: tuple[np.ndarray, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([seed, 12345])
        out = []
        for img in optical:
            expected = img + lv.background
            if lv.noise:
                out.append(rng.poisson(expected).astype(float) + rng.normal(0, lv.read_noise_e, size=img.shape))
            else:
                out.append(expected.copy())
        return out[0], out[1]

    def analysed(self, lv: Levers, recorded: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if lv.subtract_background:
            return tuple(np.clip(r - lv.background, 0.0, None) for r in recorded)
        return recorded

    # --- the run --------------------------------------------------------------

    def run(self, lv: Levers) -> Simulation:
        optics.check_na(lv.na, lv.n_medium)
        if lv.scene not in SCENES:
            raise ValueError(f"no scene {lv.scene!r}")
        sp = self.scene(lv)
        notes: list[str] = []
        fwhm = (lv.fwhm_um(1), lv.fwhm_um(2))
        pixel = max(lv.pixel_nm / 1000, sp.dx_um)
        if pixel > lv.pixel_nm / 1000 + 1e-9:
            notes.append(f"pixel raised to the scene's {sp.dx_um * 1000:.0f} nm grid")

        optical = self.optical_pixels(lv, fwhm, pixel)
        expected = tuple(o + lv.background for o in optical)
        recorded = self.record(lv, optical, lv.noise_seed)
        analysed = self.analysed(lv, recorded)
        clean = self.analysed(lv, expected)

        masks = self.masks(sp, lv, optical[0].shape, pixel, fwhm[0])
        roi = lv.roi if lv.roi in masks else "whole field"
        if roi != lv.roi:
            notes.append(f"no {lv.roi!r} region in this scene: measuring the whole field")
        mask = masks[roi]
        if mask.sum() < 16:
            notes.append(f"the {roi} region is empty at this focus: measuring the whole field")
            roi, mask = "whole field", masks["whole field"]

        measured = M.coefficients(analysed[0][mask], analysed[1][mask], lv.threshold, lv.threshold_fraction,
                                  form=lv.manders_form)
        # The noise-free image is thresholded where the recorded one was. A threshold
        # chosen on a noise-free image falls into the PSF's faint tails, which a real
        # detector never resolves from zero, and counts them all as "present".
        noise_free = M.coefficients(clean[0][mask], clean[1][mask], lv.threshold, lv.threshold_fraction,
                                    fixed=measured.thresholds, form=lv.manders_form)

        reps: dict[str, list[float]] = {k: [] for k in M.Coefficients.KEYS}
        for r in range(max(lv.realizations, 0)):
            rec = self.analysed(lv, self.record(lv, optical, 1_000_003 + 7919 * lv.noise_seed + r))
            c = M.coefficients(rec[0][mask], rec[1][mask], lv.threshold, lv.threshold_fraction, rank=False,
                               form=lv.manders_form)
            for k in reps:
                reps[k].append(c.get(k))
        repeats = Repeats({k: np.array(v, float) for k, v in reps.items()})

        block = max(int(round(fwhm[0] / pixel)), 1)
        costes_p = M.costes_significance(analysed[0], analysed[1], mask, block, trials=40, seed=lv.noise_seed)
        max_shift = max(int(round(1.0 / pixel)), 8)
        ccf = M.cross_correlation(analysed[0], analysed[1], mask, min(max_shift, analysed[0].shape[1] // 3))
        ccf = (ccf[0] * pixel, ccf[1])
        spots = M.spot_colocalization(
            analysed[0], analysed[1], mask, max(measured.thresholds.ta, 0.0), max(measured.thresholds.tb, 0.0),
            pixel, max(fwhm),
        )

        truth = self.truth(lv)
        compartments = self._compartments(sp, lv, masks, analysed, truth)

        # The ground truth in the focal plane, for the object pane.
        iz = int(np.argmin(np.abs(sp.z_slices - lv.focus))) if not sp.flat else 0
        obj = (self._density[0][iz], self._density[1][iz])

        return Simulation(
            lv, sp, fwhm, pixel, obj, optical, expected, recorded, analysed, mask, roi, masks,
            truth, noise_free, measured, repeats, costes_p, ccf, spots, compartments, notes + sp.notes,
        )

    def _compartments(self, sp, lv, masks, analysed, truth) -> list[CompartmentRow]:
        if sp.cell is None or "cell" not in masks:
            return []
        cell = masks["cell"]
        tot1 = float(analysed[0][cell].sum()) or 1.0
        tot2 = float(analysed[1][cell].sum()) or 1.0
        rows = []
        for comp, roi in ((NUCLEUS, "nucleus"), (CYTOSOL, "cytosol"), (MEMBRANE, "plasma membrane")):
            m = masks[roi]
            t1, t2 = truth.by_compartment.get(comp, (0.0, 0.0))
            coeffs = (M.coefficients(analysed[0][m], analysed[1][m], lv.threshold, lv.threshold_fraction, rank=False,
                                     form=lv.manders_form)
                      if m.sum() >= 16 else None)
            rows.append(CompartmentRow(COMPARTMENTS[comp], t1, t2, float(analysed[0][m].sum()) / tot1,
                                       float(analysed[1][m].sum()) / tot2, coeffs))
        return rows

    # --- the resolution sweep -------------------------------------------------

    def sweep(self, lv: Levers, axis: str = "auto", points: int = 11) -> "Sweep":
        """The coefficients as one kind of resolution changes and nothing else does.

        ``lateral``: the PSF's FWHM, from super-resolution to a low-NA objective,
        with channel 2 keeping its ratio to channel 1. ``axial``: the optical-section
        thickness, from a thin confocal section to widefield (no section). ``auto``
        picks axial for a cell -- where compartments are told apart in z -- and
        lateral otherwise.

        Each point is recorded with the current photons and a fixed noise
        realization, so the curve is what you would measure; thresholds are chosen
        per point as the analysis lever says. Cached on everything that changes it.
        """
        sp = self.scene(lv)
        if axis == "auto":
            axis = "lateral" if sp.flat else "axial"
        pixel = max(lv.pixel_nm / 1000, sp.dx_um)
        key = (self._scene_key, axis, lv.modality, lv.na, lv.medium, lv.focus, lv.emission_nm_1,
               lv.emission_nm_2, lv.sted_fwhm_nm, lv.shift_x_nm, lv.shift_y_nm, lv.shift_z_nm, pixel,
               lv.bleed_through, lv.roi, lv.threshold, lv.threshold_fraction, lv.subtract_background, lv.peak_1,
               lv.peak_2, lv.background, lv.read_noise_e, lv.noise, lv.manders_form, points)
        if key == self._sweep_key:
            return self._sweep
        values = {k: np.full(points, np.nan) for k in ("pcc", "m1", "m2", "moc")}
        fwhm = (lv.fwhm_um(1), lv.fwhm_um(2))
        m = optics.MODALITIES[lv.modality]
        if axis == "lateral":
            xs = np.geomspace(max(30.0, 2.5 * sp.dx_um * 1000) / 1000, 1.2, points)
            here = fwhm[0]
            ratio = fwhm[1] / fwhm[0]
            settings = [((x, x * ratio), None) for x in xs]
        else:
            # The last point stands for widefield: no section at all.
            xs = np.geomspace(0.3, 30.0, points)
            here = lv.section_um(1) if m.sectioning else xs[-1]
            ratio = lv.section_um(2) / lv.section_um(1)
            settings = [(fwhm, [x, x * ratio] if i < points - 1 else [None, None]) for i, x in enumerate(xs)]
        for i, (f, sections) in enumerate(settings):
            sums = None if sections is None else self._plane_sums(lv, sections)
            opt = self.optical_pixels(lv, f, pixel, sums)
            masks = self.masks(sp, lv, opt[0].shape, pixel, f[0])
            mask = masks.get(lv.roi, masks["whole field"])
            if mask.sum() < 16:
                mask = masks["whole field"]
            rec = self.analysed(lv, self.record(lv, opt, 424242))
            c = M.coefficients(rec[0][mask], rec[1][mask], lv.threshold, lv.threshold_fraction, rank=False,
                               form=lv.manders_form)
            for k in values:
                values[k][i] = c.get(k)
        self._sweep = Sweep(axis, xs, values, here)
        self._sweep_key = key
        return self._sweep


@dataclass
class Sweep:
    axis: str  # lateral (x = FWHM, um) or axial (x = optical section, um; the last point is widefield)
    x_um: np.ndarray
    values: dict[str, np.ndarray]
    here_um: float


def _deposit(xyz: np.ndarray, n: int, dx: float, z_slices: np.ndarray) -> np.ndarray:
    """Count labels into (plane, row, column) bins: the object, sampled at dx."""
    vol = np.zeros((z_slices.size, n, n))
    if len(xyz) == 0:
        return vol
    j = np.floor(xyz[:, 0] / dx).astype(int)
    i = np.floor(xyz[:, 1] / dx).astype(int)
    if z_slices.size == 1:
        k = np.zeros(len(xyz), int)
    else:
        dz = z_slices[1] - z_slices[0]
        k = np.clip(np.round((xyz[:, 2] - z_slices[0]) / dz).astype(int), 0, z_slices.size - 1)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    np.add.at(vol, (k[ok], i[ok], j[ok]), 1.0)
    return vol
