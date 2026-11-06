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

"""Specimens with a known answer: where every label of both colours really is.

A colocalization measurement is only interesting next to the truth it is meant to
report. Each scene places individual labels (points in 3D, in microns) for two
channels, so the widget can compare what the image says with what the molecules
do. They are built to separate two questions that a colocalization number mixes:

* **Are the two labels on the same thing?** (molecular proximity, the biology)
* **Are they in the same resolution element?** (what the image can know)

The scenes, by the case each serves:

1. *Unresolved collections.* **Puncta clusters**: many small puncta packed into
   sub-resolution clusters. With nothing co-labelled, a widefield image of an
   interleaved cluster is indistinguishable from one body carrying both labels.
2. *Distribution and morphology.* **Adjacent filaments**, **vesicles: membrane and
   lumen**, **dense independent signals**, **co-labelled puncta with variable
   stoichiometry**, **exclusive domains**: each isolates one way that the shape or
   density of a distribution moves Pearson's and Manders' coefficients without
   changing the molecular truth, or the other way round.
3. *Cells and compartments.* **Model cell**: a 3D adherent cell with a nucleus,
   cytosol and plasma membrane, and each channel's labels split among them.

Every scene declares its own parameters, so the lever panel and the command line
discover them rather than knowing them.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Compartment tags carried by every label (cells use all of them; flat scenes use 0).
NONE, NUCLEUS, CYTOSOL, MEMBRANE = 0, 1, 2, 3
COMPARTMENTS = {NUCLEUS: "nucleus", CYTOSOL: "cytosol", MEMBRANE: "plasma membrane"}


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    lo: float
    hi: float
    default: float
    unit: str = ""
    log: bool = False
    decimals: int = 0
    help: str = ""


@dataclass
class Labels:
    """One channel's labels: positions (N x 3, um) and the compartment of each."""

    xyz: np.ndarray
    compartment: np.ndarray

    @classmethod
    def empty(cls) -> "Labels":
        return cls(np.zeros((0, 3)), np.zeros(0, dtype=np.int8))

    @classmethod
    def join(cls, parts: list["Labels"]) -> "Labels":
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(np.concatenate([p.xyz for p in parts]), np.concatenate([p.compartment for p in parts]))

    def __len__(self) -> int:
        return self.xyz.shape[0]


@dataclass(frozen=True)
class CellGeometry:
    """A dome-shaped adherent cell on the coverslip (z = 0), and its nucleus."""

    centre: tuple[float, float]
    a: float  # cell semi-axes, um (x, y) and height
    b: float
    h: float
    nucleus_centre: tuple[float, float, float]
    na_: float  # nucleus semi-axes
    nb: float
    nc: float

    def cell_section(self, z: float) -> tuple[float, float]:
        """Semi-axes of the cell's cross-section at height z (0 above the cell)."""
        if z < 0 or z >= self.h:
            return 0.0, 0.0
        s = math.sqrt(max(1.0 - (z / self.h) ** 2, 0.0))
        return self.a * s, self.b * s

    def nucleus_section(self, z: float) -> tuple[float, float]:
        dz = (z - self.nucleus_centre[2]) / self.nc
        if abs(dz) >= 1:
            return 0.0, 0.0
        s = math.sqrt(1.0 - dz * dz)
        return self.na_ * s, self.nb * s

    def inside_cell(self, p: np.ndarray) -> np.ndarray:
        x, y, z = (p[:, 0] - self.centre[0]) / self.a, (p[:, 1] - self.centre[1]) / self.b, p[:, 2] / self.h
        return (x * x + y * y + z * z <= 1.0) & (p[:, 2] >= 0)

    def inside_nucleus(self, p: np.ndarray) -> np.ndarray:
        cx, cy, cz = self.nucleus_centre
        x, y, z = (p[:, 0] - cx) / self.na_, (p[:, 1] - cy) / self.nb, (p[:, 2] - cz) / self.nc
        return x * x + y * y + z * z <= 1.0


@dataclass
class ScenePoints:
    """What a scene generates: both channels' labels and the field they sit in."""

    ch1: Labels
    ch2: Labels
    field_um: float
    dx_um: float
    z_slices: np.ndarray  # plane heights the imaging grid uses
    cell: CellGeometry | None = None
    feature_nm: float = 0.0  # the length scale the scene's lesson turns on
    feature_name: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def flat(self) -> bool:
        return self.z_slices.size == 1


@dataclass(frozen=True)
class Scene:
    name: str
    description: str
    params: tuple[Param, ...]
    build: Callable[[dict, np.random.Generator], ScenePoints]
    kind: str = "flat"  # flat or cell
    focus_range: tuple[float, float] = (-1.5, 1.5)

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}

    def resolve(self, overrides: dict[str, float] | None) -> dict[str, float]:
        values = self.defaults()
        for k, v in (overrides or {}).items():
            if k not in values:
                raise ValueError(f"scene {self.name!r} has no parameter {k!r}; it has {', '.join(values)}")
            p = next(q for q in self.params if q.name == k)
            values[k] = float(min(max(v, p.lo), p.hi))
        return values

    def default_focus(self, values: dict[str, float]) -> float:
        if self.kind == "cell":
            return round(values["nucleus_height"] / 2 + 0.5, 2)
        return 0.0


# --- shared geometry --------------------------------------------------------------

FLAT_FIELD_UM = 10.24
FLAT_DX_UM = 0.02
CELL_FIELD_UM = 25.6
CELL_DX_UM = 0.05
MARGIN_UM = 0.6


def _flat(points: np.ndarray) -> Labels:
    return Labels(points, np.zeros(points.shape[0], dtype=np.int8))


def _in_disc(rng, n: int, radius: float) -> np.ndarray:
    r = radius * np.sqrt(rng.random(n))
    t = rng.random(n) * 2 * math.pi
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _labels_in_spots(rng, centres: np.ndarray, per_spot: np.ndarray | int, diameter_um: float) -> np.ndarray:
    """Poisson-many labels uniformly in a disc around each centre (2D, z = 0)."""
    counts = rng.poisson(per_spot, size=centres.shape[0]) if np.ndim(per_spot) == 0 else rng.poisson(per_spot)
    idx = np.repeat(np.arange(centres.shape[0]), counts)
    xy = centres[idx] + _in_disc(rng, idx.size, diameter_um / 2)
    return np.column_stack([xy, np.zeros(idx.size)])


def _scatter_centres(rng, n: int, field_um: float, keep_apart_um: float = 0.0) -> np.ndarray:
    """Uniform centres away from the edges, optionally kept a minimum distance apart."""
    lo, hi = MARGIN_UM + keep_apart_um / 2, field_um - MARGIN_UM - keep_apart_um / 2
    out: list[np.ndarray] = []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        c = lo + rng.random(2) * (hi - lo)
        if keep_apart_um <= 0 or all(np.hypot(*(c - o)) >= keep_apart_um for o in out):
            out.append(c)
    return np.array(out).reshape(-1, 2)


def _flat_scene(ch1: np.ndarray, ch2: np.ndarray, feature_nm: float, feature_name: str, notes=None) -> ScenePoints:
    return ScenePoints(_flat(ch1), _flat(ch2), FLAT_FIELD_UM, FLAT_DX_UM, np.array([0.0]),
                       None, feature_nm, feature_name, notes or [])


# --- 1. puncta clusters -----------------------------------------------------------


def _puncta_clusters(v: dict, rng) -> ScenePoints:
    n_clusters = int(v["clusters"])
    radius = v["cluster_radius_nm"] / 1000
    per = int(v["puncta_per_cluster"])
    size = v["punctum_nm"] / 1000
    co = v["colabelled"]
    seg = v["segregation"]
    labels = 12
    centres = _scatter_centres(rng, n_clusters, FLAT_FIELD_UM, keep_apart_um=2 * radius + 0.9)
    p1, p2 = [], []
    for c in centres:
        pos = c + _in_disc(rng, per, radius)
        both = rng.random(per) < co
        # Single-labelled puncta pick a colour; segregation biases each colour to one
        # side of a random axis through the cluster, as in adjacent sub-domains.
        axis = rng.normal(size=2)
        axis /= np.linalg.norm(axis) or 1.0
        side = np.sign((pos - c) @ axis)
        p_one = 0.5 + 0.5 * seg * side
        one = rng.random(per) < p_one
        p1.append(pos[both | (~both & one)])
        p2.append(pos[both | (~both & ~one)])
    c1 = np.concatenate(p1) if p1 else np.zeros((0, 2))
    c2 = np.concatenate(p2) if p2 else np.zeros((0, 2))
    return _flat_scene(
        _labels_in_spots(rng, c1, labels, size), _labels_in_spots(rng, c2, labels, size),
        max(v["cluster_radius_nm"] * 2, v["punctum_nm"]), "cluster diameter",
        [f"{per * len(centres)} puncta in {len(centres)} clusters"],
    )


PUNCTA = Scene(
    "Puncta clusters",
    "Small puncta packed into clusters. Each punctum carries one label or both. "
    "At co-labelled fraction 0 no punctum carries both, so nothing is molecularly "
    "colocalized; below the resolution limit the clusters still look co-labelled.",
    (
        Param("clusters", "clusters", 1, 40, 14, help="how many clusters in the field"),
        Param("cluster_radius_nm", "cluster radius", 0, 800, 250, " nm",
              help="puncta are scattered uniformly within this radius"),
        Param("puncta_per_cluster", "puncta / cluster", 1, 80, 12),
        Param("punctum_nm", "punctum diameter", 5, 200, 30, " nm", help="e.g. a 30 nm vesicle or a protein complex"),
        Param("colabelled", "co-labelled fraction", 0, 1, 0.0, decimals=2,
              help="fraction of puncta that carry BOTH labels: the true molecular colocalization"),
        Param("segregation", "segregation", 0, 1, 0.0, decimals=2,
              help="0: the two colours interleave at random; 1: each colour holds one half of the cluster"),
    ),
    _puncta_clusters,
)


# --- 2a. adjacent filaments -------------------------------------------------------


def _curve(rng, length_um: float, start: np.ndarray, step_um: float = 0.01) -> np.ndarray:
    """A gently curving filament: a heading that wanders smoothly."""
    n = int(length_um / step_um)
    s = np.linspace(0, 1, n)
    heading = rng.random() * 2 * math.pi
    for k in (1, 2, 3):
        heading = heading + rng.normal(0, 0.5 / k) * np.sin(math.pi * k * s + rng.random() * 6)
    steps = np.stack([np.cos(heading), np.sin(heading)], axis=1) * step_um
    return start + np.cumsum(steps, axis=0)


def _labels_on_curve(rng, curve: np.ndarray, per_um: float) -> np.ndarray:
    seg = np.diff(curve, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    n = rng.poisson(per_um * length.sum())
    idx = rng.choice(len(seg), size=n, p=length / length.sum())
    t = rng.random(n)[:, None]
    xy = curve[idx] + seg[idx] * t
    return np.column_stack([xy, np.zeros(n)])


def _adjacent_filaments(v: dict, rng) -> ScenePoints:
    sep = v["separation_nm"] / 1000
    p1, p2 = [], []
    for _ in range(int(v["pairs"])):
        for _attempt in range(50):
            start = MARGIN_UM + 1.0 + rng.random(2) * (FLAT_FIELD_UM - 2 * MARGIN_UM - 2.0)
            c = _curve(rng, v["length_um"], start)
            if (c.min() > MARGIN_UM) and (c.max() < FLAT_FIELD_UM - MARGIN_UM):
                break
        d = np.gradient(c, axis=0)
        normal = np.stack([-d[:, 1], d[:, 0]], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        p1.append(_labels_on_curve(rng, c, v["labels_per_um"]))
        p2.append(_labels_on_curve(rng, c + normal * sep, v["labels_per_um"]))
    return _flat_scene(np.concatenate(p1), np.concatenate(p2), v["separation_nm"], "filament separation")


FILAMENTS = Scene(
    "Adjacent filaments",
    "Pairs of filaments running side by side, one in each colour, a fixed distance "
    "apart (a microtubule and the ER tubule along it, say). Physically separate at any "
    "separation above zero; colocalized in the image until the separation is resolved.",
    (
        Param("pairs", "filament pairs", 1, 20, 7),
        Param("separation_nm", "separation", 0, 1000, 120, " nm", help="centre-to-centre distance; 0 = same filament"),
        Param("length_um", "length", 1, 12, 6, " µm", decimals=1),
        Param("labels_per_um", "labels / µm", 5, 400, 80, log=True),
    ),
    _adjacent_filaments,
)


# --- 2b. vesicles: membrane and lumen ---------------------------------------------


def _vesicles(v: dict, rng) -> ScenePoints:
    d = v["diameter_nm"] / 1000
    centres = _scatter_centres(rng, int(v["vesicles"]), FLAT_FIELD_UM, keep_apart_um=d + 0.6)
    area = math.pi * d * d
    p1, p2 = [], []
    for c in centres:
        c3 = np.array([c[0], c[1], 0.0])
        n_shell = rng.poisson(max(area * 2500, 20))
        u = rng.normal(size=(n_shell, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        p1.append(u * d / 2 + c3)
        # The second label: in the lumen (filled) or also on the membrane.
        n2 = n_shell
        on_membrane = rng.random(n2) < v["ch2_on_membrane"]
        u2 = rng.normal(size=(n2, 3))
        u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
        r = np.where(on_membrane, d / 2, (d / 2 - 0.012) * np.cbrt(rng.random(n2)))
        p2.append(u2 * r[:, None] + c3)
    # Kept in 3D, so the truth measures real distances; the image treats every
    # vesicle as in focus (they are thinner than the depth of field).
    return ScenePoints(_flat(np.concatenate(p1)), _flat(np.concatenate(p2)), FLAT_FIELD_UM, FLAT_DX_UM,
                       np.array([0.0]), None, v["diameter_nm"], "vesicle diameter")


VESICLES = Scene(
    "Vesicles: membrane and lumen",
    "Spherical vesicles with one label on the membrane and the other filling the lumen "
    "(a membrane protein and its cargo). Never in the same place, always in the same "
    "vesicle: unresolved, the ring and its contents are one spot.",
    (
        Param("vesicles", "vesicles", 1, 60, 24),
        Param("diameter_nm", "diameter", 40, 1500, 400, " nm", log=True),
        Param("ch2_on_membrane", "ch2 on the membrane", 0, 1, 0.0, decimals=2,
              help="0: channel 2 fills the lumen; 1: both labels share the membrane"),
    ),
    _vesicles,
)


# --- 2c. dense independent signals ------------------------------------------------


def _dense(v: dict, rng) -> ScenePoints:
    # The whole field, edge to edge: an empty margin common to both channels would
    # itself be a "shared territory" and correlate them.
    lo, hi = 0.0, FLAT_FIELD_UM
    area = (hi - lo) ** 2
    n1 = rng.poisson(v["density_1"] * area)
    n2 = rng.poisson(v["density_2"] * area)
    p1 = lo + rng.random((n1, 2)) * (hi - lo)
    # A fraction of channel 2 genuinely binds channel 1 (within ~10 nm).
    n_bound = min(int(round(v["bound"] * n2)), n1)
    partners = p1[rng.choice(n1, size=n_bound, replace=False)] if n_bound else np.zeros((0, 2))
    bound = partners + rng.normal(0, 0.005, size=partners.shape)
    free = lo + rng.random((n2 - n_bound, 2)) * (hi - lo)
    p2 = np.concatenate([bound, free])
    # Patchy: both channels live in the same irregular region (a cell's footprint),
    # so the two overall densities co-vary even when the molecules do not.
    if v["patchy"] > 0:
        keep1 = _patch_mask(p1, rng_seed=7, scale_um=4.0) | (rng.random(len(p1)) > v["patchy"])
        keep2 = _patch_mask(p2, rng_seed=7, scale_um=4.0) | (rng.random(len(p2)) > v["patchy"])
        p1, p2 = p1[keep1], p2[keep2]
    zeros = np.zeros
    return _flat_scene(np.column_stack([p1, zeros(len(p1))]), np.column_stack([p2, zeros(len(p2))]),
                       1000 / math.sqrt(max(v["density_1"], 1)), "mean label spacing")


def _smooth_field(seed: int, scale_um: float, n: int = 512) -> np.ndarray:
    """A zero-mean smooth random field over the flat field, unit variance.

    Gaussian-filtered white noise. Its zero crossings -- the domain boundaries --
    are spaced about ``scale_um`` apart: the filter's sigma is ``scale / 4.4``,
    measured on this generator (mean domain width along a line).
    """
    g = np.random.default_rng(seed).normal(size=(n, n))
    f = np.fft.fftfreq(n, d=FLAT_FIELD_UM / n)
    f2 = f[:, None] ** 2 + f[None, :] ** 2
    sigma = scale_um / 4.4
    out = np.real(np.fft.ifft2(np.fft.fft2(g) * np.exp(-2 * math.pi**2 * sigma**2 * f2)))
    return (out - out.mean()) / (out.std() or 1.0)


def _sample_field(fieldmap: np.ndarray, p: np.ndarray) -> np.ndarray:
    n = fieldmap.shape[0]
    i = np.clip((p[:, 1] / FLAT_FIELD_UM * n).astype(int), 0, n - 1)
    j = np.clip((p[:, 0] / FLAT_FIELD_UM * n).astype(int), 0, n - 1)
    return fieldmap[i, j]


def _patch_mask(p: np.ndarray, rng_seed: int, scale_um: float) -> np.ndarray:
    return _sample_field(_smooth_field(rng_seed, scale_um), p) > 0


DENSE = Scene(
    "Dense independent signals",
    "Two channels scattered at random, independently. At high density every pixel "
    "holds some of each, so Manders' coefficients approach 1 with no relationship "
    "between the molecules. Pearson's stays near 0 -- unless both share a patchy "
    "territory, when it rises for a reason that is not binding.",
    (
        Param("density_1", "ch1 density", 1, 3000, 100, " /µm²", log=True),
        Param("density_2", "ch2 density", 1, 3000, 100, " /µm²", log=True),
        Param("bound", "ch2 bound to ch1", 0, 1, 0.0, decimals=2,
              help="fraction of channel-2 molecules sitting within ~10 nm of a channel-1 molecule"),
        Param("patchy", "shared territory", 0, 1, 0.0, decimals=2,
              help="both channels confined to the same irregular region: correlation without binding"),
    ),
    _dense,
)


# --- 2d. co-labelled puncta with variable stoichiometry ---------------------------


def _stoichiometry(v: dict, rng) -> ScenePoints:
    n = int(v["puncta"])
    centres = _scatter_centres(rng, n, FLAT_FIELD_UM, keep_apart_um=0.35)
    # Both channels on every punctum, so molecular colocalization is total. Channel 1's
    # copy number varies from punctum to punctum; channel 2's tracks it with a spread.
    base = 15.0 * np.exp(rng.normal(0, v["brightness_spread"], size=len(centres)))
    ratio = np.exp(rng.normal(0, v["ratio_spread"], size=len(centres)))
    size = v["punctum_nm"] / 1000
    return _flat_scene(
        _labels_in_spots(rng, centres, base, size),
        _labels_in_spots(rng, centres, base * ratio, size),
        v["punctum_nm"], "punctum diameter",
    )


STOICHIOMETRY = Scene(
    "Co-labelled puncta, variable stoichiometry",
    "Every punctum carries both labels, so co-occurrence is total. But the ratio of "
    "the two varies from punctum to punctum. Manders' coefficients (co-occurrence) "
    "stay near 1; Pearson's (proportionality) falls as the ratio scatters.",
    (
        Param("puncta", "puncta", 5, 300, 120),
        Param("punctum_nm", "punctum diameter", 5, 300, 40, " nm"),
        Param("brightness_spread", "brightness spread", 0, 1.5, 0.6, decimals=2,
              help="log-normal sigma of each punctum's copy number (both channels together)"),
        Param("ratio_spread", "ratio spread", 0, 2.0, 0.0, decimals=2,
              help="log-normal sigma of ch2/ch1 per punctum: 0 = fixed stoichiometry"),
    ),
    _stoichiometry,
)


# --- 2e. exclusive domains --------------------------------------------------------


def _domains(v: dict, rng) -> ScenePoints:
    lo, hi = 0.0, FLAT_FIELD_UM
    area = (hi - lo) ** 2
    fieldmap = _smooth_field(int(rng.integers(1 << 30)), v["domain_nm"] / 1000)
    n = rng.poisson(v["density"] * area * 2)
    p = lo + rng.random((n, 2)) * (hi - lo)
    inside = _sample_field(fieldmap, p) > 0
    p1 = p[inside]
    q = lo + rng.random((n, 2)) * (hi - lo)
    ignores = rng.random(n) < v["mixing"]
    p2 = q[(_sample_field(fieldmap, q) <= 0) | ignores]
    zeros = np.zeros
    return _flat_scene(np.column_stack([p1, zeros(len(p1))]), np.column_stack([p2, zeros(len(p2))]),
                       v["domain_nm"], "domain size")


DOMAINS = Scene(
    "Exclusive domains",
    "A membrane partitioned into domains: channel 1 occupies some, channel 2 the "
    "rest. Resolved, Pearson's is strongly negative. Blurred beyond the domain size, "
    "both look uniform and the anti-correlation vanishes into noise.",
    (
        Param("domain_nm", "domain size", 50, 3000, 400, " nm", log=True),
        Param("density", "label density", 20, 3000, 500, " /µm²", log=True),
        Param("mixing", "ch2 ignoring domains", 0, 1, 0.0, decimals=2),
    ),
    _domains,
)


# --- 3. model cell ----------------------------------------------------------------


def _cell_geometry(v: dict) -> CellGeometry:
    c = CELL_FIELD_UM / 2
    a, b, h = v["cell_length_um"] / 2, v["cell_width_um"] / 2, v["cell_height_um"]
    nh = min(v["nucleus_height"], h - 0.8)
    na_ = min(v["nucleus_diameter_um"] / 2, a - 1.5)
    nb = min(v["nucleus_diameter_um"] / 2 * 0.8, b - 1.5)
    return CellGeometry((c, c), a, b, h, (c + a * 0.15, c, nh / 2 + 0.5), na_, nb, nh / 2)


def _sample_membrane(rng, g: CellGeometry, n: int) -> np.ndarray:
    """Labels uniform per unit area over the apical dome and the basal disc."""
    # Area of the dome from its surface element on the unit hemisphere.
    probe = rng.normal(size=(4000, 3))
    probe[:, 2] = np.abs(probe[:, 2])
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)

    def element(u):
        return g.a * g.b * g.h * np.sqrt((u[:, 0] / g.a) ** 2 + (u[:, 1] / g.b) ** 2 + (u[:, 2] / g.h) ** 2)

    dome = 2 * math.pi * element(probe).mean()
    disc = math.pi * g.a * g.b
    n_dome = int(round(n * dome / (dome + disc)))
    out = []
    emax = element(probe).max() * 1.05
    while sum(len(o) for o in out) < n_dome:
        u = rng.normal(size=(n_dome * 2, 3))
        u[:, 2] = np.abs(u[:, 2])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        keep = rng.random(len(u)) < element(u) / emax
        out.append(u[keep])
    u = np.concatenate(out)[:n_dome]
    top = np.column_stack([g.centre[0] + g.a * u[:, 0], g.centre[1] + g.b * u[:, 1], g.h * u[:, 2]])
    base = _in_disc(rng, n - n_dome, 1.0)
    bottom = np.column_stack([g.centre[0] + g.a * base[:, 0], g.centre[1] + g.b * base[:, 1], np.full(len(base), 0.02)])
    return np.concatenate([top, bottom])


def _sample_volume(rng, g: CellGeometry, n: int, want_nucleus: bool) -> np.ndarray:
    """Uniform labels in the nucleus, or in the cytosol (the cell minus the nucleus)."""
    if n <= 0:
        return np.zeros((0, 3))
    if want_nucleus:
        lo = np.array([g.nucleus_centre[0] - g.na_, g.nucleus_centre[1] - g.nb, g.nucleus_centre[2] - g.nc])
        span = 2 * np.array([g.na_, g.nb, g.nc])
    else:
        lo = np.array([g.centre[0] - g.a, g.centre[1] - g.b, 0.0])
        span = np.array([2 * g.a, 2 * g.b, g.h])
    out, have = [], 0
    while have < n:
        p = lo + rng.random((max(n * 3, 1000), 3)) * span
        ok = g.inside_nucleus(p) & g.inside_cell(p) if want_nucleus else g.inside_cell(p) & ~g.inside_nucleus(p)
        out.append(p[ok])
        have += int(ok.sum())
    return np.concatenate(out)[:n]


# Puncta per compartment: a fixed number of bodies, so the morphology (how many,
# how big) stays put while the share of a channel in them is varied.
NUCLEAR_BODIES = 30
CYTOSOLIC_PUNCTA = 250


@dataclass
class _PunctaPools:
    """Body centres for one compartment: pool A is channel 1's, pool B channel 2's own."""

    a: np.ndarray
    b: np.ndarray
    sigma_um: float


def _body_centres(rng, g: CellGeometry, n: int, nucleus: bool, radius_um: float) -> np.ndarray:
    """Centres far enough inside their compartment that a body of this radius fits."""
    if nucleus:
        # Shrink the nucleus by the body radius, so bodies do not poke out of it.
        inner = dataclasses.replace(
            g, na_=max(g.na_ - radius_um, 0.3), nb=max(g.nb - radius_um, 0.3), nc=max(g.nc - radius_um, 0.3))
        return _sample_volume(rng, inner, n, True)
    return _sample_volume(rng, g, n, False)


def _place_in_puncta(rng, n: int, centres: np.ndarray, sigma_um: float) -> np.ndarray:
    idx = rng.integers(0, len(centres), size=n)
    return centres[idx] + rng.normal(0, sigma_um, size=(n, 3))


def _compartment_labels(rng, g: CellGeometry, n: int, nucleus: bool, punctate: float,
                        pools: _PunctaPools | None, shared: float, channel: int) -> np.ndarray:
    """``n`` labels in the nucleus or the cytosol, a fraction ``punctate`` of them in puncta.

    Channel 1's puncta sit on pool A. Each of channel 2's punctate labels sits on
    pool A with probability ``shared`` (the same bodies as channel 1) and on its own
    pool B otherwise.
    """
    n_punct = int(round(n * punctate)) if pools is not None else 0
    diffuse = _sample_volume(rng, g, n - n_punct, nucleus)
    if n_punct == 0:
        return diffuse
    if channel == 1:
        punct = _place_in_puncta(rng, n_punct, pools.a, pools.sigma_um)
    else:
        on_a = rng.random(n_punct) < shared
        punct = np.concatenate([
            _place_in_puncta(rng, int(on_a.sum()), pools.a, pools.sigma_um),
            _place_in_puncta(rng, int((~on_a).sum()), pools.b, pools.sigma_um),
        ])
    return np.concatenate([diffuse, punct])


def _cell_channel(rng, g: CellGeometry, weights: tuple[float, float, float], total: int,
                  punctate: tuple[float, float], pools: dict[int, _PunctaPools | None], shared: float,
                  channel: int) -> Labels:
    w = np.array(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
    n_nuc, n_cyt, n_mem = (int(round(total * x)) for x in w)
    parts = []
    if n_nuc:
        xyz = _compartment_labels(rng, g, n_nuc, True, punctate[0], pools[NUCLEUS], shared, channel)
        parts.append(Labels(xyz, np.full(len(xyz), NUCLEUS, np.int8)))
    if n_cyt:
        xyz = _compartment_labels(rng, g, n_cyt, False, punctate[1], pools[CYTOSOL], shared, channel)
        parts.append(Labels(xyz, np.full(len(xyz), CYTOSOL, np.int8)))
    if n_mem:
        parts.append(Labels(_sample_membrane(rng, g, n_mem), np.full(n_mem, MEMBRANE, np.int8)))
    return Labels.join(parts)


def _cell(v: dict, rng) -> ScenePoints:
    g = _cell_geometry(v)
    total = 120_000
    p1 = (v["ch1_nuc_puncta"], v["ch1_cyto_puncta"])
    p2 = (v["ch2_nuc_puncta"], v["ch2_cyto_puncta"])
    # Body positions come from their own generator, and only when some channel has
    # puncta there: a cell with no puncta is exactly the specimen it was before.
    pools: dict[int, _PunctaPools | None] = {NUCLEUS: None, CYTOSOL: None}
    body_rng = rng.spawn(1)[0]
    for comp, nucleus, count, size_key, i in ((NUCLEUS, True, NUCLEAR_BODIES, "nuclear_body_nm", 0),
                                              (CYTOSOL, False, CYTOSOLIC_PUNCTA, "cyto_punctum_nm", 1)):
        if p1[i] > 0 or p2[i] > 0:
            d = v[size_key] / 1000
            # Diameter ~ 4 sigma: 95% of a body's labels lie within it.
            pools[comp] = _PunctaPools(_body_centres(body_rng, g, count, nucleus, d / 2),
                                       _body_centres(body_rng, g, count, nucleus, d / 2), d / 4)
    shared = v["shared_puncta"]
    ch1 = _cell_channel(rng, g, (v["ch1_nucleus"], v["ch1_cytosol"], v["ch1_membrane"]), total, p1, pools,
                        shared, 1)
    ch2 = _cell_channel(rng, g, (v["ch2_nucleus"], v["ch2_cytosol"], v["ch2_membrane"]), total, p2, pools,
                        shared, 2)
    dz = 0.4
    z = np.arange(0.0, g.h + dz, dz)
    return ScenePoints(ch1, ch2, CELL_FIELD_UM, CELL_DX_UM, z, g, g.h * 1000, "cell height")


_W = dict(lo=0, hi=1, decimals=2)
CELL = Scene(
    "Model cell",
    "A 3D adherent cell -- dome-shaped, on the coverslip -- with a nucleus, cytosol and "
    "plasma membrane. Each channel's labels are divided among the three compartments. "
    "In widefield, the apical and basal membranes lie over everything in projection.",
    (
        Param("ch1_nucleus", "ch1 in nucleus", default=0.0, **_W),
        Param("ch1_cytosol", "ch1 in cytosol", default=0.0, **_W),
        Param("ch1_membrane", "ch1 on membrane", default=1.0, **_W),
        Param("ch2_nucleus", "ch2 in nucleus", default=0.0, **_W),
        Param("ch2_cytosol", "ch2 in cytosol", default=1.0, **_W),
        Param("ch2_membrane", "ch2 on membrane", default=0.0, **_W),
        Param("ch1_nuc_puncta", "ch1 nucleus in puncta", default=0.0,
              help="fraction of channel 1's nuclear labels in nuclear bodies (speckles, PML bodies, foci); "
                   "0 = diffuse nucleoplasm", **_W),
        Param("ch1_cyto_puncta", "ch1 cytosol in puncta", default=0.0,
              help="fraction of channel 1's cytosolic labels in puncta (vesicles, granules); 0 = diffuse", **_W),
        Param("ch2_nuc_puncta", "ch2 nucleus in puncta", default=0.0,
              help="fraction of channel 2's nuclear labels in nuclear bodies; 0 = diffuse", **_W),
        Param("ch2_cyto_puncta", "ch2 cytosol in puncta", default=0.0,
              help="fraction of channel 2's cytosolic labels in puncta; 0 = diffuse", **_W),
        Param("shared_puncta", "puncta shared", default=0.0,
              help="chance that a channel-2 punctate label sits on one of channel 1's bodies rather than "
                   "its own: 1 = both channels in the same bodies, 0 = separate bodies in the same "
                   "compartment", **_W),
        Param("nuclear_body_nm", "nuclear body size", 100, 2000, 500, " nm", log=True,
              help=f"diameter of each of the {NUCLEAR_BODIES} nuclear bodies per channel"),
        Param("cyto_punctum_nm", "cytosolic punctum size", 30, 1000, 120, " nm", log=True,
              help=f"diameter of each of the {CYTOSOLIC_PUNCTA} cytosolic puncta per channel"),
        Param("cell_length_um", "cell length", 8, 24, 20, " µm", decimals=1),
        Param("cell_width_um", "cell width", 6, 20, 14, " µm", decimals=1),
        Param("cell_height_um", "cell height", 2, 10, 5, " µm", decimals=1,
              help="flat cells put the membranes close to every focal plane"),
        Param("nucleus_diameter_um", "nucleus diameter", 3, 14, 9, " µm", decimals=1),
        Param("nucleus_height", "nucleus height", 1.5, 8, 3.5, " µm", decimals=1),
    ),
    _cell,
    kind="cell",
    focus_range=(0.0, 10.0),
)


SCENES: dict[str, Scene] = {s.name: s for s in (PUNCTA, FILAMENTS, VESICLES, DENSE, STOICHIOMETRY, DOMAINS, CELL)}
DEFAULT_SCENE = PUNCTA.name


def build_scene(name: str, overrides: dict[str, float] | None, seed: int) -> tuple[ScenePoints, dict[str, float]]:
    scene = SCENES[name]
    values = scene.resolve(overrides)
    rng = np.random.default_rng(seed)
    return scene.build(values, rng), values
