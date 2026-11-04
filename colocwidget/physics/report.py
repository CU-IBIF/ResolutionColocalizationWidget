"""The explained numbers: every coefficient, with its equation, your values, and the truth.

The widget's scorecard. Each row is something a demonstrator would point at --
"this r is the PSF talking", "Manders' is counting background", "crosstalk made
this colocalized" -- and each carries the arithmetic so it can be redone on paper.
Where the image disagrees with the molecules, the row says which stage is to
blame: resolution (the noise-free image already disagrees) or noise.
"""

from __future__ import annotations

import math

import numpy as np

from . import optics
from .base import Report, Row, Status
from .imaging import Simulation
from .metrics import MANDERS_FORMS, Coefficients


def _f(v: float, digits: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def _pct(v: float) -> str:
    return "--" if v is None or not math.isfinite(v) else f"{v:.0%}"


# --- optics -------------------------------------------------------------------------


def optics_rows(sim: Simulation) -> list[Row]:
    lv = sim.levers
    sp = sim.scene
    m = optics.MODALITIES[lv.modality]
    rows = []
    for c in (1, 2):
        lam = lv.wavelength_um(c)
        fwhm = sim.fwhm_um[c - 1]
        if m.fixed_fwhm:
            eq = f"STED: set by depletion, {lv.sted_fwhm_nm:.0f} nm (capped at the confocal spot)"
        else:
            eq = (f"FWHM = 0.514 lambda / NA x {m.lateral_factor:.2f} = 0.514 x {lam * 1000:.0f} nm / "
                  f"{lv.na:.2f} x {m.lateral_factor:.2f}")
        status, remedy = Status.INFO, ""
        summary = f"{fwhm * 1000:.0f} nm ({lv.modality})"
        if c == 1 and sp.feature_nm and sp.feature_nm < fwhm * 1000:
            status = Status.WARN
            summary += f" -- wider than the {sp.feature_name} ({sp.feature_nm:.0f} nm)"
            remedy = ("the image cannot tell 'together' from 'within one PSF': raise the resolution "
                      "(pixel reassignment, STED) or report the coefficient as 'within ~FWHM'")
        rows.append(Row(f"lateral resolution, ch{c}", summary, eq, status, remedy))

    f1, f2 = sim.fwhm_um
    if abs(f1 - f2) / f1 > 0.05:
        rows.append(Row(
            "two colours, two PSFs",
            f"ch2's PSF is {f2 / f1:.2f}x ch1's",
            "FWHM scales with lambda: a point source is a wider spot in the redder channel, so even "
            "perfect colocalization gives r < 1 at the edges of small objects",
        ))

    n = lv.n_medium
    if m.sectioning:
        section = lv.section_um(1)
        rows.append(Row(
            "optical section",
            f"{section:.2f} µm FWHM",
            f"sqrt((0.88 lambda / (n - sqrt(n^2 - NA^2)))^2 + (sqrt2 n PH / NA)^2), PH = 1 AU; a plane dz away "
            f"is weighted 1 / (1 + (2 dz / {section:.2f})^2)",
        ))
    else:
        dof = optics.depth_of_field_um(lv.wavelength_um(1), lv.na, n)
        flat = sp.flat
        rows.append(Row(
            "optical section",
            "none: widefield keeps out-of-focus light" + ("" if flat else f" from all {sp.z_slices[-1]:.1f} µm of the cell"),
            f"depth of field lambda n / NA^2 = {dof:.2f} µm, but a plane 2 µm away still contributes all its "
            f"photons, blurred by sigma = dz tan(theta) / 2 = {2 * optics.cone_tan(lv.na, n) / 2:.2f} µm",
            Status.INFO if flat else Status.WARN,
            "" if flat else "membranes above and below the focal plane are projected onto it: section optically "
                            "(confocal) before comparing compartments",
        ))

    shift = math.hypot(lv.shift_x_nm, lv.shift_y_nm)
    if shift or lv.shift_z_nm:
        frac = shift / (f1 * 1000)
        rows.append(Row(
            "channel registration",
            f"ch2 offset by {shift:.0f} nm laterally, {lv.shift_z_nm:.0f} nm axially ({frac:.0%} of the FWHM)",
            "chromatic aberration or misaligned cameras: the CCF peak moves off zero by the shift",
            Status.WARN if frac > 0.2 or abs(lv.shift_z_nm) > 200 else Status.INFO,
            "register the channels on multicolour beads before measuring",
        ))
    else:
        rows.append(Row("channel registration", "perfect (no chromatic shift)", "", Status.PASS))

    nyq = optics.nyquist_pixel_um(min(f1, f2))
    ok = sim.pixel_um <= nyq * 1.05
    rows.append(Row(
        "pixel size",
        f"{sim.pixel_um * 1000:.0f} nm, Nyquist {nyq * 1000:.0f} nm",
        f"Nyquist pixel = FWHM / (2 x 1.029) = {min(f1, f2) * 1000:.0f} / 2.06 nm",
        Status.PASS if ok else Status.WARN,
        "" if ok else "under-sampled: each pixel averages over more than the optics resolve",
    ))
    return rows


# --- signal ---------------------------------------------------------------------------


def signal_rows(sim: Simulation) -> list[Row]:
    lv = sim.levers
    rows = []
    rows.append(Row(
        "signal",
        f"peak {lv.peak_1:.0f} / {lv.peak_2:.0f} photons per pixel (ch1 / ch2), background {lv.background:.0f}",
        "each channel is scaled so its brightest 0.1% of noise-free pixels reach the peak you set",
    ))
    if lv.subtract_background:
        rows.append(Row("background", f"{lv.background:.0f} photons subtracted, negatives set to 0", "", Status.PASS))
    else:
        rows.append(Row(
            "background", f"{lv.background:.0f} photons left in",
            "MOC = sum ab / sqrt(sum a^2 sum b^2) and Manders' sums count background as overlapping signal",
            Status.WARN, "subtract the background (or use thresholded coefficients) before Manders' or MOC",
        ))
    if lv.bleed_through > 0:
        rows.append(Row(
            "bleed-through",
            f"{lv.bleed_through:.0%} of channel 1 appears in channel 2",
            "b_recorded = b + beta a: a copy of channel 1 in channel 2 correlates with channel 1 by construction",
            Status.WARN, "acquire sequentially, narrow the emission filter, or unmix spectrally",
        ))
    if lv.noise:
        nf, me = sim.noise_free.pcc, sim.measured.pcc
        mask = sim.roi_mask
        var_s = [float(o[mask].var()) for o in sim.optical]
        var_n = [float(e[mask].mean()) + lv.read_noise_e**2 for e in sim.expected]
        att = math.sqrt((1 + var_n[0] / var_s[0]) * (1 + var_n[1] / var_s[1])) if min(var_s) > 0 else float("nan")
        rows.append(Row(
            "noise",
            f"r = {_f(sim.repeats.mean('pcc'))} +/- {_f(sim.repeats.sd('pcc'), 3)} over {lv.realizations} frames "
            f"(noise-free {_f(nf)})",
            f"noise attenuates r: r_noise-free / sqrt((1 + var_n,a / var_a)(1 + var_n,b / var_b)) = "
            f"{_f(nf)} / {_f(att)} = {_f(nf / att) if math.isfinite(att) else '--'}; measured {_f(me)}",
            Status.WARN if math.isfinite(att) and att > 1.25 else Status.INFO,
            "more photons (or averaging) before reading r: noise alone pulls every coefficient toward 0",
        ))
    else:
        rows.append(Row("noise", "off: the recorded image is the noise-free image", ""))
    return rows


# --- colocalization ---------------------------------------------------------------------


def _blame(truth: float, nf: float, rec: float) -> str:
    return f"resolution moves it {nf - truth:+.2f}, noise {rec - nf:+.2f}"


def coloc_rows(sim: Simulation) -> list[Row]:
    lv = sim.levers
    t, nf, me = sim.truth, sim.noise_free, sim.measured
    d = f"{t.distance_um * 1000:.0f} nm"
    roi = f"{me.n_pixels:,} px in '{sim.roi_name}'"
    rows = []

    off = abs(me.pcc - t.pcc) > 0.3 if math.isfinite(t.pcc) and math.isfinite(me.pcc) else False
    rows.append(Row(
        Coefficients.LABELS["pcc"],
        f"{_f(me.pcc)}  (noise-free image {_f(nf.pcc)}; molecules {_f(t.pcc)} in {d} voxels)",
        f"r = sum (a - a_)(b - b_) / sqrt(sum (a - a_)^2 sum (b - b_)^2) over {roi}; "
        + _blame(t.pcc, nf.pcc, me.pcc),
        Status.WARN if off else Status.INFO,
        "r measures proportionality within the PSF, not molecular contact: compare at a higher "
        "resolution, or against a randomized (Costes) control" if off else "",
    ))
    rows.append(Row(
        Coefficients.LABELS["srcc"], _f(me.srcc),
        "Pearson's r of the ranks: robust to a few very bright pixels and to non-linear (but monotone) relations",
    ))
    rows.append(Row(
        Coefficients.LABELS["icq"], _f(me.icq),
        "N((a - a_)(b - b_) > 0) / N - 0.5: +0.5 dependent, 0 random, -0.5 segregated (Li et al. 2004)",
    ))
    rows.append(Row(
        Coefficients.LABELS["moc"],
        f"{_f(me.moc)}  (k1 {_f(me.k1)}, k2 {_f(me.k2)})",
        "MOC = sum ab / sqrt(sum a^2 sum b^2); k1 = sum ab / sum a^2. Not mean-subtracted: it rises with "
        "background and with any signal that merely fills the field",
    ))
    th = me.thresholds
    th_eq = f"T1 = {th.ta:.3g}, T2 = {th.tb:.3g} photons"
    if th.mode == "Costes" and math.isfinite(th.slope):
        th_eq += (f"; b = {th.slope:.3g} a + {th.intercept:.3g} (orthogonal fit); T stepped down from the top until "
                  "r of the pixels not above both thresholds first reaches 0")
    # Background noise after subtraction, in the same units as the thresholds.
    sigma = math.sqrt(lv.background + lv.read_noise_e**2) if lv.noise else 0.0
    in_noise = lv.noise and th.mode != "none" and min(th.ta, th.tb) < 3 * sigma
    status, remedy = Status.INFO, ""
    if th.mode == "none":
        status = Status.WARN
        remedy = "with no threshold every pixel above 0 counts as 'present': noise and dim haze inflate M1, M2"
    elif th.note:
        status = Status.WARN
    elif in_noise:
        status = Status.WARN
        remedy = (f"a threshold below 3 sigma of the background noise ({3 * sigma:.3g} photons) admits noise "
                  "pixels: they enter the denominators and pull M1, M2 down (and the co-occurrence of two noise "
                  "pixels pulls them up). More photons, or a less noisy background, lift the threshold clear")
    rows.append(Row(
        "thresholds",
        f"{th.mode}" + (f" -- {th.note}" if th.note else "") + (" -- inside the background noise" if in_noise else ""),
        th_eq + (f"; background noise sigma = sqrt(B + read^2) = {sigma:.3g} photons" if lv.noise else ""),
        status, remedy,
    ))
    all_signal = lv.manders_form == MANDERS_FORMS[1]
    for key, true, label in (("m1", t.m1, "ch1 near ch2"), ("m2", t.m2, "ch2 near ch1")):
        val, clean = me.get(key), nf.get(key)
        off = abs(val - true) > 0.3 if math.isfinite(val) and math.isfinite(true) else False
        if key == "m1":
            eq = "M1 = sum a_i[a_i > T1 and b_i > T2] / " + ("sum a_i" if all_signal else "sum a_i[a_i > T1]")
        else:
            eq = "M2 = sum b_i[a_i > T1 and b_i > T2] / " + ("sum b_i" if all_signal else "sum b_i[b_i > T2]")
        eq += f" ({lv.manders_form}); " + _blame(true, clean, val)
        rows.append(Row(
            Coefficients.LABELS[key],
            f"{_f(val)}  (noise-free {_f(clean)}; molecules {_f(true)}: {label} within {d})",
            eq,
            Status.WARN if off else Status.INFO,
            "Manders' counts co-occurrence in a pixel, and a pixel is ~one PSF wide: two separate "
            "things within it count as colocalized" if off else "",
        ))
    p, rmean, rsd = sim.costes_p
    if math.isfinite(p):
        sig = p > 0.95
        rows.append(Row(
            "Costes randomization",
            f"P = {p:.2f} ({'significant' if sig else 'not significant'})",
            f"ch1 scrambled in PSF-sized blocks: r_random = {_f(rmean)} +/- {_f(rsd, 3)}, observed {_f(me.pcc)}. "
            "Significance says the pattern is not chance; it says nothing about molecular distance",
            Status.PASS if sig else Status.INFO,
        ))
    s = sim.spots
    if s.n1 and s.n2:
        rows.append(Row(
            "spot-based (objects)",
            f"{_pct(s.frac1)} of {s.n1} ch1 spots have a ch2 spot within {s.distance_um * 1000:.0f} nm "
            f"(chance {_pct(s.chance1)}); {_pct(s.frac2)} of {s.n2} ch2 spots (chance {_pct(s.chance2)})",
            "local maxima above the thresholds; distance = the larger FWHM. Chance = 1 - exp(-rho pi d^2) "
            "for the other channel's spot density rho",
            Status.WARN if math.isfinite(s.chance1) and s.chance1 > 0.3 else Status.INFO,
            "spots are so dense that most would be 'colocalized' by chance" if s.chance1 > 0.3 else "",
        ))
    xs, ys = sim.ccf
    if np.isfinite(ys).any():
        # On a flat top, report the shift nearest zero rather than an arbitrary maximum.
        top = np.nanmax(ys)
        near = np.nonzero(ys >= top - 0.005)[0]
        i = int(near[np.argmin(np.abs(xs[near]))])
        mid = int(np.argmin(np.abs(xs)))
        rows.append(Row(
            "cross-correlation (van Steensel)",
            f"peak r = {_f(ys[i])} at dx = {xs[i] * 1000:+.0f} nm; at 0: {_f(ys[mid])}",
            "r as ch2 slides along x. A peak at 0 as wide as the PSF: colocalized. A peak off 0: a "
            "systematic offset. A dip at 0: exclusion",
            Status.WARN if abs(xs[i]) > 0.5 * sim.fwhm_um[0] and ys[i] > 0.2 else Status.INFO,
        ))
    return rows


# --- truth --------------------------------------------------------------------------------


def truth_rows(sim: Simulation) -> list[Row]:
    t = sim.truth
    d = t.distance_um * 1000
    rows = [
        Row(
            "molecular colocalization",
            f"{_pct(t.m1)} of ch1 labels within {d:.0f} nm of a ch2 label; {_pct(t.m2)} the other way",
            "M1_true = #{i : min_j |p_i - q_j| <= d} / N1, from the label positions themselves "
            "-- what the image is trying to report",
        ),
        Row(
            "molecular Pearson's r",
            _f(t.pcc),
            f"r of the two label counts in {d:.0f} nm voxels: what a noise-free microscope with "
            f"{d:.0f} nm resolution would measure",
        ),
        Row("labels", f"{t.n1:,} ch1, {t.n2:,} ch2", sim.levers.scene_model.description),
    ]
    if t.by_compartment and sim.scene.cell is not None:
        share1, share2 = t.compartment_overlap()
        rows.insert(1, Row(
            "compartment colocalization",
            f"{_pct(share1)} of ch1 labels are in a compartment that holds ch2; {_pct(share2)} the other way",
            "sum over compartments of the channel's share, where the other channel is present at all. "
            "Diffuse proteins sharing a compartment rarely sit within molecular distance of each other: "
            "the two truths answer different questions",
        ))
    return rows


def compartment_rows(sim: Simulation) -> list[Row]:
    rows = []
    for c in sim.compartments:
        bits = []
        status, remedy = Status.INFO, ""
        for ch, true, seen in ((1, c.true_1, c.seen_1), (2, c.true_2, c.seen_2)):
            bits.append(f"ch{ch} {_pct(true)} of labels / {_pct(seen)} of signal")
            if true < 0.02 and seen > 0.10:
                status = Status.WARN
                remedy = (f"ch{ch} has no labels in the {c.name} but {seen:.0%} of its in-focus signal appears "
                          "there: light from above and below the plane (widefield) or blur across the boundary")
        coeff = (f"; r {_f(c.coeffs.pcc)}, M1 {_f(c.coeffs.m1)}, M2 {_f(c.coeffs.m2)}" if c.coeffs else
                 "; not in the focal plane")
        rows.append(Row(
            c.name, "   ".join(bits) + coeff,
            "labels: true 3D share of the channel; signal: share of the cell's recorded signal in this "
            "region at the focal plane",
            status, remedy,
        ))
    return rows


def build_report(sim: Simulation) -> dict[str, Report]:
    sections = {
        "optics and sampling": Report(optics_rows(sim)),
        "signal and noise": Report(signal_rows(sim)),
        "colocalization": Report(coloc_rows(sim)),
        "the truth": Report(truth_rows(sim)),
    }
    if sim.compartments:
        sections["compartments"] = Report(compartment_rows(sim))
    return sections


def format_report(sections: dict[str, Report]) -> str:
    out = []
    for name, rep in sections.items():
        out.append(f"-- {name} --")
        out.append(rep.format())
        out.append("")
    return "\n".join(out).rstrip()
