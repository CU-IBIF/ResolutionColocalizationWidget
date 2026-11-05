"""The lessons, as assertions: if one of these fails, a preset no longer makes its point."""

import numpy as np
import pytest

from colocwidget.physics.imaging import Levers, Simulator
from colocwidget.physics.presets import PRESETS


@pytest.fixture(scope="module")
def sim():
    return Simulator()


def run(sim, name_or_levers, **changes):
    lv = PRESETS[name_or_levers].levers if isinstance(name_or_levers, str) else name_or_levers
    return sim.run(lv.with_(realizations=4, **changes))


# --- case 1: unresolved collections -------------------------------------------------


def test_interleaved_puncta_look_colocalized_at_widefield_but_are_not(sim):
    s = run(sim, "1a. Interleaved puncta, widefield")
    assert s.truth.m1 < 0.25 and abs(s.truth.pcc) < 0.1
    assert s.noise_free.pcc > 0.5
    assert s.measured.m1 > 0.6 and s.measured.m2 > 0.6


def test_sted_resolves_the_interleaving(sim):
    wf = run(sim, "1a. Interleaved puncta, widefield")
    sted = run(sim, "1b. ... the same field at STED (50 nm)")
    assert sted.measured.pcc < wf.measured.pcc - 0.3
    assert sted.measured.m1 < wf.measured.m1 - 0.2


def test_widefield_cannot_tell_interleaved_from_colabelled_but_sted_can(sim):
    inter_wf = run(sim, "1a. Interleaved puncta, widefield")
    co_wf = run(sim, "1c. Co-labelled bodies, widefield")
    inter_sted = run(sim, "1b. ... the same field at STED (50 nm)")
    co_sted = run(sim, "1d. ... the same field at STED (50 nm)")
    assert co_wf.truth.m1 > 0.9
    # The molecular truths differ by ~0.9; the widefield M1s by far less than that.
    assert abs(co_wf.measured.m1 - inter_wf.measured.m1) < 0.3
    assert co_sted.measured.pcc - inter_sted.measured.pcc > 0.6


def test_the_resolution_sweep_falls_toward_the_truth(sim):
    lv = PRESETS["1a. Interleaved puncta, widefield"].levers
    sw = sim.sweep(lv, "lateral")
    pcc = sw.values["pcc"]
    assert sw.axis == "lateral" and np.all(np.diff(sw.x) > 0) and sw.unit == "nm"
    assert pcc[0] < 0.3 and pcc[-1] > 0.6  # 30 nm vs 1.2 um
    assert sw.here == pytest.approx(lv.fwhm_um(1) * 1000)


# --- case 2: distribution and morphology ------------------------------------------


def test_adjacent_filaments_merge_at_widefield_and_split_at_sted(sim):
    wf = run(sim, "2a. Adjacent filaments 120 nm apart, widefield")
    sted = run(sim, "2b. ... resolved by STED")
    assert wf.truth.m1 < 0.1
    assert wf.measured.pcc > 0.45 and sted.measured.pcc < 0.2


def test_dense_independent_signals_inflate_manders_not_pearson(sim):
    s = run(sim, "2d. Dense independent signals: Manders' inflated")
    assert abs(s.measured.pcc) < 0.1
    assert s.measured.m1 > 0.95 and s.measured.m2 > 0.95
    costes = run(sim, "2d. Dense independent signals: Manders' inflated", threshold="Costes")
    assert costes.measured.m1 < 0.7


def test_shared_territory_correlates_without_binding(sim):
    s = run(sim, "2e. Shared territory: correlation without binding")
    assert s.measured.pcc > 0.5 and s.truth.m1 < 0.3


def test_stoichiometry_splits_pearson_from_manders(sim):
    fixed = run(sim, "2f. Variable stoichiometry: Pearson's vs Manders'", params={"ratio_spread": 0.0})
    varied = run(sim, "2f. Variable stoichiometry: Pearson's vs Manders'")
    assert fixed.truth.m1 > 0.9 and varied.truth.m1 > 0.9
    assert varied.measured.pcc < fixed.measured.pcc - 0.2
    assert varied.measured.m1 > 0.6


def test_exclusive_domains_anticorrelate_until_blurred(sim):
    big = run(sim, "2g. Exclusive domains, resolved")
    small = run(sim, "2h. Exclusive domains below the resolution")
    assert big.measured.pcc < -0.8 and big.measured.icq < -0.35
    assert small.measured.pcc > big.measured.pcc + 0.4


def test_a_chromatic_shift_moves_the_ccf_peak(sim):
    s = run(sim, "2i. Chromatic shift (150 nm)")
    xs, ys = s.ccf
    assert xs[np.nanargmax(ys)] * 1000 == pytest.approx(150, abs=s.pixel_um * 1000)
    aligned = run(sim, "2i. Chromatic shift (150 nm)", shift_x_nm=0.0)
    assert aligned.measured.pcc > s.measured.pcc + 0.15


def test_bleed_through_manufactures_correlation(sim):
    clean = run(sim, "1b. ... the same field at STED (50 nm)")
    bled = run(sim, "2j. Bleed-through (30%) manufactures colocalization")
    assert bled.measured.pcc > clean.measured.pcc + 0.15
    assert bled.truth.m1 == pytest.approx(clean.truth.m1)


# --- case 3: cells and compartments -----------------------------------------------


def test_confocal_moves_the_membrane_back_to_the_rim(sim):
    wf = run(sim, "3a. Cell: membrane vs cytosol, widefield")
    conf = run(sim, "3b. ... the same cell, confocal section")
    seen = {k: {c.name: c for c in v.compartments} for k, v in (("wf", wf), ("conf", conf))}
    assert seen["wf"]["plasma membrane"].seen_1 < 0.3
    assert seen["conf"]["plasma membrane"].seen_1 > seen["wf"]["plasma membrane"].seen_1 + 0.2
    assert seen["wf"]["nucleus"].seen_1 > 2 * seen["conf"]["nucleus"].seen_1


def test_split_compartments_whole_cell_vs_nucleus(sim):
    whole = run(sim, "3c. Cell: split across nucleus, cytosol and membrane")
    assert whole.truth.compartment_overlap() == pytest.approx((0.5, 0.5), abs=0.01)
    assert whole.measured.pcc > 0.8
    rows = {c.name: c for c in whole.compartments}
    assert rows["nucleus"].true_1 == pytest.approx(0.5, abs=0.01)
    assert rows["cytosol"].true_1 < 0.01 and rows["cytosol"].true_2 == pytest.approx(0.5, abs=0.01)
    nucleus = run(sim, "3d. ... measured in the nucleus only")
    assert nucleus.roi_name == "nucleus" and nucleus.roi_mask.sum() < whole.roi_mask.sum()


def test_widefield_contaminates_the_nuclear_roi(sim):
    wf = run(sim, "3e. Nuclear vs cytosolic, widefield: the nuclear ROI is contaminated")
    conf = run(sim, "3e. Nuclear vs cytosolic, widefield: the nuclear ROI is contaminated", modality="confocal (1 AU)")
    nuc_wf = {c.name: c for c in wf.compartments}["nucleus"]
    nuc_conf = {c.name: c for c in conf.compartments}["nucleus"]
    assert nuc_wf.true_2 < 0.01
    assert nuc_wf.seen_2 > 0.15 and nuc_conf.seen_2 < nuc_wf.seen_2 * 0.7


def test_the_empty_background_inflates_whole_field_r(sim):
    field = run(sim, "3f. Whole field vs cell: the empty background inflates r")
    cell = run(sim, "3f. Whole field vs cell: the empty background inflates r", roi="cell")
    assert field.measured.pcc > cell.measured.pcc + 0.2


def test_the_axial_sweep_ends_at_widefield(sim):
    lv = PRESETS["3a. Cell: membrane vs cytosol, widefield"].levers
    sw = sim.sweep(lv)  # auto: axial for a cell
    assert sw.axis == "axial" and sw.here == sw.x[-1]
    conf = sim.sweep(PRESETS["3b. ... the same cell, confocal section"].levers)
    assert conf.here < 1.0


# --- the machinery ------------------------------------------------------------------


def test_noise_spreads_and_averages(sim):
    lv = Levers(realizations=12)
    s = sim.run(lv)
    assert 0 < s.repeats.sd("pcc") < 0.05
    quiet = sim.run(lv.with_(noise=False, realizations=0))
    assert quiet.measured.pcc == pytest.approx(quiet.noise_free.pcc)


def test_a_new_noise_seed_changes_the_frame_not_the_scene(sim):
    a = sim.run(Levers(realizations=0))
    b = sim.run(Levers(realizations=0, noise_seed=1))
    assert (a.recorded[0] != b.recorded[0]).any()
    assert np.array_equal(a.optical[0], b.optical[0])


def test_peaks_set_the_brightness(sim):
    s = sim.run(Levers(peak_1=500.0, realizations=0))
    assert np.percentile(s.optical[0], 99.9) == pytest.approx(500.0, rel=1e-6)


def test_background_without_subtraction_inflates_moc(sim):
    sub = sim.run(Levers(realizations=0, background=50.0))
    raw = sim.run(Levers(realizations=0, background=50.0, subtract_background=False))
    assert raw.measured.moc > sub.measured.moc


def test_a_roi_that_does_not_exist_falls_back(sim):
    s = sim.run(Levers(roi="nucleus", realizations=0))
    assert s.roi_name == "whole field" and any("nucleus" in n for n in s.notes)


def test_pixel_is_never_finer_than_the_scene_grid(sim):
    s = sim.run(Levers(scene="Model cell", pixel_nm=20.0, realizations=0))
    assert s.pixel_um == pytest.approx(0.05)


def test_impossible_na_is_refused(sim):
    with pytest.raises(ValueError, match="exceeds"):
        sim.run(Levers(na=1.4, medium="air"))


# --- sweeping any lever or scene parameter -------------------------------------------


def test_every_axis_of_a_flat_scene_sweeps(sim):
    from colocwidget.physics.imaging import sweep_axes

    lv = Levers()
    axes = sweep_axes(lv)
    assert "lever:na" in axes and "param:colabelled" in axes
    assert "lever:sted_fwhm_nm" not in axes  # only STED has a FWHM of its own
    assert "lever:threshold_fraction" not in axes  # only for a fixed-fraction threshold
    for key, spec in axes.items():
        sw = sim.sweep(lv, key, points=4)
        assert sw.x[0] == pytest.approx(spec.lo * spec.scale if spec.kind == "lever" else spec.lo)
        assert np.isfinite(sw.values["pcc"]).any(), key
        assert sw.here == pytest.approx(spec.current(lv)), key


def test_sweeping_the_colabelled_fraction_sweeps_the_truth_too(sim):
    sw = sim.sweep(Levers(), "param:colabelled", points=5)
    assert sw.truth is not None
    assert sw.truth["m1"][0] < 0.25 and sw.truth["m1"][-1] > 0.9
    assert sw.values["pcc"][-1] > sw.values["pcc"][0] + 0.2


def test_a_lever_sweep_leaves_the_truth_alone_and_the_main_view_cached(sim):
    lv = Levers()
    sim.run(lv.with_(realizations=0))
    key = sim._scene_key
    sw = sim.sweep(lv, "param:cluster_radius_nm", points=3)
    assert sim._scene_key == key  # scene sweeps run on their own simulator
    na = sim.sweep(lv, "lever:na", points=4)
    assert na.truth is None and na.x[-1] <= 1.49
    assert sim.sweep(lv.with_(medium="water", na=1.2), "lever:na", points=3).x[-1] <= 1.333
    assert sw.values["pcc"][0] > sw.values["pcc"][-1]  # tight clusters look colocalized


def test_bleed_through_is_shown_in_percent(sim):
    sw = sim.sweep(Levers(bleed_through=0.1), "lever:bleed_through", points=3)
    assert sw.unit == "%" and sw.x[-1] == pytest.approx(50) and sw.here == pytest.approx(10)


def test_cell_compartment_sweep_tracks_the_shared_truth(sim):
    lv = Levers(scene="Model cell", modality="confocal (1 AU)", roi="cell")
    sw = sim.sweep(lv, "param:ch2_membrane", points=3)
    assert sw.truth["shared"][0] == pytest.approx(0.0) and sw.truth["shared"][-1] > 0.4


def test_a_fully_vesicular_cytosol_builds(sim):
    s = sim.run(Levers(scene="Model cell", realizations=0).with_(params={"ch1_cytosol": 1.0, "ch1_membrane": 0.0, "ch1_cyto_puncta": 1.0, "ch2_cyto_puncta": 1.0, "ch2_nucleus": 1.0, "ch2_nuc_puncta": 1.0}))
    assert s.truth.n2 > 0


# --- puncta in the nucleus and the cytosol ---------------------------------------------


def test_nuclear_bodies_separate_vs_shared(sim):
    separate = run(sim, "3g. Nuclear bodies: same compartment, separate bodies")
    same = run(sim, "3h. ... the same bodies")
    assert separate.truth.compartment_overlap() == pytest.approx((1.0, 1.0))
    assert abs(separate.measured.pcc) < 0.2 and separate.measured.m1 < 0.4
    assert same.measured.pcc > 0.85 and same.measured.m1 > 0.85


def test_puncta_stay_in_their_compartment_and_fractions_hold():
    from colocwidget.physics.scenes import CELL, CYTOSOL, NUCLEUS, build_scene

    sp, _ = build_scene(CELL.name, {"ch1_nucleus": 0.5, "ch1_cytosol": 0.5, "ch1_membrane": 0.0,
                                    "ch1_nuc_puncta": 1.0, "ch1_cyto_puncta": 0.5}, 1)
    comp = sp.ch1.compartment
    assert (comp == NUCLEUS).mean() == pytest.approx(0.5, abs=0.01)
    nuc = sp.ch1.xyz[comp == NUCLEUS]
    assert sp.cell.inside_nucleus(nuc).mean() > 0.99
    # All-punctate nuclear labels crowd into 30 bodies: far denser locally than diffuse ones.
    import numpy as np
    from colocwidget.physics import metrics as M

    diffuse, _ = build_scene(CELL.name, {"ch1_nucleus": 0.5, "ch1_cytosol": 0.5, "ch1_membrane": 0.0}, 1)
    near = M.fraction_within(nuc[:2000], nuc[2000:12000], 0.03)
    dnuc = diffuse.ch1.xyz[diffuse.ch1.compartment == NUCLEUS]
    assert near > 5 * M.fraction_within(dnuc[:2000], dnuc[2000:12000], 0.03)
    assert (sp.ch1.compartment == CYTOSOL).sum() > 0 and np.isfinite(sp.ch1.xyz).all()


def test_no_puncta_leaves_the_specimen_unchanged():
    # Body positions are drawn only when some channel has puncta, so the presets'
    # specimens are exactly what they were before puncta existed.
    import numpy as np

    from colocwidget.physics.scenes import CELL, build_scene

    a, _ = build_scene(CELL.name, None, 1)
    b, _ = build_scene(CELL.name, {"nuclear_body_nm": 900.0, "shared_puncta": 1.0}, 1)
    assert np.array_equal(a.ch1.xyz, b.ch1.xyz) and np.array_equal(a.ch2.xyz, b.ch2.xyz)
