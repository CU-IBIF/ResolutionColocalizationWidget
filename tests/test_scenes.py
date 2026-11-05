import numpy as np
import pytest

from colocwidget.physics import metrics as M
from colocwidget.physics.scenes import CELL, MEMBRANE, NUCLEUS, SCENES, build_scene


@pytest.mark.parametrize("name", list(SCENES))
def test_every_scene_builds_inside_its_field(name):
    sp, values = build_scene(name, None, 1)
    assert len(sp.ch1) > 0 and len(sp.ch2) > 0
    for labels in (sp.ch1, sp.ch2):
        xy = labels.xyz[:, :2]
        assert (xy > -0.05).all() and (xy < sp.field_um + 0.05).all()
    assert values == SCENES[name].defaults()


def test_parameters_are_clamped_and_unknown_ones_refused():
    scene = SCENES["Puncta clusters"]
    assert scene.resolve({"colabelled": 7})["colabelled"] == 1.0
    with pytest.raises(ValueError, match="no parameter"):
        scene.resolve({"nonsense": 1})


def test_the_same_seed_gives_the_same_specimen():
    a, _ = build_scene("Adjacent filaments", None, 3)
    b, _ = build_scene("Adjacent filaments", None, 3)
    c, _ = build_scene("Adjacent filaments", None, 4)
    assert np.array_equal(a.ch1.xyz, b.ch1.xyz)
    assert not np.array_equal(a.ch1.xyz, c.ch1.xyz)


def test_colabelled_fraction_is_the_molecular_truth():
    d = 0.025
    lo, _ = build_scene("Puncta clusters", {"colabelled": 0.0}, 1)
    hi, _ = build_scene("Puncta clusters", {"colabelled": 1.0}, 1)
    assert M.fraction_within(lo.ch1.xyz, lo.ch2.xyz, d) < 0.25
    assert M.fraction_within(hi.ch1.xyz, hi.ch2.xyz, d) > 0.9


def test_filament_separation_is_respected():
    sp, _ = build_scene("Adjacent filaments", {"separation_nm": 300}, 1)
    assert M.fraction_within(sp.ch1.xyz, sp.ch2.xyz, 0.1) < 0.1
    same, _ = build_scene("Adjacent filaments", {"separation_nm": 0}, 1)
    assert M.fraction_within(same.ch1.xyz, same.ch2.xyz, 0.05) > 0.8


def test_cell_compartments_follow_the_weights():
    sp, _ = build_scene(CELL.name, {"ch1_nucleus": 0.25, "ch1_cytosol": 0.25, "ch1_membrane": 0.5}, 1)
    comp = sp.ch1.compartment
    assert (comp == NUCLEUS).mean() == pytest.approx(0.25, abs=0.01)
    assert (comp == MEMBRANE).mean() == pytest.approx(0.5, abs=0.01)
    g = sp.cell
    nuc = sp.ch1.xyz[comp == NUCLEUS]
    assert g.inside_nucleus(nuc).all() and g.inside_cell(nuc).all()
    mem = sp.ch1.xyz[comp == MEMBRANE]
    # Membrane labels sit on the surface: the apical dome or the basal disc.
    x = (mem[:, 0] - g.centre[0]) / g.a
    y = (mem[:, 1] - g.centre[1]) / g.b
    rho = np.sqrt(x**2 + y**2 + (mem[:, 2] / g.h) ** 2)
    on_surface = (np.abs(rho - 1) < 0.01) | (mem[:, 2] < 0.05)
    assert on_surface.all()


def test_cell_sections_shrink_toward_the_top():
    sp, _ = build_scene(CELL.name, None, 1)
    g = sp.cell
    assert g.cell_section(0.1)[0] > g.cell_section(g.h * 0.9)[0] > 0
    assert g.cell_section(g.h + 1) == (0.0, 0.0)
