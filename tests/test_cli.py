import pytest

from colocwidget.__main__ import main, parse_assignment
from colocwidget.physics.imaging import Levers, Simulator
from colocwidget.physics.presets import PRESETS
from colocwidget.physics.report import build_report


def test_default_report_runs(capsys):
    assert main(["--realizations", "3"]) == 0
    out = capsys.readouterr().out
    for heading in ("-- optics and sampling --", "-- signal and noise --", "-- colocalization --", "-- the truth --"):
        assert heading in out


def test_set_parses_levers_and_scene_parameters():
    lv = parse_assignment(Levers(), "na=1.2")
    assert lv.na == 1.2
    lv = parse_assignment(lv, "modality=STED")
    assert lv.modality == "STED"
    lv = parse_assignment(lv, "colabelled=0.5")
    assert lv.params["colabelled"] == 0.5
    lv = parse_assignment(lv, "noise=off")
    assert lv.noise is False
    lv = parse_assignment(lv, "realizations=5")
    assert lv.realizations == 5 and isinstance(lv.realizations, int)


def test_changing_scene_resets_its_parameters():
    lv = parse_assignment(Levers(), "colabelled=1")
    lv = parse_assignment(lv, "scene=Model cell")
    assert lv.scene_params == () and lv.focus_um is None
    assert "ch1_membrane" in lv.params


def test_unknown_names_are_errors():
    with pytest.raises(SystemExit):
        parse_assignment(Levers(), "nonsense=1")
    with pytest.raises(SystemExit):
        parse_assignment(Levers(), "scene=Nowhere")


def test_impossible_na_is_reported_not_raised(capsys):
    assert main(["--set", "medium=air", "--set", "na=1.4"]) == 2
    assert "exceeds" in capsys.readouterr().out


def test_unknown_preset(capsys):
    assert main(["--preset", "nope"]) == 2


def test_listings(capsys):
    assert main(["--list-presets"]) == 0
    assert "1a." in capsys.readouterr().out
    assert main(["--list-scenes"]) == 0
    assert "colabelled" in capsys.readouterr().out


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_runs_and_reports(name):
    sim = Simulator().run(PRESETS[name].levers.with_(realizations=2))
    sections = build_report(sim)
    assert all(rep.rows for rep in sections.values())
    if sim.scene.cell is not None:
        assert "compartments" in sections


def test_a_cell_preset_reports_compartments(capsys):
    assert main(["--preset", "3c. Cell: split across nucleus, cytosol and membrane", "--realizations", "2"]) == 0
    out = capsys.readouterr().out
    assert "-- compartments --" in out and "compartment colocalization" in out
