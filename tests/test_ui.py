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

"""The widget builds, follows its levers, and shuts down cleanly -- headless."""

import pytest

pytest.importorskip("PySide6", reason="the widget needs the 'ui' extra")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from colocwidget.physics.imaging import Levers  # noqa: E402
from colocwidget.physics.presets import PRESETS  # noqa: E402
from colocwidget.physics.scenes import SCENES  # noqa: E402
from colocwidget.ui import style  # noqa: E402
from colocwidget.ui.controls import LeverPanel, LeverSlider  # noqa: E402
from colocwidget.ui.widget import ColocWidget, ColocWindow  # noqa: E402

FAST = dict(realizations=2)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    windows = []

    def make(levers=None):
        w = ColocWindow(levers or Levers(**FAST))
        windows.append(w)
        return w

    yield make
    for w in windows:
        w.close()
        w.deleteLater()
    QtWidgets.QApplication.processEvents()


def test_window_builds_and_renders(window):
    w = window()
    w.show()
    QtWidgets.QApplication.processEvents()
    widget = w.widget
    assert widget.sim is not None and widget.sweep is not None
    for pane in widget.images.panes():
        assert pane.image is not None
    assert widget.graphs.sweep.spec.series and widget.graphs.ccf.spec.series
    assert widget.graphs.scatter.image is not None
    assert w.grab().width() > 0
    assert "M1" in w.statusBar().currentMessage()


def test_the_widget_embeds_without_a_window(app):
    widget = ColocWidget(Levers(**FAST))
    parent = QtWidgets.QTabWidget()
    parent.addTab(widget, "coloc")
    assert widget.sim is not None
    parent.deleteLater()


def test_a_lever_reaches_the_simulation(window):
    w = window()
    widget = w.widget
    widget.levers_panel.modality.setCurrentText("STED")
    widget.refresh()
    assert widget.sim.levers.modality == "STED"
    assert widget.sim.fwhm_um[0] < 0.1


def test_scene_parameters_are_built_from_the_scene(app):
    panel = LeverPanel(Levers(**FAST))
    assert set(panel.param_sliders) == {p.name for p in SCENES["Puncta clusters"].params}
    panel.param_sliders["colabelled"].valueChanged.emit(1.0)
    assert panel.levers.params["colabelled"] == 1.0
    panel.scene.setCurrentText("Model cell")
    assert set(panel.param_sliders) == {p.name for p in SCENES["Model cell"].params}
    assert panel.levers.scene_params == ()  # a new scene starts from its defaults


def test_objective_preset_sets_na_and_medium(app):
    panel = LeverPanel(Levers(**FAST))
    seen = []
    panel.leversChanged.connect(seen.append)
    panel.objective.setCurrentText("60x / 1.20 water")
    assert (seen[-1].na, seen[-1].medium) == (1.2, "water")


def test_na_is_pulled_below_the_immersion_index(app):
    panel = LeverPanel(Levers(**FAST))
    panel.medium.setCurrentText("air")
    assert panel.levers.na < 1.0


def test_sted_fwhm_only_for_sted(app):
    panel = LeverPanel(Levers(**FAST))
    assert not panel.sted.isEnabled()
    panel.modality.setCurrentText("STED")
    assert panel.sted.isEnabled()


def test_presets_round_trip_through_the_panel(window):
    w = window()
    for name, preset in PRESETS.items():
        w.widget.load_preset(name)
        lv = w.widget.sim.levers
        assert lv.scene == preset.levers.scene, name
        assert lv.params == preset.levers.params, name
        assert lv.modality == preset.levers.modality, name
        assert lv.roi == preset.levers.roi, name
        assert lv.bleed_through == pytest.approx(preset.levers.bleed_through), name


def test_cell_presets_fill_the_compartment_tab(window):
    w = window()
    w.widget.load_preset("3c. Cell: split across nucleus, cytosol and membrane")
    assert w.widget.compartments.table.topLevelItemCount() == 3
    w.widget.load_preset("1a. Interleaved puncta, widefield")
    assert w.widget.compartments.table.topLevelItemCount() == 0


def test_sweep_axis_choice(window):
    w = window()
    graphs = w.widget.graphs
    graphs.axis.setCurrentIndex(graphs.axis.findData("axial"))
    w.widget.refresh()
    assert w.widget.sweep.axis == "axial"
    graphs.axis.setCurrentIndex(graphs.axis.findData("lateral"))
    w.widget.refresh()
    assert w.widget.sweep.axis == "lateral"


def test_a_swept_lever_is_disabled_in_the_left_pane(window):
    w = window()
    graphs, panel = w.widget.graphs, w.widget.levers_panel
    graphs.axis.setCurrentIndex(graphs.axis.findData("lever:pixel_nm"))
    w.widget.refresh()
    assert w.widget.sweep.axis == "lever:pixel_nm"
    assert not panel.pixel.isEnabled() and "x-axis" in panel.pixel.toolTip()
    assert "pixel size" in graphs.sweep.spec.title
    graphs.axis.setCurrentIndex(graphs.axis.findData("param:colabelled"))
    w.widget.refresh()
    assert panel.pixel.isEnabled() and not panel.pixel.toolTip().startswith("on the plot")
    assert not panel.param_sliders["colabelled"].isEnabled()
    assert any(s.label.startswith("truth") for s in graphs.sweep.spec.series)


def test_a_scene_axis_falls_back_when_the_scene_changes(window):
    w = window()
    graphs, panel = w.widget.graphs, w.widget.levers_panel
    graphs.axis.setCurrentIndex(graphs.axis.findData("param:colabelled"))
    w.widget.refresh()
    w.widget.load_preset("2a. Adjacent filaments 120 nm apart, widefield")
    assert graphs.axis_key() == "auto"
    assert graphs.axis.findData("param:separation_nm") >= 0
    assert all(s.isEnabled() for s in panel.param_sliders.values())


def test_sted_fwhm_is_offered_only_for_sted(window):
    w = window()
    graphs = w.widget.graphs
    assert graphs.axis.findData("lever:sted_fwhm_nm") < 0
    w.widget.load_preset("1b. ... the same field at STED (50 nm)")
    assert graphs.axis.findData("lever:sted_fwhm_nm") >= 0


def test_new_frame_changes_noise_not_signal(window):
    w = window()
    widget = w.widget
    first = widget.sim.recorded[0].copy()
    optical = widget.sim.optical[0].copy()
    widget.new_noise()
    assert (widget.sim.recorded[0] != first).any()
    assert (widget.sim.optical[0] == optical).all()


def test_display_modes(window):
    w = window()
    for name in style.MERGES:
        w.widget.set_display(merge_name=name)
    w.widget.set_display(show_roi=False)
    assert w.widget.images.recorded.image is not None


def test_each_channel_alone_and_the_manders_map(window):
    w = window()
    for view in ("channel 1", "channel 2"):
        w.view.setCurrentText(view)
        assert "counts" in w.widget.images.product.title
        img = w.widget.images.recorded.image
        # Only one channel drawn: green (ch1) or magenta (ch2) in green/magenta.
        px = [img.pixelColor(x, y) for x in range(0, img.width(), 7) for y in range(0, img.height(), 7)]
        if view == "channel 1":
            assert all(c.red() == 0 and c.blue() == 0 for c in px)
        else:
            assert all(c.green() == 0 for c in px)
    w.view.setCurrentText("both channels")
    assert "nMDP" in w.widget.images.product.title


def test_objects_view_draws_the_detected_spots(window):
    w = window()
    w.view.setCurrentText("objects (spots)")
    spots = w.widget.images.recorded.spots
    sim = w.widget.sim
    assert len(spots) == sim.spots.n1 + sim.spots.n2 > 0
    assert sum(m for *_, m in spots) == int(sim.spots.matched1.sum() + sim.spots.matched2.sum())
    w.view.setCurrentText("both channels")
    assert w.widget.images.recorded.spots == []


def test_apply_thresholds_blanks_what_manders_ignores(window):
    w = window()
    before = w.widget.images.recorded.image.copy()
    w.threshold_images.setChecked(True)
    after = w.widget.images.recorded.image
    assert after != before
    assert "above T1" in w.widget.images.recorded.caption
    w.threshold_images.setChecked(False)
    assert "above T1" not in w.widget.images.recorded.caption


def test_explanations_wrap_instead_of_scrolling(window):
    w = window()
    for tree in (w.widget.coefficients.rows, w.widget.report_panel):
        assert tree.wordWrap()
        assert tree.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff


def test_manders_form_lever(app):
    panel = LeverPanel(Levers(**FAST))
    panel.manders_form.setCurrentIndex(1)
    assert panel.levers.manders_form.startswith("all signal")


def test_live_mode_starts_and_stops(window):
    w = window()
    w.live.setChecked(True)
    assert w.widget.live_timer.isActive()
    w.live.setChecked(False)
    assert not w.widget.live_timer.isActive()


def test_log_slider_round_trips(app):
    s = LeverSlider(0.1, 10_000, 100.0, log=True)
    s.slider.setValue(s.slider.maximum())
    assert s.value() == pytest.approx(10_000)


def test_an_impossible_setting_shows_a_problem_not_a_crash(window):
    w = window()
    widget = w.widget
    widget.levers_panel.levers = Levers(na=1.45, medium="water", **FAST)
    widget.refresh()
    assert not widget.problem.isHidden()
