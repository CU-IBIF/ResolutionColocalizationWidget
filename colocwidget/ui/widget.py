"""The colocalization widget: levers, images, the resolution sweep and the coefficients.

:class:`ColocWidget` is a plain ``QWidget`` so MicroscopeWidget can drop it into a
tab or a dock as-is; :class:`ColocWindow` adds the toolbar and status bar for
standalone use, in NoiseWidget's workspace style.

Two clocks, as in NoiseWidget:

* **Lever changes** are debounced (40 ms) and recompute only what they touch. The
  scene and its per-plane FFTs are cached; the sum over planes is cached on the
  axial optics; the resolution sweep on everything but the photons and the noise.
* **Live mode** re-rolls the noise on a timer, so the user sees which part of a
  coefficient is the specimen and which part is the frame.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..physics.base import Report, Status
from ..physics.imaging import Levers, Simulation, Simulator
from ..physics.metrics import Coefficients, nmdp
from ..physics.presets import PRESETS
from ..physics.report import build_report
from . import style
from .controls import LeverPanel
from .panes import Cytofluorogram, ImageGrid, diverging, merge, outline_of, scale
from .plots import Band, Marker, PlotCanvas, PlotSpec, Series

SWEEP_KEYS = ("pcc", "m1", "m2", "moc")


def _fmt(v: float, digits: int = 2) -> str:
    return "--" if v is None or not math.isfinite(v) else f"{v:.{digits}f}"


# --- the graphs under the images -------------------------------------------------


class Graphs(QtWidgets.QWidget):
    """Coefficients against resolution, the cross-correlation, the cytofluorogram."""

    axisChanged = QtCore.Signal()
    AXES = {"auto": "auto (lateral; optical section for a cell)", "lateral": "lateral FWHM",
            "axial": "optical section"}

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("sweep:"))
        self.axis = QtWidgets.QComboBox()
        for key, label in self.AXES.items():
            self.axis.addItem(label, key)
        self.axis.setToolTip(
            "lateral: the PSF width, from STED to a low-NA objective.\n"
            "optical section: from a thin confocal section to widefield, where every plane adds."
        )
        row.addWidget(self.axis)
        row.addStretch(1)
        outer.addLayout(row)
        layout = QtWidgets.QHBoxLayout()
        outer.addLayout(layout, 1)
        self.sweep = PlotCanvas()
        self.sweep.setToolTip(
            "every coefficient as one kind of resolution changes and nothing else does, at your photon "
            "budget. Dashed lines: the truth. Dots: this recorded frame."
        )
        self.ccf = PlotCanvas()
        self.scatter = Cytofluorogram()
        layout.addWidget(self.sweep, 5)
        layout.addWidget(self.ccf, 3)
        layout.addWidget(self.scatter, 3)
        self.axis.currentIndexChanged.connect(lambda _i: self.axisChanged.emit())

    def axis_key(self) -> str:
        return self.axis.currentData()

    def show_simulation(self, sim: Simulation, sweep) -> None:
        lateral = sweep.axis == "lateral"
        xs = sweep.x_um * (1000 if lateral else 1)
        here = sweep.here_um * (1000 if lateral else 1)
        series = []
        for k in SWEEP_KEYS:
            series.append(Series(xs, sweep.values[k], style.COEF_COLORS[k], Coefficients.LABELS[k], 2.5))
        rec = sim.measured
        for k in SWEEP_KEYS:
            series.append(Series(np.array([here]), np.array([rec.get(k)]), style.COEF_COLORS[k], "", 4,
                                 kind="points", hover=False))
        t = sim.truth
        if sim.scene.cell is not None:
            share1, _ = t.compartment_overlap()
            markers = [Marker(share1, style.COEF_COLORS["m1"], "ch1 in shared compartments (truth)",
                              vertical=False)]
        else:
            markers = [
                Marker(t.pcc, style.COEF_COLORS["pcc"], "molecular r", vertical=False),
                Marker(t.m1, style.COEF_COLORS["m1"], f"molecular M1 (within {t.distance_um * 1000:.0f} nm)",
                       vertical=False),
            ]
        markers.insert(0, Marker(here, "#e6edf3", "you are here"))
        bands = []
        sp = sim.scene
        if lateral:
            if sp.feature_nm and xs[0] < sp.feature_nm < xs[-1]:
                bands.append(Band(sp.feature_nm, xs[-1], style.STATUS_COLORS[Status.WARN], "unresolved"))
            title, xlabel, unit = "coefficients vs PSF width", "lateral FWHM, channel 1 (nm)", "nm"
        else:
            bands.append(Band(xs[-2] * 1.05, xs[-1] * 1.2, style.STATUS_COLORS[Status.WARN], "widefield"))
            title, xlabel, unit = "coefficients vs optical section", "optical-section FWHM (µm)", "µm"
        self.sweep.set_spec(PlotSpec(
            title=title + " (all else fixed)", xlabel=xlabel, ylabel="coefficient", xlog=True,
            xrange=(xs[0], xs[-1] * 1.2), yrange=(-1.05, 1.05), series=series, markers=markers, bands=bands,
            x_unit=unit, legend_left=True,
        ))

        cx, cy = sim.ccf
        self.ccf.set_spec(PlotSpec(
            title="cross-correlation (van Steensel)", xlabel="ch2 shift dx (nm)", ylabel="Pearson's r",
            series=[Series(cx * 1000, cy, style.COEF_COLORS["pcc"], "r(dx)", 2.5)],
            markers=[Marker(0.0, "#5c6370", ""), Marker(sim.levers.shift_x_nm, style.COEF_COLORS["m1"], "true shift")
                     if sim.levers.shift_x_nm else Marker(0.0, "#5c6370", "")],
            bands=[Band(-here / 2, here / 2, "#5c6370", "")],
            yrange=(min(-0.1, float(np.nanmin(cy)) - 0.05) if np.isfinite(cy).any() else -1, 1.05),
            x_unit="nm", legend=False,
        ))

        a, b = sim.analysed
        m = sim.roi_mask
        th = sim.measured.thresholds
        self.scatter.show_pairs(a[m], b[m], th.ta, th.tb, th.slope, th.intercept,
                                f"cytofluorogram ({th.mode} thresholds)")


# --- the coefficient table ---------------------------------------------------------


class CoefficientPanel(QtWidgets.QWidget):
    """Every coefficient at the three stages, then the explained rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.headline = QtWidgets.QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-weight: 600; padding: 4px;")
        layout.addWidget(self.headline)
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["coefficient", "molecules", "noise-free", "recorded", "± frames"])
        self.table.setRootIsDecorated(False)
        self.table.setColumnWidth(0, 170)
        for i in (1, 2, 3, 4):
            self.table.setColumnWidth(i, 74)
        self.table.setToolTip(
            "molecules: from the label positions (within the 'together' distance).\n"
            "noise-free: the image the optics and pixels form, before photon counting.\n"
            "recorded: this frame; ± is the spread over repeat frames."
        )
        layout.addWidget(self.table, 2)
        self.rows = QtWidgets.QTreeWidget()
        self.rows.setHeaderLabels(["", "value"])
        self.rows.setColumnWidth(0, 190)
        layout.addWidget(self.rows, 3)

    def show_simulation(self, sim: Simulation, report: Report) -> None:
        t, nf, me, reps = sim.truth, sim.noise_free, sim.measured, sim.repeats
        truth_of = {"pcc": t.pcc, "m1": t.m1, "m2": t.m2}
        self.table.clear()
        for k in Coefficients.KEYS:
            sd = reps.sd(k) if k in reps.values else float("nan")
            item = QtWidgets.QTreeWidgetItem([
                Coefficients.LABELS[k], _fmt(truth_of.get(k, float("nan"))), _fmt(nf.get(k)), _fmt(me.get(k)),
                _fmt(sd, 3),
            ])
            item.setForeground(0, QtGui.QColor(style.COEF_COLORS.get(k, style.TEXT)))
            for col in (1, 2, 3, 4):
                item.setTextAlignment(col, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.table.addTopLevelItem(item)
        if sim.scene.cell is not None:
            # In a cell the question is usually "same compartment?", not "same molecule?".
            share1, share2 = t.compartment_overlap()
            truth_text = f"compartments shared: {share1:.0%} of ch1, {share2:.0%} of ch2"
            gap = abs(me.m1 - share1) if math.isfinite(me.m1) else 0.0
            verdict = ("M1 overstates the shared compartments" if me.m1 - share1 > 0.3 else
                       "M1 understates the shared compartments" if share1 - me.m1 > 0.3 else
                       "M1 is close to the shared-compartment share")
        else:
            truth_text = f"molecules: r {_fmt(t.pcc)}, M1 {_fmt(t.m1)}"
            verdict = "the image agrees with the molecules"
            gap = abs(me.pcc - t.pcc) if math.isfinite(t.pcc) else 0.0
            mgap = abs(me.m1 - t.m1) if math.isfinite(t.m1) and math.isfinite(me.m1) else 0.0
            if max(gap, mgap) > 0.25:
                res = abs(nf.pcc - t.pcc) + abs(nf.m1 - t.m1)
                noise = abs(me.pcc - nf.pcc) + abs(me.m1 - nf.m1)
                verdict = ("the image disagrees with the molecules -- mostly because of "
                           + ("resolution" if res >= noise else "noise"))
        self.headline.setText(
            f"r {_fmt(me.pcc)} · M1 {_fmt(me.m1)} · M2 {_fmt(me.m2)} in '{sim.roi_name}'   —   "
            f"{truth_text}. {verdict}."
        )
        _fill(self.rows, report.rows)


def _fill(tree: QtWidgets.QTreeWidget, rows, parent=None) -> None:
    if parent is None:
        tree.clear()
    for r in rows:
        node = QtWidgets.QTreeWidgetItem([r.name, r.summary])
        node.setForeground(0, QtGui.QColor(style.STATUS_COLORS[r.status]))
        node.setToolTip(1, r.summary)
        if r.equation:
            child = QtWidgets.QTreeWidgetItem(["", r.equation])
            child.setToolTip(1, r.equation)
            node.addChild(child)
        if r.remedy and r.status in (Status.WARN, Status.FAIL):
            remedy = QtWidgets.QTreeWidgetItem(["", f"remedy: {r.remedy}"])
            remedy.setForeground(1, QtGui.QColor(style.STATUS_COLORS[Status.WARN]))
            node.addChild(remedy)
        if parent is None:
            tree.addTopLevelItem(node)
        else:
            parent.addChild(node)
        node.setExpanded(r.status in (Status.WARN, Status.FAIL))


class CompartmentPanel(QtWidgets.QWidget):
    """Where each channel really is, where it appears to be, and the coefficients per region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["compartment", "ch1 labels", "ch1 signal", "ch2 labels", "ch2 signal",
                                    "r", "M1", "M2"])
        self.table.setRootIsDecorated(False)
        self.table.setColumnWidth(0, 120)
        for i in range(1, 8):
            self.table.setColumnWidth(i, 66)
        layout.addWidget(self.table, 1)
        self.share = PlotCanvas()
        layout.addWidget(self.share, 1)

    def show_simulation(self, sim: Simulation) -> None:
        self.table.clear()
        if not sim.compartments:
            self.message.setText("Compartments exist in the model cell. Choose 'Model cell' (or a 3x preset).")
            self.share.set_spec(PlotSpec(title=""))
            return
        self.message.setText(
            "labels: each channel's true 3D share. signal: its share of the cell's recorded signal in that "
            "region of the focal plane (rim, nucleus outline, the rest). Amber: signal where there are no labels."
        )
        for c in sim.compartments:
            co = c.coeffs
            item = QtWidgets.QTreeWidgetItem([
                c.name, f"{c.true_1:.0%}", f"{c.seen_1:.0%}", f"{c.true_2:.0%}", f"{c.seen_2:.0%}",
                _fmt(co.pcc) if co else "--", _fmt(co.m1) if co else "--", _fmt(co.m2) if co else "--",
            ])
            for ch, true, seen in ((2, c.true_1, c.seen_1), (4, c.true_2, c.seen_2)):
                if true < 0.02 and seen > 0.10:
                    item.setForeground(ch, QtGui.QColor(style.STATUS_COLORS[Status.WARN]))
            self.table.addTopLevelItem(item)
        idx = np.arange(len(sim.compartments), dtype=float)
        names = ", ".join(f"{i}: {c.name}" for i, c in enumerate(sim.compartments))
        self.share.set_spec(PlotSpec(
            title="share of each channel, by compartment: labels (bars) vs recorded signal (dots)",
            xlabel=names, ylabel="share", xrange=(-0.6, len(idx) - 0.4), yrange=(0, 1.05),
            series=[
                Series(idx - 0.18, np.array([c.true_1 for c in sim.compartments]), "#36c26e", "ch1 labels",
                       kind="bars", bar_width=0.34),
                Series(idx + 0.18, np.array([c.true_2 for c in sim.compartments]), "#c95bd6", "ch2 labels",
                       kind="bars", bar_width=0.34),
                Series(idx - 0.18, np.array([c.seen_1 for c in sim.compartments]), "#e6edf3", "ch1 signal",
                       kind="points", width=4),
                Series(idx + 0.18, np.array([c.seen_2 for c in sim.compartments]), "#9aa3b0", "ch2 signal",
                       kind="points", width=4),
            ],
        ))


class ReportPanel(QtWidgets.QTreeWidget):
    """Every derived number, with its equation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["quantity", "value"])
        self.setColumnWidth(0, 210)

    def show_report(self, sections: dict[str, Report]) -> None:
        self.clear()
        for name, report in sections.items():
            head = QtWidgets.QTreeWidgetItem([name, ""])
            font = head.font(0)
            font.setBold(True)
            head.setFont(0, font)
            self.addTopLevelItem(head)
            _fill(self, report.rows, head)
            head.setExpanded(True)


# --- the widget ------------------------------------------------------------------


class ColocWidget(QtWidgets.QWidget):
    """Embeddable: levers on the left, images over the graphs, coefficients on the right."""

    simulated = QtCore.Signal(object)  # the Simulation, after every recompute
    summaryChanged = QtCore.Signal(str)

    def __init__(self, levers: Levers | None = None, parent=None):
        super().__init__(parent)
        self.simulator = Simulator()
        self.sim: Simulation | None = None
        self.sweep = None
        self.noise_seed = 0
        self.merge_name = "green / magenta"
        self.show_roi = True

        self.levers_panel = LeverPanel(levers)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.levers_panel)
        scroll.setMinimumWidth(340)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.images = ImageGrid()
        self.graphs = Graphs()
        self.coefficients = CoefficientPanel()
        self.compartments = CompartmentPanel()
        self.report_panel = ReportPanel()
        self.problem = QtWidgets.QLabel("")
        self.problem.setStyleSheet(style.BANNER)
        self.problem.setWordWrap(True)
        self.problem.setVisible(False)

        centre = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.problem)
        top_layout.addWidget(self.images, 1)
        centre.addWidget(top)
        centre.addWidget(self.graphs)
        centre.setSizes([380, 420])

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.coefficients, "coefficients")
        self.tabs.addTab(self.compartments, "compartments")
        self.tabs.addTab(self.report_panel, "the numbers")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(centre)
        splitter.addWidget(self.tabs)
        splitter.setSizes([350, 820, 520])
        splitter.setStretchFactor(1, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self.refresh)
        self.live_timer = QtCore.QTimer(self)
        self.live_timer.setInterval(300)
        self.live_timer.timeout.connect(self.new_noise)

        self.levers_panel.leversChanged.connect(lambda _lv: self._debounce.start())
        self.graphs.axisChanged.connect(self._debounce.start)
        self.refresh()

    # --- state --------------------------------------------------------------

    @property
    def levers(self) -> Levers:
        return self.levers_panel.levers.with_(noise_seed=self.noise_seed)

    def set_levers(self, levers: Levers) -> None:
        self.levers_panel.set_levers(levers)
        self.refresh()

    def load_preset(self, name: str) -> None:
        self.set_levers(PRESETS[name].levers)

    def new_noise(self) -> None:
        """Record another frame of the same scene: new noise, same everything else."""
        self.noise_seed += 1
        self.refresh()

    def set_live(self, on: bool) -> None:
        if on:
            self.live_timer.start()
        else:
            self.live_timer.stop()

    def set_display(self, merge_name: str | None = None, show_roi: bool | None = None) -> None:
        if merge_name is not None:
            self.merge_name = merge_name
        if show_roi is not None:
            self.show_roi = show_roi
        if self.sim is not None:
            self._draw_images(self.sim)

    # --- the loop -----------------------------------------------------------

    def refresh(self) -> None:
        levers = self.levers
        try:
            sim = self.simulator.run(levers)
            sweep = self.simulator.sweep(levers, self.graphs.axis_key())
        except ValueError as exc:
            self.problem.setText(f"cannot image this: {exc}")
            self.problem.setVisible(True)
            return
        self.problem.setVisible(False)
        self.sim, self.sweep = sim, sweep
        self._draw_images(sim)
        self.graphs.show_simulation(sim, sweep)
        sections = build_report(sim)
        self.coefficients.show_simulation(sim, sections["colocalization"])
        self.compartments.show_simulation(sim)
        self.report_panel.show_report(sections)
        self.simulated.emit(sim)
        t, me = sim.truth, sim.measured
        bits = [
            f"FWHM {sim.fwhm_um[0] * 1000:.0f} / {sim.fwhm_um[1] * 1000:.0f} nm",
            f"pixel {sim.pixel_um * 1000:.0f} nm",
        ]
        if sim.scene.cell is not None:
            share1, share2 = t.compartment_overlap()
            bits += [f"r {_fmt(me.pcc)}", f"M1 {_fmt(me.m1)} M2 {_fmt(me.m2)} "
                     f"(shared compartments {share1:.0%} / {share2:.0%})"]
        else:
            bits += [f"r {_fmt(me.pcc)} (molecules {_fmt(t.pcc)})",
                     f"M1 {_fmt(me.m1)} M2 {_fmt(me.m2)} (molecules {_fmt(t.m1)} / {_fmt(t.m2)})"]
        bits += sim.notes
        problems = [r.name for rep in sections.values() for r in rep.rows if r.status is Status.WARN]
        if problems:
            bits.append("check: " + ", ".join(dict.fromkeys(problems)))
        self.summaryChanged.emit("   |   ".join(bits))

    def _draw_images(self, sim: Simulation) -> None:
        field = sim.extent_um
        outline = outline_of(sim.roi_mask) if self.show_roi and sim.roi_name != "whole field" else None

        # The truth: labels in the focal plane, block-summed to <= 256 px for display.
        o1, o2 = sim.object_slice
        k = max(o1.shape[0] // 256, 1)
        if k > 1:
            m = o1.shape[0] // k
            o1 = o1[: m * k, : m * k].reshape(m, k, m, k).sum(axis=(1, 3))
            o2 = o2[: m * k, : m * k].reshape(m, k, m, k).sum(axis=(1, 3))

        def hi(img):
            nz = img[img > 0]
            return float(np.percentile(nz, 95)) if nz.size else 1.0

        sp = sim.scene
        plane = "" if sp.flat else f" at z = {sim.levers.focus:.1f} µm"
        self.images.object.show_image(
            merge(scale(o1, 0, hi(o1)), scale(o2, 0, hi(o2)), self.merge_name), sp.field_um,
            f"{sp.dx_um * k * 1000:.0f} nm bins{plane} · white = both",
        )
        # Optical and recorded share each channel's display range, so noise looks as big as it is.
        a_e, b_e = sim.optical
        ra, rb = float(np.percentile(a_e, 99.5)) or 1.0, float(np.percentile(b_e, 99.5)) or 1.0
        bg = sim.levers.background
        self.images.optical.show_image(
            merge(scale(a_e, 0, ra), scale(b_e, 0, rb), self.merge_name, outline), field,
            f"FWHM {sim.fwhm_um[0] * 1000:.0f} / {sim.fwhm_um[1] * 1000:.0f} nm · {sim.pixel_um * 1000:.0f} nm px",
        )
        r1, r2 = sim.recorded
        self.images.recorded.show_image(
            merge(scale(r1, bg, bg + ra), scale(r2, bg, bg + rb), self.merge_name, outline), field,
            f"peak {sim.levers.peak_1:.0f} / {sim.levers.peak_2:.0f} photons" + (" · noise" if sim.levers.noise else ""),
        )
        a, b = sim.analysed
        m = sim.roi_mask
        prod = np.zeros_like(a)
        if m.sum() > 2:
            prod[m] = nmdp(a[m], b[m])
        lim = float(np.percentile(np.abs(prod[m]), 99.5)) if m.any() else 1.0
        self.images.product.show_image(
            diverging(prod, lim or 1.0, outline), field,
            "orange: both above / below mean · blue: opposite",
        )


# --- the window ------------------------------------------------------------------


class ColocWindow(QtWidgets.QMainWindow):
    def __init__(self, levers: Levers | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ColocWidget -- what resolution does to colocalization")
        self.widget = ColocWidget(levers)
        self.widget.summaryChanged.connect(self.statusBar().showMessage)
        self.setCentralWidget(self.widget)
        self._build_toolbar()
        self._warn_about_model()
        self.widget.refresh()  # again, now that the status bar is listening
        self.resize(1680, 980)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("main")
        bar.setMovable(False)

        new = QtGui.QAction("New frame", self)
        new.setShortcut("Ctrl+R")
        new.setToolTip("record the same scene again: same specimen, same optics, new noise")
        new.triggered.connect(lambda _checked=False: self.widget.new_noise())
        bar.addAction(new)

        self.live = QtGui.QAction("Live", self, checkable=True, checked=False)
        self.live.setShortcut("Space")
        self.live.setToolTip("stream frames: the coefficients wobble by what noise alone does")
        self.live.toggled.connect(self.widget.set_live)
        bar.addAction(self.live)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel("  preset: "))
        self.presets = QtWidgets.QComboBox()
        self.presets.addItem("--")
        for name in PRESETS:
            self.presets.addItem(name)
        self.presets.setToolTip("starting points that each demonstrate one thing")
        self.presets.currentTextChanged.connect(self._preset_chosen)
        bar.addWidget(self.presets)

        reset = QtGui.QAction("Reset", self)
        reset.triggered.connect(lambda _checked=False: self.widget.set_levers(Levers()))
        bar.addAction(reset)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel("  colours: "))
        self.merge = QtWidgets.QComboBox()
        self.merge.addItems(list(style.MERGES))
        self.merge.currentTextChanged.connect(lambda t: self.widget.set_display(merge_name=t))
        bar.addWidget(self.merge)
        roi = QtGui.QAction("Show region", self, checkable=True, checked=True)
        roi.setToolTip("outline the region the coefficients are computed in")
        roi.toggled.connect(lambda on: self.widget.set_display(show_roi=on))
        bar.addAction(roi)

        bar.addSeparator()
        help_action = QtGui.QAction("How this works", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(lambda _checked=False: self.show_help())
        bar.addAction(help_action)

    def _preset_chosen(self, name: str) -> None:
        if name in PRESETS:
            self.widget.load_preset(name)
            self.statusBar().showMessage(PRESETS[name].description, 10000)

    def _warn_about_model(self) -> None:
        banner = QtWidgets.QLabel(
            "Scalar, aberration-free optics; geometric defocus; a Lorentzian pinhole section. "
            "Built to show trends, not to calibrate an instrument."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(style.BANNER)
        self.statusBar().addPermanentWidget(banner, 0)

    def show_help(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("How the colocalization widget works")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText("<b>Colocalization is a statement about distance. The image can only see distances "
                    "larger than its resolution.</b>")
        box.setInformativeText(
            "<p><b>Images</b>, left to right: the labels themselves in the focal plane (the truth); the image "
            "the optics and pixels form, with no noise; what is recorded; and a map of where Pearson's r comes "
            "from (orange: both channels above or both below their means).</p>"
            "<p><b>Three stages</b>: every coefficient is computed from the molecules, from the noise-free image "
            "and from the recorded image. Resolution is the gap between the first two; noise the gap between "
            "the last two.</p>"
            "<p><b>The sweep</b> plots each coefficient against the PSF width with everything else fixed. "
            "Dashed lines are the molecular truth.</p>"
            "<p><b>Correlation vs co-occurrence</b>: Pearson's r asks whether the intensities are proportional; "
            "Manders' M1/M2 ask what fraction of one channel lies where the other is. They answer different "
            "questions, and both describe pixels about one PSF wide.</p>"
            "<p><b>Cases</b>: 1x presets show unresolved collections that look co-labelled; 2x presets show "
            "distribution and morphology effects; 3x presets put labels into a model cell's nucleus, cytosol "
            "and plasma membrane -- see the compartments tab.</p>"
            "<p><b>New frame</b> (Ctrl+R) re-records the scene; <b>Live</b> (Space) streams frames.</p>"
        )
        box.exec()

    def closeEvent(self, event) -> None:
        self.widget.set_live(False)
        super().closeEvent(event)


def launch(levers: Levers | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = ColocWindow(levers)
    window.show()
    return app.exec()
