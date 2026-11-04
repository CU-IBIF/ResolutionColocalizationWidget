"""The levers: the specimen, the microscope, the detector, and the analysis.

Grouped the way the decisions are made: what is there (the specimen), how it is
imaged (resolution and sectioning), how it is recorded (pixels, photons, crosstalk),
and how it is measured (region, threshold, the distance that counts as "together").

Each scene declares its own parameters, so its group is rebuilt from that
declaration when the scene changes. The panel owns a :class:`Levers` value and
emits a new one on every change; it never computes anything itself.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from ..physics import optics
from ..physics.imaging import ROIS, Levers
from ..physics.metrics import MANDERS_FORMS, THRESHOLD_MODES
from ..physics.scenes import SCENES
from . import style


class LeverSlider(QtWidgets.QWidget):
    """A slider and a spin box bound together, optionally on a log scale.

    The slider is for feel -- dragging exposure across three decades and watching
    the image change is the point -- and the spin box is for typing an exact value.
    """

    valueChanged = QtCore.Signal(float)
    STEPS = 1000

    def __init__(self, lo: float, hi: float, value: float, log: bool = False, decimals: int = 2,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.log = lo, hi, log
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(92)
        if log:
            self.spin.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        else:
            self.spin.setSingleStep((hi - lo) / 100)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _to_pos(self, v: float) -> int:
        v = min(max(v, self.lo), self.hi)
        if self.log:
            f = (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        else:
            f = (v - self.lo) / (self.hi - self.lo)
        return int(round(f * self.STEPS))

    def _from_pos(self, pos: int) -> float:
        f = pos / self.STEPS
        if self.log:
            return 10 ** (math.log10(self.lo) + f * (math.log10(self.hi) - math.log10(self.lo)))
        return self.lo + f * (self.hi - self.lo)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, v: float) -> None:
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(self._to_pos(v))
        for w in (self.slider, self.spin):
            w.blockSignals(False)

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = lo, hi
        v = min(max(self.value(), lo), hi)
        self.spin.blockSignals(True)
        self.spin.setRange(lo, hi)
        self.spin.blockSignals(False)
        self.set_value(v)

    def _from_slider(self, pos: int) -> None:
        v = self._from_pos(pos)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


def _combo(items, current) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox()
    box.addItems(list(items))
    box.setCurrentText(current)
    box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(12)
    return box


def _muted(text: str = "") -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {style.MUTED}; font-size: 11px;")
    return label


class LeverPanel(QtWidgets.QWidget):
    leversChanged = QtCore.Signal(object)

    def __init__(self, levers: Levers | None = None, parent=None):
        super().__init__(parent)
        self.levers = levers or Levers()
        self._updating = False
        self.param_sliders: dict[str, LeverSlider] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # --- specimen -------------------------------------------------------
        box, form = self._group("specimen (the truth)")
        self.scene = _combo(SCENES, self.levers.scene)
        self.scene.setToolTip("each scene isolates one way the image and the molecules can disagree")
        form.addRow(self.scene)
        self.scene_info = _muted()
        form.addRow(self.scene_info)
        self.params_box = QtWidgets.QWidget()
        self.params_form = QtWidgets.QFormLayout(self.params_box)
        self.params_form.setContentsMargins(0, 0, 0, 0)
        self.params_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(self.params_box)
        self.reseed = QtWidgets.QPushButton("new specimen")
        self.reseed.setToolTip("another field of the same specimen: new positions, same rules")
        form.addRow(self.reseed)
        outer.addWidget(box)

        # --- optics -----------------------------------------------------------
        box, form = self._group("resolution and sectioning")
        self.modality = _combo(optics.MODALITIES, self.levers.modality)
        form.addRow("microscope", self.modality)
        self.modality_info = _muted()
        form.addRow(self.modality_info)
        self.objective = QtWidgets.QComboBox()
        self.objective.addItem("custom")
        for o in optics.OBJECTIVES:
            self.objective.addItem(o.label)
        form.addRow("objective", self.objective)
        self.na = LeverSlider(0.1, 1.49, self.levers.na, decimals=2)
        self.na.setToolTip("NA = n sin(theta): FWHM = 0.514 lambda / NA")
        form.addRow("NA", self.na)
        self.medium = _combo(optics.IMMERSION_MEDIA, self.levers.medium)
        form.addRow("immersion", self.medium)
        self.lambda1 = LeverSlider(400, 800, self.levers.emission_nm_1, suffix=" nm")
        form.addRow("ch1 emission", self.lambda1)
        self.lambda2 = LeverSlider(400, 800, self.levers.emission_nm_2, suffix=" nm")
        form.addRow("ch2 emission", self.lambda2)
        self.sted = LeverSlider(20, 250, self.levers.sted_fwhm_nm, log=True, suffix=" nm")
        self.sted.setToolTip("STED only: the spot width the depletion power gives")
        form.addRow("STED FWHM", self.sted)
        self.focus = LeverSlider(-1.5, 1.5, self.levers.focus, decimals=2, suffix=" µm")
        self.focus.setToolTip("height of the focal plane above the coverslip")
        form.addRow("focal plane", self.focus)
        self.resolution_info = _muted()
        form.addRow(self.resolution_info)
        outer.addWidget(box)

        # --- registration -----------------------------------------------------
        box, form = self._group("channel registration (ch2 vs ch1)")
        self.shift_x = LeverSlider(-500, 500, self.levers.shift_x_nm, suffix=" nm")
        self.shift_x.setToolTip("lateral chromatic aberration or camera misalignment")
        form.addRow("shift x", self.shift_x)
        self.shift_y = LeverSlider(-500, 500, self.levers.shift_y_nm, suffix=" nm")
        form.addRow("shift y", self.shift_y)
        self.shift_z = LeverSlider(-1500, 1500, self.levers.shift_z_nm, suffix=" nm")
        self.shift_z.setToolTip("axial chromatic aberration: ch2 focuses at a different depth")
        form.addRow("shift z", self.shift_z)
        outer.addWidget(box)

        # --- detection --------------------------------------------------------
        box, form = self._group("detection")
        self.pixel = LeverSlider(20, 400, self.levers.pixel_nm, log=True, suffix=" nm")
        self.pixel.setToolTip("pixel size at the sample")
        form.addRow("pixel", self.pixel)
        self.peak1 = LeverSlider(2, 5000, self.levers.peak_1, log=True, suffix=" ph")
        self.peak1.setToolTip("photons in the brightest pixels of channel 1's noise-free image")
        form.addRow("ch1 peak", self.peak1)
        self.peak2 = LeverSlider(2, 5000, self.levers.peak_2, log=True, suffix=" ph")
        form.addRow("ch2 peak", self.peak2)
        self.background = LeverSlider(0, 500, self.levers.background, suffix=" ph")
        self.background.setToolTip("uniform background per pixel, each channel (autofluorescence, haze)")
        form.addRow("background", self.background)
        self.read = LeverSlider(0, 10, self.levers.read_noise_e, decimals=1, suffix=" e-")
        form.addRow("read noise", self.read)
        self.bleed = LeverSlider(0, 50, self.levers.bleed_through * 100, decimals=1, suffix=" %")
        self.bleed.setToolTip("fraction of channel 1's photons that reach channel 2's detector")
        form.addRow("bleed-through", self.bleed)
        self.noise = QtWidgets.QCheckBox("photon and read noise")
        self.noise.setChecked(self.levers.noise)
        form.addRow(self.noise)
        outer.addWidget(box)

        # --- analysis ---------------------------------------------------------
        box, form = self._group("analysis")
        self.roi = _combo(ROIS, self.levers.roi)
        self.roi.setToolTip("where the coefficients are computed; compartments exist only in the model cell")
        form.addRow("region", self.roi)
        self.threshold = _combo(THRESHOLD_MODES, self.levers.threshold)
        self.threshold.setToolTip("which pixels count as 'present' for Manders' M1 and M2")
        form.addRow("threshold", self.threshold)
        self.fraction = LeverSlider(0, 100, self.levers.threshold_fraction * 100, suffix=" %")
        form.addRow("fixed at", self.fraction)
        self.manders_form = _combo(MANDERS_FORMS, self.levers.manders_form)
        self.manders_form.setToolTip(
            "the denominator of M1 (and M2):\n"
            "above own threshold (Coloc 2): sum of a over pixels where a > T1\n"
            "all signal (Manders 1993): sum of a over every pixel in the region"
        )
        form.addRow("Manders' over", self.manders_form)
        self.subtract = QtWidgets.QCheckBox("subtract background")
        self.subtract.setChecked(self.levers.subtract_background)
        form.addRow(self.subtract)
        self.truth = LeverSlider(5, 500, self.levers.truth_nm, log=True, suffix=" nm")
        self.truth.setToolTip("the molecular distance that counts as 'together' in the truth")
        form.addRow("'together' within", self.truth)
        self.realizations = QtWidgets.QSpinBox()
        self.realizations.setRange(0, 200)
        self.realizations.setValue(self.levers.realizations)
        self.realizations.setToolTip("repeat frames of the same scene, for the spread noise causes")
        form.addRow("repeat frames", self.realizations)
        outer.addWidget(box)
        outer.addStretch(1)

        self._build_params()
        self._connect()
        self._sync_dependent()

    def _group(self, title: str):
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(6, 6, 6, 6)
        return box, form

    def _build_params(self) -> None:
        """One slider per parameter the current scene declares."""
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.param_sliders = {}
        values = self.levers.params
        for p in self.levers.scene_model.params:
            s = LeverSlider(max(p.lo, 1e-3) if p.log else p.lo, p.hi, values[p.name], log=p.log,
                            decimals=p.decimals, suffix=p.unit)
            if p.help:
                s.setToolTip(p.help)
            s.valueChanged.connect(lambda v, name=p.name: self._set(params={name: v}))
            self.param_sliders[p.name] = s
            label = QtWidgets.QLabel(p.label)
            if p.help:
                label.setToolTip(p.help)
            self.params_form.addRow(label, s)
        # Rows were removed and added: settle the geometry now, not on the next event.
        self.params_form.activate()
        self.layout().activate() if self.layout() else None

    def _connect(self) -> None:
        self.scene.currentTextChanged.connect(self._scene_changed)
        self.reseed.clicked.connect(lambda _checked=False: self._set(scene_seed=self.levers.scene_seed + 1))
        self.modality.currentTextChanged.connect(lambda t: self._set(modality=t))
        self.objective.currentIndexChanged.connect(self._objective_chosen)
        self.na.valueChanged.connect(lambda v: self._set(na=v, _custom=True))
        self.medium.currentTextChanged.connect(lambda t: self._set(medium=t, _custom=True))
        self.lambda1.valueChanged.connect(lambda v: self._set(emission_nm_1=v))
        self.lambda2.valueChanged.connect(lambda v: self._set(emission_nm_2=v))
        self.sted.valueChanged.connect(lambda v: self._set(sted_fwhm_nm=v))
        self.focus.valueChanged.connect(lambda v: self._set(focus_um=v))
        self.shift_x.valueChanged.connect(lambda v: self._set(shift_x_nm=v))
        self.shift_y.valueChanged.connect(lambda v: self._set(shift_y_nm=v))
        self.shift_z.valueChanged.connect(lambda v: self._set(shift_z_nm=v))
        self.pixel.valueChanged.connect(lambda v: self._set(pixel_nm=v))
        self.peak1.valueChanged.connect(lambda v: self._set(peak_1=v))
        self.peak2.valueChanged.connect(lambda v: self._set(peak_2=v))
        self.background.valueChanged.connect(lambda v: self._set(background=v))
        self.read.valueChanged.connect(lambda v: self._set(read_noise_e=v))
        self.bleed.valueChanged.connect(lambda v: self._set(bleed_through=v / 100.0))
        self.noise.toggled.connect(lambda on: self._set(noise=on))
        self.roi.currentTextChanged.connect(lambda t: self._set(roi=t))
        self.threshold.currentTextChanged.connect(lambda t: self._set(threshold=t))
        self.fraction.valueChanged.connect(lambda v: self._set(threshold_fraction=v / 100.0))
        self.manders_form.currentTextChanged.connect(lambda t: self._set(manders_form=t))
        self.subtract.toggled.connect(lambda on: self._set(subtract_background=on))
        self.truth.valueChanged.connect(lambda v: self._set(truth_nm=v))
        self.realizations.valueChanged.connect(lambda v: self._set(realizations=int(v)))

    # --- changes ------------------------------------------------------------

    def _set(self, _custom: bool = False, **changes) -> None:
        if self._updating:
            return
        self.levers = self.levers.with_(**changes)
        if _custom and self.objective.currentIndex() != 0:
            self.objective.blockSignals(True)
            self.objective.setCurrentIndex(0)
            self.objective.blockSignals(False)
        self._sync_dependent()
        self.leversChanged.emit(self.levers)

    def _scene_changed(self, name: str) -> None:
        if self._updating:
            return
        # A new scene starts from its own parameters and its own focal plane.
        self.levers = self.levers.with_(scene=name, scene_params=(), focus_um=None)
        self._build_params()
        self._set()

    def _objective_chosen(self, index: int) -> None:
        if index == 0:
            return
        o = optics.OBJECTIVES[index - 1]
        self._updating = True
        self.medium.setCurrentText(o.medium)
        self.na.set_value(o.na)
        self._updating = False
        self._set(na=o.na, medium=o.medium)

    def _sync_dependent(self) -> None:
        """Keep the controls that depend on other controls honest."""
        lv = self.levers
        scene = lv.scene_model
        self.scene_info.setText(scene.description)
        m = optics.MODALITIES[lv.modality]
        self.modality_info.setText(m.note)
        self.sted.setEnabled(m.fixed_fwhm)
        self.fraction.setEnabled(lv.threshold == "fixed fraction")
        n = lv.n_medium
        self._updating = True
        if lv.na > n:
            # NA = n sin(theta) cannot exceed n; pull the NA back rather than refuse.
            self.levers = lv = lv.with_(na=round(n * 0.99, 2))
            self.na.set_value(lv.na)
        lo, hi = scene.focus_range
        if scene.kind == "cell":
            hi = lv.params["cell_height_um"]
        self.focus.set_range(lo, hi)
        self.focus.set_value(lv.focus)
        self._updating = False
        f1, f2 = lv.fwhm_um(1) * 1000, lv.fwhm_um(2) * 1000
        nyq = optics.nyquist_pixel_um(min(f1, f2) / 1000) * 1000
        section = "none (all planes add)" if not m.sectioning else f"{lv.section_um(1):.2f} µm"
        self.resolution_info.setText(
            f"FWHM {f1:.0f} / {f2:.0f} nm (ch1 / ch2) · Nyquist pixel {nyq:.0f} nm · optical section {section}"
        )

    def set_levers(self, levers: Levers) -> None:
        """Load a whole set of levers (a preset) into the controls."""
        self._updating = True
        self.levers = levers
        self.scene.setCurrentText(levers.scene)
        self._build_params()
        self.modality.setCurrentText(levers.modality)
        self.objective.setCurrentIndex(0)
        self.na.set_value(levers.na)
        self.medium.setCurrentText(levers.medium)
        self.lambda1.set_value(levers.emission_nm_1)
        self.lambda2.set_value(levers.emission_nm_2)
        self.sted.set_value(levers.sted_fwhm_nm)
        self.shift_x.set_value(levers.shift_x_nm)
        self.shift_y.set_value(levers.shift_y_nm)
        self.shift_z.set_value(levers.shift_z_nm)
        self.pixel.set_value(levers.pixel_nm)
        self.peak1.set_value(levers.peak_1)
        self.peak2.set_value(levers.peak_2)
        self.background.set_value(levers.background)
        self.read.set_value(levers.read_noise_e)
        self.bleed.set_value(levers.bleed_through * 100)
        self.noise.setChecked(levers.noise)
        self.roi.setCurrentText(levers.roi)
        self.threshold.setCurrentText(levers.threshold)
        self.fraction.set_value(levers.threshold_fraction * 100)
        self.manders_form.setCurrentText(levers.manders_form)
        self.subtract.setChecked(levers.subtract_background)
        self.truth.set_value(levers.truth_nm)
        self.realizations.setValue(levers.realizations)
        self._updating = False
        self._sync_dependent()
        self.leversChanged.emit(self.levers)
