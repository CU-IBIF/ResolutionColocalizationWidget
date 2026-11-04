"""Image panes: two-colour merges, the correlation map and the cytofluorogram.

Pixels are drawn as blocks (nearest-neighbour), never smoothed: pixel size is a
lever, and interpolation would hide what it does.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import style


def _rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    buf = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    h, w = buf.shape[:2]
    return QtGui.QImage(buf.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


def scale(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo
    return np.clip((img - lo) / span, 0.0, 1.0) if span > 0 else np.zeros_like(img, dtype=float)


def merge(a: np.ndarray, b: np.ndarray, merge_name: str = "green / magenta",
          outline: np.ndarray | None = None) -> QtGui.QImage:
    """Additive two-colour merge of two images already scaled to [0, 1]."""
    ca, cb = style.MERGES[merge_name]
    rgb = a[..., None] * np.array(ca) + b[..., None] * np.array(cb)
    if outline is not None and outline.any():
        rgb[outline] = (0.85, 0.64, 0.25)
    return _rgb_to_qimage(rgb)


def diverging(values: np.ndarray, limit: float, outline: np.ndarray | None = None) -> QtGui.QImage:
    """Signed map: blue negative, orange positive, dark grey zero."""
    t = np.clip(values / limit, -1, 1)[..., None] if limit > 0 else np.zeros(values.shape + (1,))
    neg = np.array([0x39, 0x87, 0xE5]) / 255.0
    pos = np.array([0xD9, 0x59, 0x26]) / 255.0
    mid = np.array([0.16, 0.17, 0.19])
    rgb = np.where(t < 0, mid + (neg - mid) * (-t), mid + (pos - mid) * t)
    if outline is not None and outline.any():
        rgb[outline] = (0.85, 0.64, 0.25)
    return _rgb_to_qimage(rgb)


def outline_of(mask: np.ndarray) -> np.ndarray:
    """Pixels of the mask that touch a pixel outside it."""
    if mask.all():
        return np.zeros_like(mask)
    p = np.pad(mask, 1, constant_values=False)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


class ImagePane(QtWidgets.QWidget):
    """One image, a title, a caption, a scale bar."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.caption = ""
        self.image: QtGui.QImage | None = None
        self.field_um = 1.0
        self.message = "--"
        self.setMinimumSize(170, 190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def show_image(self, qimage: QtGui.QImage, field_um: float, caption: str = "") -> None:
        self.image = qimage
        self.field_um = field_um
        self.caption = caption
        self.update()

    def image_rect(self) -> QtCore.QRectF:
        top, bottom = 18, 18
        side = max(min(self.width() - 4, self.height() - top - bottom), 10)
        return QtCore.QRectF((self.width() - side) / 2, top, side, side)

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(style.BACKGROUND))
        font = p.font()
        font.setPointSizeF(9)
        bold = QtGui.QFont(font)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QtGui.QColor(style.TEXT))
        p.drawText(QtCore.QRectF(4, 0, self.width() - 8, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.title)
        p.setFont(font)
        rect = self.image_rect()
        if self.image is None:
            p.setPen(QtGui.QColor(style.AXIS))
            p.drawText(rect, QtCore.Qt.AlignCenter, self.message)
            p.end()
            return
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        p.drawImage(rect, self.image)
        px_per_um = rect.width() / self.field_um
        bar_um = next((b for b in (0.5, 1.0, 2.0, 5.0) if b * px_per_um >= 30), 5.0)
        x1 = rect.right() - 8
        x0 = x1 - bar_um * px_per_um
        y = rect.bottom() - 8
        p.fillRect(QtCore.QRectF(x0 - 1, y - 2, x1 - x0 + 2, 6), QtGui.QColor(0, 0, 0, 160))
        p.fillRect(QtCore.QRectF(x0, y - 1, x1 - x0, 4), QtGui.QColor("#ffffff"))
        p.setPen(QtGui.QColor("#ffffff"))
        p.drawText(QtCore.QRectF(x0 - 40, y - 18, x1 - x0 + 40, 14), QtCore.Qt.AlignRight, f"{bar_um:g} µm")
        p.setPen(QtGui.QColor(style.MUTED))
        p.drawText(QtCore.QRectF(4, rect.bottom() + 1, self.width() - 8, 16),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.caption)
        p.end()


class ImageGrid(QtWidgets.QWidget):
    """Object -> optics -> recording -> where r comes from."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.object = ImagePane("labels in the focal plane (truth)")
        self.optical = ImagePane("noise-free image")
        self.recorded = ImagePane("recorded image")
        self.product = ImagePane("where r comes from (nMDP)")
        for i, pane in enumerate(self.panes()):
            layout.addWidget(pane, 0, i)
            layout.setColumnStretch(i, 1)

    def panes(self) -> list[ImagePane]:
        return [self.object, self.optical, self.recorded, self.product]


class Cytofluorogram(QtWidgets.QWidget):
    """The 2D histogram of (ch1, ch2) pixel pairs, with the Costes fit and thresholds.

    Colocalized signal lies along a line through the origin; independent signals
    fill the two axes; exclusion hugs them. Log-scaled counts, so the sparse bright
    pixels that carry the colocalization are visible next to the dark majority.
    """

    BINS = 96

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 180)
        self.image: QtGui.QImage | None = None
        self.amax = self.bmax = 1.0
        self.ta = self.tb = 0.0
        self.slope = self.intercept = float("nan")
        self.title = "cytofluorogram"
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def show_pairs(self, a: np.ndarray, b: np.ndarray, ta: float, tb: float, slope: float, intercept: float,
                   title: str) -> None:
        self.amax = float(np.percentile(a, 99.9)) or 1.0
        self.bmax = float(np.percentile(b, 99.9)) or 1.0
        h, _, _ = np.histogram2d(b, a, bins=self.BINS, range=[[0, self.bmax], [0, self.amax]])
        v = np.log1p(h)
        v = v / v.max() if v.max() > 0 else v
        v = v[::-1]  # ch2 increases upward
        t = v[..., None]
        lo = np.array([0.08, 0.09, 0.11])
        hi = np.array([0.90, 0.93, 0.96])
        rgb = np.where(t > 0, lo + (hi - lo) * t ** 0.7, np.array([0.043, 0.051, 0.063]))
        self.image = _rgb_to_qimage(rgb)
        self.ta, self.tb, self.slope, self.intercept, self.title = ta, tb, slope, intercept, title
        self.update()

    def plot_rect(self) -> QtCore.QRectF:
        left, top, bottom = 44, 18, 30
        side = max(min(self.width() - left - 10, self.height() - top - bottom), 10)
        return QtCore.QRectF(left, top, side, side)

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(style.BACKGROUND))
        font = p.font()
        font.setPointSizeF(8.5)
        bold = QtGui.QFont(font)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QtGui.QColor(style.TEXT))
        p.drawText(QtCore.QPointF(6, 13), self.title)
        p.setFont(font)
        r = self.plot_rect()
        if self.image is not None:
            p.drawImage(r, self.image)
        p.setPen(QtGui.QColor(style.AXIS))
        p.drawRect(r)

        def X(v):
            return r.left() + v / self.amax * r.width()

        def Y(v):
            return r.bottom() - v / self.bmax * r.height()

        p.save()
        p.setClipRect(r)
        pen = QtGui.QPen(QtGui.QColor(style.STATUS_COLORS[style.Status.WARN]), 1.2, QtCore.Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(X(self.ta), r.top()), QtCore.QPointF(X(self.ta), r.bottom()))
        p.drawLine(QtCore.QPointF(r.left(), Y(self.tb)), QtCore.QPointF(r.right(), Y(self.tb)))
        if math.isfinite(self.slope):
            p.setPen(QtGui.QPen(QtGui.QColor(style.COEF_COLORS["pcc"]), 2))
            p.drawLine(QtCore.QPointF(X(0), Y(self.intercept)),
                       QtCore.QPointF(X(self.amax), Y(self.slope * self.amax + self.intercept)))
        p.restore()
        p.setPen(QtGui.QColor(style.MUTED))
        p.drawText(QtCore.QRectF(r.left(), r.bottom() + 2, r.width(), 14), QtCore.Qt.AlignHCenter,
                   f"channel 1 (0 - {self.amax:.3g} photons)")
        p.save()
        p.translate(12, r.center().y())
        p.rotate(-90)
        p.drawText(QtCore.QRectF(-r.height() / 2, -8, r.height(), 14), QtCore.Qt.AlignHCenter,
                   f"channel 2 (0 - {self.bmax:.3g})")
        p.restore()
        p.end()
