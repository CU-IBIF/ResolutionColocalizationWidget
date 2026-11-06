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


def manders_map(value: np.ndarray, own: np.ndarray, partner: np.ndarray, colour: tuple[float, float, float],
                outline: np.ndarray | None = None) -> QtGui.QImage:
    """Which pixels Manders' coefficient counts, for one channel.

    Bright (the channel's colour, scaled by intensity): above its own threshold and
    the partner is present -- the numerator. Dim slate: above its own threshold
    only -- denominator only. Black: not counted.
    """
    hi = float(np.percentile(value[own], 99.5)) if own.any() else 1.0
    t = np.clip(value / (hi or 1.0), 0, 1)[..., None]
    lit = 0.25 + 0.75 * t
    rgb = np.zeros(value.shape + (3,))
    num = own & partner
    rgb[num] = (lit * np.array(colour))[num]
    rgb[own & ~partner] = (lit * np.array([0.30, 0.36, 0.48]))[own & ~partner]
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
        self.spots: list[tuple[float, float, str, bool]] = []  # x um, y um, colour, matched
        self.setMinimumSize(170, 190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def show_image(self, qimage: QtGui.QImage, field_um: float, caption: str = "",
                   spots: list[tuple[float, float, str, bool]] | None = None) -> None:
        self.image = qimage
        self.field_um = field_um
        self.caption = caption
        self.spots = spots or []
        self.update()

    def image_rect(self) -> QtCore.QRectF:
        top, bottom = 18, 32  # room for a caption of two lines
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
        if self.spots:
            # Rings at each detected object; a filled centre when it has a partner within d.
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            for x, y, colour, matched in self.spots:
                c = QtCore.QPointF(rect.left() + x * px_per_um, rect.top() + y * px_per_um)
                p.setPen(QtGui.QPen(QtGui.QColor(colour), 1.4))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(c, 4.5, 4.5)
                if matched:
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(QtGui.QColor("#ffffff"))
                    p.drawEllipse(c, 1.8, 1.8)
            p.setBrush(QtCore.Qt.NoBrush)
            p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        bar_um = next((b for b in (0.5, 1.0, 2.0, 5.0) if b * px_per_um >= 30), 5.0)
        x1 = rect.right() - 8
        x0 = x1 - bar_um * px_per_um
        y = rect.bottom() - 8
        p.fillRect(QtCore.QRectF(x0 - 1, y - 2, x1 - x0 + 2, 6), QtGui.QColor(0, 0, 0, 160))
        p.fillRect(QtCore.QRectF(x0, y - 1, x1 - x0, 4), QtGui.QColor("#ffffff"))
        p.setPen(QtGui.QColor("#ffffff"))
        p.drawText(QtCore.QRectF(x0 - 40, y - 18, x1 - x0 + 40, 14), QtCore.Qt.AlignRight, f"{bar_um:g} µm")
        p.setPen(QtGui.QColor(style.MUTED))
        p.drawText(QtCore.QRectF(4, rect.bottom() + 1, self.width() - 8, 30),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap, self.caption)
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


class _WrapDelegate(QtWidgets.QStyledItemDelegate):
    """Measures each cell's text wrapped to its column's real width.

    A plain tree view asks for a cell's size without telling the delegate how wide
    the column is, so long text gets one line and an ellipsis. This delegate asks
    the view for the width, and paints with word wrap on.
    """

    PAD = 6

    def __init__(self, view: QtWidgets.QTreeView):
        super().__init__(view)
        self.view = view

    def _text_width(self, index: QtCore.QModelIndex) -> int:
        width = self.view.columnWidth(index.column())
        if index.column() == 0:
            depth, parent = 0, index.parent()
            while parent.isValid():
                depth += 1
                parent = parent.parent()
            width -= self.view.indentation() * (depth + (1 if self.view.rootIsDecorated() else 0))
        return max(width - self.PAD * 2, 30)

    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        option.features |= QtWidgets.QStyleOptionViewItem.WrapText
        option.textElideMode = QtCore.Qt.ElideNone

    def sizeHint(self, option, index) -> QtCore.QSize:
        text = index.data(QtCore.Qt.DisplayRole)
        if not text:
            return super().sizeHint(option, index)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        width = self._text_width(index)
        rect = QtGui.QFontMetrics(opt.font).boundingRect(
            QtCore.QRect(0, 0, width, 100_000), QtCore.Qt.TextWordWrap, str(text))
        return QtCore.QSize(width, rect.height() + 4)


class WrappingTree(QtWidgets.QTreeWidget):
    """A tree whose text wraps to the pane instead of scrolling sideways.

    Rows are re-measured whenever the pane or a column changes width, so an
    equation reflows like a paragraph.
    """

    def __init__(self, labels: list[str], first_width: int = 170, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(labels)
        self.setWordWrap(True)
        self.setUniformRowHeights(False)
        self.setTextElideMode(QtCore.Qt.ElideNone)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setIndentation(12)
        self.setItemDelegate(_WrapDelegate(self))
        header = self.header()
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(40)
        self.setColumnWidth(0, first_width)
        header.sectionResized.connect(lambda *_: self.scheduleDelayedItemsLayout())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.scheduleDelayedItemsLayout()
