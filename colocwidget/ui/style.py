"""Colours shared by every pane, matching NoiseWidget and MicroscopeVuilder."""

from __future__ import annotations

from ..physics.base import Status

BACKGROUND = "#0b0d10"
PANEL = "#14171c"
GRID = "#262b33"
AXIS = "#7a828e"
TEXT = "#c9d1d9"
MUTED = "#9aa3b0"

STATUS_COLORS = {
    Status.PASS: "#4fbf78",
    Status.WARN: "#d9a441",
    Status.FAIL: "#d95f5f",
    Status.INFO: "#7a828e",
}

# One colour per coefficient, the same in the sweep, the tables and the legends.
# NoiseWidget's validated categorical set (CVD-separable on #0b0d10), in its order.
COEF_COLORS = {
    "pcc": "#3987e5",
    "m1": "#d95926",
    "m2": "#c98500",
    "moc": "#9085e9",
    "srcc": "#199e70",
    "icq": "#d55181",
    "truth": "#e6edf3",
}

# Two-channel merges. Green/magenta is the default because it survives the common
# red-green colour-vision deficiencies; overlap reads as white either way.
MERGES = {
    "green / magenta": ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
    "red / green": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "cyan / yellow": ((0.0, 0.85, 1.0), (1.0, 0.85, 0.0)),
}

# What the top row shows: both channels merged, one channel alone, or the detected
# objects (spots) over a dimmed merge.
VIEWS = ("both channels", "channel 1", "channel 2", "objects (spots)")

BANNER = "background: #3a2f18; color: #d9c45a; padding: 4px; border-radius: 3px;"
