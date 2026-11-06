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

"""Headless runner: ``python -m colocwidget`` prints the explained coefficients.

The same model the widget draws, with the same numbers, for a terminal or a test.
Any lever -- or any parameter of the current scene -- can be set with
``--set name=value``; ``--ui`` opens the window.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .physics.imaging import Levers, Simulator
from .physics.presets import PRESETS
from .physics.report import build_report, format_report
from .physics.scenes import SCENES


def parse_assignment(levers: Levers, text: str) -> Levers:
    """Apply one ``name=value`` to the levers or the scene's parameters, coercing types."""
    if "=" not in text:
        raise SystemExit(f"--set expects name=value, got {text!r}")
    name, raw = (part.strip() for part in text.split("=", 1))
    scene_params = {p.name for p in levers.scene_model.params}
    if name in scene_params:
        try:
            return levers.with_(params={name: float(raw)})
        except ValueError:
            raise SystemExit(f"{name} expects a number, got {raw!r}") from None
    fields = {f.name for f in dataclasses.fields(Levers)} - {"scene_params"}
    if name not in fields:
        known = ", ".join(sorted(fields | scene_params))
        raise SystemExit(f"unknown lever {name!r}; known: {known}")
    value = getattr(levers, name)
    if raw.lower() == "none":
        coerced = None
    elif isinstance(value, bool):
        coerced = raw.lower() in ("1", "true", "on", "yes")
    elif isinstance(value, int):
        coerced = int(raw)
    elif isinstance(value, float) or value is None:
        try:
            coerced = float(raw)
        except ValueError:
            coerced = raw
    else:
        coerced = raw
    if name == "scene":
        if coerced not in SCENES:
            raise SystemExit(f"no scene {coerced!r}; scenes: {', '.join(SCENES)}")
        # A new scene starts from its own parameters and focal plane.
        return levers.with_(scene=coerced, scene_params=(), focus_um=None)
    return levers.with_(**{name: coerced})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="colocwidget")
    parser.add_argument("--ui", action="store_true", help="open the widget")
    parser.add_argument("--preset", help="start from a named preset (see --list-presets)")
    parser.add_argument("--list-presets", action="store_true", help="list the presets")
    parser.add_argument("--list-scenes", action="store_true", help="list the scenes and their parameters")
    parser.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE",
        help="set a lever or scene parameter, e.g. --set modality=STED --set colabelled=1",
    )
    parser.add_argument("--realizations", type=int, default=None, help="repeat frames for the noise spread")
    args = parser.parse_args(argv)

    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"{name}\n    {preset.description}")
        return 0
    if args.list_scenes:
        for scene in SCENES.values():
            print(f"{scene.name}\n    {scene.description}")
            for p in scene.params:
                print(f"      {p.name} = {p.default:g}{p.unit}  [{p.lo:g} .. {p.hi:g}]  {p.label}")
        return 0

    if args.preset:
        if args.preset not in PRESETS:
            print(f"no preset {args.preset!r}; try --list-presets")
            return 2
        levers = PRESETS[args.preset].levers
    else:
        levers = Levers()
    for item in args.set:
        levers = parse_assignment(levers, item)
    if args.realizations is not None:
        levers = levers.with_(realizations=args.realizations)

    if args.ui:
        try:
            from .ui.widget import launch
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            print(f"the widget needs the 'ui' extra: pip install 'colocwidget[ui]' ({exc})")
            return 2
        return launch(levers)

    try:
        sim = Simulator().run(levers)
    except ValueError as exc:
        print(f"cannot image this: {exc}")
        return 2
    print(f"[{levers.scene}]  {levers.modality}  |  NA {levers.na:g} {levers.medium}  |  "
          f"{levers.emission_nm_1:.0f} / {levers.emission_nm_2:.0f} nm  |  ROI: {sim.roi_name}")
    print(format_report(build_report(sim)))
    for note in sim.notes:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
