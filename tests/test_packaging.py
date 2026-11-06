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

"""The things that break only once the app is frozen."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_entry_point_defaults_to_opening_the_widget():
    # A double-clicked executable has no arguments, so it must land somewhere
    # useful rather than printing a report to a console nobody can see.
    source = (REPO / "packaging" / "entry.py").read_text(encoding="utf-8")
    assert "--ui" in source
    assert "freeze_support" in source


def test_the_spec_builds_one_file_named_colocwidget():
    spec = (REPO / "packaging" / "colocwidget.spec").read_text(encoding="utf-8")
    assert 'name="ColocWidget"' in spec
    assert "COLLECT(" not in spec, "COLLECT produces a one-folder build"
    assert "colocwidget.ui.widget" in spec  # imported lazily by the CLI


def test_release_smoke_test_uses_this_apps_commands():
    import yaml

    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    assert set(spec["jobs"]) == {"build", "publish"}
    assert "--frame" not in text  # a NoiseWidget flag; this CLI has none
    assert "--list-presets" in text and "--list-scenes" in text


def test_release_builds_all_four_platforms_on_runners_that_exist():
    # Same matrix as NoiseWidget and MicroscopeWidget. macos-13 was retired and never
    # schedules, so a release job pinned to it waits forever and nothing publishes.
    import yaml

    spec = yaml.safe_load((REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
    include = spec["jobs"]["build"]["strategy"]["matrix"]["include"]
    assert {m["label"] for m in include} == {"linux-x86_64", "windows-x86_64", "macos-x86_64", "macos-arm64"}
    assert "macos-13" not in {m["os"] for m in include}
    assert all(m["asset"].startswith("ColocWidget-") for m in include)


def test_pyside_is_not_a_release_that_aborts_on_setpen():
    """Guard the PySide6 exclusion, as NoiseWidget does.

    PySide6 6.12.0 over-decrefs None inside setPen; on Python 3.11 enough redraws
    abort the interpreter (exit 134) with no test named. Every pane here calls
    setPen on every paint. If this fails, verify the bug is fixed before widening
    the pin in pyproject.toml.
    """
    import pytest

    pyside6 = pytest.importorskip("PySide6")
    assert pyside6.__version__ not in {"6.12.0"}, (
        f"PySide6 {pyside6.__version__} is excluded in pyproject.toml because setPen aborts the interpreter"
    )
