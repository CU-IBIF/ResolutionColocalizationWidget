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

"""Starting points that each make one point.

A preset is a set of levers plus the sentence it is there to demonstrate. They come
in pairs where that helps -- the same specimen at widefield and at STED, or the
interleaved cluster next to the co-labelled one -- so the comparison is one click.
Numbered by the three cases the widget is built around:

1. unresolved collections that look like one co-labelled body;
2. how the distribution and morphology of the signal move the coefficients;
3. model cells, with labels split across nucleus, cytosol and plasma membrane.
"""

from __future__ import annotations

from dataclasses import dataclass

from .imaging import Levers


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    levers: Levers


def L(scene: str = "Puncta clusters", params: dict | None = None, **kw) -> Levers:
    return Levers(scene=scene, **kw).with_(params=params or {})


STED = dict(modality="STED", sted_fwhm_nm=50.0, pixel_nm=25.0)
AIRYSCAN = dict(modality="pixel reassignment (Airyscan-type)", pixel_nm=40.0)
CONFOCAL = dict(modality="confocal (1 AU)")
CELL = "Model cell"
_BODIES = {"ch1_nucleus": 1.0, "ch1_membrane": 0.0, "ch2_nucleus": 1.0, "ch2_cytosol": 0.0,
           "ch1_nuc_puncta": 0.8, "ch2_nuc_puncta": 0.8}

PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        # --- 1. unresolved collections --------------------------------------------
        Preset(
            "1a. Interleaved puncta, widefield",
            "No punctum carries both labels -- yet each cluster is one spot in both colours. "
            "Pearson's r and Manders' M1/M2 report colocalization the molecules do not have.",
            L(),
        ),
        Preset(
            "1b. ... the same field at STED (50 nm)",
            "Resolve the puncta and the colocalization disappears: the coefficients fall toward "
            "the molecular truth.",
            L(**STED),
        ),
        Preset(
            "1c. Co-labelled bodies, widefield",
            "Every punctum carries both labels. At widefield this image is nearly indistinguishable "
            "from 1a: the coefficients cannot tell the two specimens apart.",
            L(params={"colabelled": 1.0}),
        ),
        Preset(
            "1d. ... the same field at STED (50 nm)",
            "Truly co-labelled puncta stay colocalized when resolved. Only the resolved pair "
            "1b/1d separates the two specimens.",
            L(params={"colabelled": 1.0}, **STED),
        ),
        Preset(
            "1e. Segregated halves, pixel reassignment",
            "Each colour holds one half of every 600 nm cluster (adjacent sub-domains). Segregation at "
            "the cluster scale shows up sooner than interleaving at the punctum scale: compare r here "
            "and at widefield with 1a.",
            L(params={"segregation": 1.0, "cluster_radius_nm": 300.0}, **AIRYSCAN),
        ),
        # --- 2. distribution and morphology ----------------------------------------
        Preset(
            "2a. Adjacent filaments 120 nm apart, widefield",
            "Two filaments that never touch overlap completely in the image. Drag the separation "
            "to find where the coefficients start to fall: that is the resolution.",
            L("Adjacent filaments"),
        ),
        Preset(
            "2b. ... resolved by STED",
            "The same filaments at 50 nm: two lines, r near 0.",
            L("Adjacent filaments", **STED),
        ),
        Preset(
            "2c. Vesicle membrane vs lumen cargo",
            "A membrane protein and the cargo inside share a vesicle but rarely come within 25 nm. "
            "Even at STED r stays high: in projection the vesicle's top and bottom membrane lie over "
            "the cargo. Lateral resolution alone cannot separate them.",
            L("Vesicles: membrane and lumen"),
        ),
        Preset(
            "2d. Dense independent signals: Manders' inflated",
            "Unrelated molecules, randomly placed, ~3 per PSF. r is ~0, but with no threshold every "
            "pixel holds both channels, so M1 and M2 reach 1. Switch the threshold to Costes and watch "
            "them fall.",
            L("Dense independent signals", threshold="none"),
        ),
        Preset(
            "2e. Shared territory: correlation without binding",
            "The same independent molecules confined to one irregular region. Pearson's r rises "
            "because both channels share the region, not because they bind -- an ROI matters.",
            L("Dense independent signals", params={"patchy": 1.0}),
        ),
        Preset(
            "2f. Variable stoichiometry: Pearson's vs Manders'",
            "Every punctum carries both labels, in varying ratio. Co-occurrence (M1, M2) stays "
            "high while proportionality (r) falls: the two families answer different questions.",
            L("Co-labelled puncta, variable stoichiometry", params={"ratio_spread": 1.2}),
        ),
        Preset(
            "2g. Exclusive domains, resolved",
            "Mutually exclusive 400 nm domains: strongly negative r and an ICQ near -0.5.",
            L("Exclusive domains"),
        ),
        Preset(
            "2h. Exclusive domains below the resolution",
            "60 nm domains under a 190 nm PSF: the exclusion is blurred toward r ~ 0 and M1, M2 rise. "
            "'Weak correlation' is not 'no relationship'.",
            L("Exclusive domains", params={"domain_nm": 60.0}),
        ),
        Preset(
            "2i. Chromatic shift (150 nm)",
            "Truly co-labelled puncta, channel 2 registered 150 nm off. r and M1/M2 fall, and the "
            "cross-correlation peak moves off zero by the shift.",
            L(params={"colabelled": 1.0}, shift_x_nm=150.0),
        ),
        Preset(
            "2j. Bleed-through (30%) manufactures colocalization",
            "The resolved, interleaved puncta of 1b -- but 30% of channel 1 leaks into channel 2's "
            "detector. Crosstalk is a copy of channel 1, so r roughly triples with no change to the specimen.",
            L(bleed_through=0.30, **STED),
        ),
        # --- 3. cells and compartments --------------------------------------------
        Preset(
            "3a. Cell: membrane vs cytosol, widefield",
            "Ch1 on the plasma membrane, ch2 in the cytosol. Widefield lays the apical and basal "
            "membranes over the whole cell: most of ch1's in-focus signal lands in the cytosol and "
            "nucleus regions (compartments tab). Whole-field r is strongly positive, in-cell r "
            "negative -- neither is about binding.",
            L(CELL, roi="cell"),
        ),
        Preset(
            "3b. ... the same cell, confocal section",
            "A pinhole rejects the membranes above and below: the membrane becomes a rim, and its "
            "apparent share over the nucleus falls from ~25% to under 10%.",
            L(CELL, roi="cell", **CONFOCAL),
        ),
        Preset(
            "3c. Cell: split across nucleus, cytosol and membrane",
            "Ch1: half nuclear, half membrane. Ch2: half nuclear, half cytosolic. They share only the "
            "nucleus, yet whole-cell r is ~0.95: the bright shared nucleus dominates the sum. See the "
            "compartments tab for the three different answers it averages.",
            L(CELL, params={"ch1_nucleus": 0.5, "ch1_membrane": 0.5, "ch2_nucleus": 0.5,
                            "ch2_cytosol": 0.5}, roi="cell", **CONFOCAL),
        ),
        Preset(
            "3d. ... measured in the nucleus only",
            "The same cell, measured only in the nucleus, where both channels truly are. A per-"
            "compartment ROI answers the question the whole-cell number blurred.",
            L(CELL, params={"ch1_nucleus": 0.5, "ch1_membrane": 0.5, "ch2_nucleus": 0.5,
                            "ch2_cytosol": 0.5}, roi="nucleus", **CONFOCAL),
        ),
        Preset(
            "3e. Nuclear vs cytosolic, widefield: the nuclear ROI is contaminated",
            "Ch1 purely nuclear, ch2 purely cytosolic. In widefield the cytosol above and below the "
            "nucleus is projected into the nuclear ROI: ~20% of ch2's in-focus signal 'appears' there. "
            "Switch to confocal and it halves.",
            L(CELL, params={"ch1_nucleus": 1.0, "ch1_membrane": 0.0, "ch2_cytosol": 1.0}, roi="nucleus"),
        ),
        Preset(
            "3f. Whole field vs cell: the empty background inflates r",
            "Membrane vs cytosol in a confocal section, measured over the whole field: the dark area "
            "outside the cell is 'low in both' and lifts r from ~0 to ~0.4. Set the region to 'cell'.",
            L(CELL, roi="whole field", **CONFOCAL),
        ),
        Preset(
            "3g. Nuclear bodies: same compartment, separate bodies",
            "Both channels are entirely nuclear, 80% of each in nuclear bodies -- but in different bodies. "
            "The compartment truth says 100% shared; a confocal section resolves the bodies and finds "
            "r ~ 0. Drag 'puncta shared' toward 1, or sweep it.",
            L(CELL, params=_BODIES, roi="nucleus", **CONFOCAL),
        ),
        Preset(
            "3h. ... the same bodies",
            "The same nucleus with both channels in the same bodies: r and M1/M2 near 1. Only the "
            "bodies' identity changed, not the compartment.",
            L(CELL, params={**_BODIES, "shared_puncta": 1.0}, roi="nucleus", **CONFOCAL),
        ),
    )
}
