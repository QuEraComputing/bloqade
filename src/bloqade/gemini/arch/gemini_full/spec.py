"""Gemini full architecture: 3-zone neutral atom processor.

Layout: storage_top → entangling → storage_bottom.
Each zone has 5 rows x 2 columns = 10 words, with 17 sites per word. All
zones share ONE 10-word template (170 template sites) addressed by
zone-local word IDs; ``zone_id`` disambiguates. The entangling zone
declares 5 entangling pairs (85 CZ locations).
"""

from __future__ import annotations

from bloqade.gemini.arch.build.topology import (
    MatchingTopology,
    DiagonalWordTopology,
    HypercubeSiteTopology,
)
from bloqade.gemini.arch.build.blueprint import (
    ZoneSpec,
    ArchResult,
    DeviceLayout,
    ArchBlueprint,
    build_arch,
)

_SITES_PER_WORD = 17
_NUM_ROWS = 5
_NUM_COLS = 2
_SITE_SPACING = 10.0


def get_arch() -> ArchResult:
    """Build the Gemini full architecture.

    Returns:
        ArchResult with the validated ArchSpec and zone metadata.
    """
    entangling_zone = ZoneSpec(
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        entangling=True,
        measurement=True,
        site_topology=HypercubeSiteTopology(),
        word_topology=DiagonalWordTopology(),
    )

    storage_zone = ZoneSpec(
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        entangling=False,
        measurement=False,
    )

    blueprint = ArchBlueprint(
        zones={
            "storage_top": storage_zone,
            "entangling": entangling_zone,
            "storage_bottom": storage_zone,
        },
        layout=DeviceLayout(
            sites_per_word=_SITES_PER_WORD,
            site_spacing=_SITE_SPACING,
            pair_spacing=_SITE_SPACING,
            row_spacing=20.0,
            zone_gap=20.0,
        ),
        feed_forward=True,
        atom_reloading=True,
        blockade_radius=_SITE_SPACING,
    )

    return build_arch(
        blueprint,
        connections={
            ("storage_top", "entangling"): MatchingTopology(),
            ("entangling", "storage_bottom"): MatchingTopology(),
        },
    )
