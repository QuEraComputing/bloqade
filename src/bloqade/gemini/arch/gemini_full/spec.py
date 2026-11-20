"""Gemini full architecture: 3-zone neutral atom processor.

Layout: storage_top → entangling → storage_bottom.
Each zone has 5 rows x 2 columns = 10 words, with 17 sites per word. All
zones share ONE 10-word template (170 template sites) addressed by
zone-local word IDs; ``zone_id`` disambiguates. The entangling zone
declares 5 entangling pairs (85 CZ locations).
"""

from bloqade.gemini.arch._three_zone import three_zone_arch
from bloqade.gemini.arch.build.result import ArchResult


def get_arch() -> ArchResult:
    """Build the gemini full architecture and its zone metadata."""
    return three_zone_arch(
        num_rows=5, num_cols=2, sites_per_word=17, storage_buses=False
    )
