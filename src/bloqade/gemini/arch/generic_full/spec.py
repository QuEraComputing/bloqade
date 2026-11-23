"""Generic full architecture: 3-zone neutral atom processor.

Layout: storage_top → entangling → storage_bottom.
Each zone has 8 rows x 4 columns = 32 words, with 8 sites per word. All
zones share ONE 32-word template (256 template sites) addressed by
zone-local word IDs; ``zone_id`` disambiguates. The entangling zone
declares 16 entangling pairs.

All zones have full site (hypercube) and word (diagonal) connectivity.
"""

from bloqade.gemini.arch._three_zone import three_zone_arch
from bloqade.gemini.arch.build.result import ArchResult


def get_arch() -> ArchResult:
    """Build the generic full architecture and its zone metadata."""
    return three_zone_arch(num_rows=8, num_cols=4, sites_per_word=8, storage_buses=True)
