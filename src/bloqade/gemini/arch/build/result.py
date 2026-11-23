from dataclasses import dataclass

from bloqade.lanes.arch.spec import ArchSpec

from .word_factory import WordGrid


@dataclass(frozen=True)
class ArchResult:
    """An example architecture with its word grids and zone IDs."""

    arch: ArchSpec
    zone_grids: dict[str, WordGrid]
    zone_indices: dict[str, int]
