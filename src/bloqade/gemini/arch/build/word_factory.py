"""Word creation helpers for zone-based architectures.

Creates row-words arranged in a 2D grid. CZ pairing between horizontally
adjacent words is defined at the zone level via entangling_pairs,
not per-word.

Words now contain grid index pairs (x_idx, y_idx). Physical positions are
resolved via the zone's Grid at the ArchSpec level. Each word has unique
grid indices that map to distinct positions within the zone grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.lanes.bytecode.word import Word

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .blueprint import DeviceLayout, ZoneSpec


@dataclass(frozen=True)
class WordGrid:
    """2D grid of words within a zone, preserving row/col structure.

    Word IDs are zone-local (``0..Nw-1``) under the shared-template
    convention: every zone reuses the same word IDs, disambiguated by
    ``zone_id`` at the ArchSpec level.
    """

    words: tuple[Word, ...]
    num_rows: int
    num_cols: int

    def word_at(self, row: int, col: int) -> Word:
        """Get the word at a given grid position."""
        return self.words[row * self.num_cols + col]

    def word_id_at(self, row: int, col: int) -> int:
        """Get the zone-local word ID at a given grid position."""
        return row * self.num_cols + col

    @property
    def all_word_ids(self) -> range:
        """All zone-local word IDs in this grid (``0..Nw-1``)."""
        return range(len(self.words))

    def cz_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield (word_id_a, word_id_b) for all CZ entangling pairs."""
        for row in range(self.num_rows):
            for col in range(0, self.num_cols, 2):
                yield (self.word_id_at(row, col), self.word_id_at(row, col + 1))


def create_zone_words(
    zone_spec: ZoneSpec,
    layout: DeviceLayout,
) -> WordGrid:
    """Create all words for a zone in a 2D grid layout.

    Each word's sites are represented as grid index pairs (x_idx, y_idx).
    For interleaved CZ pairs:
    - Even-column words: sites at x_idx = 0, 2, 4, ... (in the full grid)
    - Odd-column words: sites at x_idx = 1, 3, 5, ...

    Words in different rows use different y_idx values.
    """
    n = layout.sites_per_word
    num_cols = zone_spec.num_cols

    words: list[Word] = []

    for row in range(zone_spec.num_rows):
        for col in range(num_cols):
            pair_idx = col // 2
            is_odd = col % 2
            # Number of x positions per pair: 2*n (interleaved)
            x_base = pair_idx * 2 * n
            if is_odd:
                # Odd column: sites at odd x indices
                sites = tuple((x_base + 2 * i + 1, row) for i in range(n))
            else:
                # Even column: sites at even x indices
                sites = tuple((x_base + 2 * i, row) for i in range(n))
            words.append(Word(sites))

    return WordGrid(
        words=tuple(words),
        num_rows=zone_spec.num_rows,
        num_cols=zone_spec.num_cols,
    )
