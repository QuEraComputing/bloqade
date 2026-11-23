"""Tests for the generic full architecture and DiagonalWordTopology generalization."""

from __future__ import annotations

import pytest

from bloqade.gemini.arch.generic_full import get_arch
from bloqade.gemini.arch.build.topology import DiagonalWordTopology
from bloqade.gemini.arch.build.word_factory import WordGrid, create_zone_words

# ── DiagonalWordTopology: multi-column support ──


class TestDiagonalWordTopologyMultiColumn:
    def _make_grid(self, num_rows: int, num_cols: int) -> WordGrid:
        return create_zone_words(num_rows=num_rows, num_cols=num_cols, sites_per_word=2)

    def test_2_cols_unchanged(self):
        """2-column grid still produces 2*N-1 buses."""
        grid = self._make_grid(5, 2)
        topo = DiagonalWordTopology()
        buses = topo.generate_word_buses(grid)
        assert len(buses) == 2 * 5 - 1  # 9

    def test_4_cols_produces_45_buses(self):
        """4-column, 8-row grid: 3 pairs × (2*8-1) = 45 buses."""
        grid = self._make_grid(8, 4)
        topo = DiagonalWordTopology()
        buses = topo.generate_word_buses(grid)
        assert len(buses) == 3 * (2 * 8 - 1)  # 45

    def test_4_cols_buses_connect_adjacent_only(self):
        """Each bus only connects words from adjacent columns."""
        grid = self._make_grid(4, 4)
        topo = DiagonalWordTopology()
        buses = topo.generate_word_buses(grid)
        for bus in buses:
            # Word IDs are zone-local (0..Nw-1); column = id % num_cols.
            src_cols = {w % 4 for w in bus.src}
            dst_cols = {w % 4 for w in bus.dst}
            # src and dst should each be from a single column
            assert len(src_cols) == 1
            assert len(dst_cols) == 1
            col_a = src_cols.pop()
            col_b = dst_cols.pop()
            assert abs(col_a - col_b) == 1


# ── Generic full architecture ──


@pytest.fixture(scope="module")
def generic_full():
    return get_arch()


class TestGenericFull:
    def test_word_count(self, generic_full):
        # Shared template: all 3 zones reuse ONE 32-word list (8 rows x 4
        # cols), addressed zone-locally.
        assert len(generic_full.arch.words) == 32

    def test_site_count(self, generic_full):
        # Template sites: 32 words x 8 sites.
        total_sites = sum(len(w.sites) for w in generic_full.arch.words)
        assert total_sites == 256

    def test_zone_count(self, generic_full):
        assert len(generic_full.arch.zones) == 3

    def test_zone_names(self, generic_full):
        names = [z.name for z in generic_full.arch.zones]
        assert names == ["storage_top", "entangling", "storage_bottom"]

    def test_all_zones_have_site_buses(self, generic_full):
        for zone in generic_full.arch.zones:
            assert len(zone.site_buses) == 3  # hypercube on 8 = 2^3 sites

    def test_all_zones_have_word_buses(self, generic_full):
        for zone in generic_full.arch.zones:
            assert len(zone.word_buses) == 45  # 3 pairs × 15

    def test_entangling_zone_pairs(self, generic_full):
        zone = generic_full.arch.zones[1]
        assert len(zone.entangling_pairs) == 16  # 8 rows × 2 pairs/row

    def test_storage_zones_no_entangling_pairs(self, generic_full):
        for i in (0, 2):
            assert len(generic_full.arch.zones[i].entangling_pairs) == 0

    def test_zone_buses(self, generic_full):
        assert len(generic_full.arch.zone_buses) == 2
        for zb in generic_full.arch.zone_buses:
            assert len(zb.src) == 32

    def test_feed_forward(self, generic_full):
        assert generic_full.arch.feed_forward is True

    def test_atom_reloading(self, generic_full):
        assert generic_full.arch.atom_reloading is True

    def test_blockade_radius(self, generic_full):
        assert generic_full.arch.blockade_radius == 10.0
