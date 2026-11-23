"""Tests for topology protocols and implementations."""

import pytest

from bloqade.gemini.arch.build.topology import (
    MatchingTopology,
    AllToAllSiteTopology,
    HypercubeSiteTopology,
    HypercubeWordTopology,
)
from bloqade.gemini.arch.build.word_factory import WordGrid, create_zone_words


def _make_grid(num_rows: int = 2, num_cols: int = 4) -> WordGrid:
    return create_zone_words(num_rows=num_rows, num_cols=num_cols, sites_per_word=4)


# ── Site topologies ──


class TestHypercubeSiteTopology:
    def test_4_sites(self) -> None:
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(4)
        assert len(buses) == 2  # log2(4) = 2 dims

        # dim 0: bit 0 flip — (0↔1), (2↔3)
        assert set(buses[0].src) == {0, 2}
        assert set(buses[0].dst) == {1, 3}
        # dim 1: bit 1 flip — (0↔2), (1↔3)
        assert set(buses[1].src) == {0, 1}
        assert set(buses[1].dst) == {2, 3}

    def test_8_sites(self) -> None:
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(8)
        assert len(buses) == 3  # log2(8) = 3 dims
        for bus in buses:
            assert len(bus.src) == 4
            assert len(bus.dst) == 4
            assert set(bus.src).isdisjoint(set(bus.dst))

    def test_disjointness(self) -> None:
        topo = HypercubeSiteTopology()
        for n in [2, 4, 8]:
            buses = topo.generate_site_buses(n)
            for bus in buses:
                assert set(bus.src).isdisjoint(set(bus.dst))

    def test_non_power_of_two_accepted(self) -> None:
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(5)
        # 5 rounds up to 8 (3 dims), buses are filtered to indices < 5
        assert len(buses) == 3
        for bus in buses:
            assert all(s < 5 for s in bus.src)
            assert all(d < 5 for d in bus.dst)


class TestAllToAllSiteTopology:
    def test_4_sites(self) -> None:
        topo = AllToAllSiteTopology()
        buses = topo.generate_site_buses(4)
        assert len(buses) == 6  # 4*3/2

    def test_pairs_cover_all(self) -> None:
        topo = AllToAllSiteTopology()
        buses = topo.generate_site_buses(4)
        pairs = {(bus.src[0], bus.dst[0]) for bus in buses}
        expected = {(i, j) for i in range(4) for j in range(i + 1, 4)}
        assert pairs == expected

    def test_disjointness(self) -> None:
        topo = AllToAllSiteTopology()
        buses = topo.generate_site_buses(5)
        for bus in buses:
            assert set(bus.src).isdisjoint(set(bus.dst))

    def test_single_element_buses(self) -> None:
        topo = AllToAllSiteTopology()
        buses = topo.generate_site_buses(3)
        for bus in buses:
            assert len(bus.src) == 1
            assert len(bus.dst) == 1


# ── Word topologies ──


class TestHypercubeWordTopology:
    def test_2x2_grid(self) -> None:
        grid = _make_grid(num_rows=2, num_cols=2)
        topo = HypercubeWordTopology()
        buses = topo.generate_word_buses(grid)
        # 1 row dim + 1 col dim = 2 buses
        assert len(buses) == 2

    def test_2x4_grid(self) -> None:
        grid = _make_grid(num_rows=2, num_cols=4)
        topo = HypercubeWordTopology()
        buses = topo.generate_word_buses(grid)
        # 1 row dim + 2 col dims = 3 buses
        assert len(buses) == 3

    def test_4x4_grid(self) -> None:
        grid = _make_grid(num_rows=4, num_cols=4)
        topo = HypercubeWordTopology()
        buses = topo.generate_word_buses(grid)
        # 2 row dims + 2 col dims = 4 buses
        assert len(buses) == 4

    def test_row_dim_connects_rows(self) -> None:
        grid = _make_grid(num_rows=2, num_cols=2)
        topo = HypercubeWordTopology()
        buses = topo.generate_word_buses(grid)
        # First bus is row dim 0: w(0,c) ↔ w(1,c)
        row_bus = buses[0]
        assert row_bus.src == [grid.word_id_at(0, 0), grid.word_id_at(0, 1)]
        assert row_bus.dst == [grid.word_id_at(1, 0), grid.word_id_at(1, 1)]

    def test_col_dim_connects_cols(self) -> None:
        grid = _make_grid(num_rows=2, num_cols=2)
        topo = HypercubeWordTopology()
        buses = topo.generate_word_buses(grid)
        # Second bus is col dim 0: w(r,0) ↔ w(r,1)
        col_bus = buses[1]
        assert col_bus.src == [grid.word_id_at(0, 0), grid.word_id_at(1, 0)]
        assert col_bus.dst == [grid.word_id_at(0, 1), grid.word_id_at(1, 1)]

    def test_word_ids_are_zone_local(self) -> None:
        grid = _make_grid(num_rows=2, num_cols=2)
        topo = HypercubeWordTopology()
        buses = topo.generate_word_buses(grid)
        all_ids = set()
        for bus in buses:
            all_ids.update(bus.src)
            all_ids.update(bus.dst)
        # Zone-local IDs 0..Nw-1 (shared template; no per-zone offset).
        assert all_ids == {0, 1, 2, 3}

    def test_non_power_of_two_rows_raises(self) -> None:
        grid = _make_grid(num_rows=2, num_cols=4)
        # Hack the grid to have 3 rows
        bad_grid = WordGrid(words=grid.words, num_rows=3, num_cols=4)
        topo = HypercubeWordTopology()
        with pytest.raises(ValueError, match="power of 2"):
            topo.generate_word_buses(bad_grid)


# ── Inter-zone topologies ──


class TestMatchingTopology:
    def test_1_to_1_matching(self) -> None:
        grid_a = _make_grid(num_rows=2, num_cols=2)
        grid_b = _make_grid(num_rows=2, num_cols=2)
        topo = MatchingTopology()
        buses = topo.generate_word_buses(grid_a, grid_b)
        assert len(buses) == 1
        bus = buses[0]
        assert len(bus.src) == 4
        assert len(bus.dst) == 4
        # Position (0,0) → (0,0), etc.
        for r in range(2):
            for c in range(2):
                idx = r * 2 + c
                assert bus.src[idx] == grid_a.word_id_at(r, c)
                assert bus.dst[idx] == grid_b.word_id_at(r, c)

    def test_mismatched_dimensions_raises(self) -> None:
        grid_a = _make_grid(num_rows=2, num_cols=2)
        grid_b = _make_grid(num_rows=2, num_cols=4)
        topo = MatchingTopology()
        with pytest.raises(ValueError, match="Grid dimensions must match"):
            topo.generate_word_buses(grid_a, grid_b)
