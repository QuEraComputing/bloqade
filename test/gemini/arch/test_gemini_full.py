"""Tests for the Gemini full architecture and related topology extensions."""

from __future__ import annotations

import pytest

from bloqade.gemini.arch.gemini_full import get_arch
from bloqade.gemini.arch.build.topology import HypercubeSiteTopology

# ── HypercubeSiteTopology: non-power-of-2 extension ──


class TestHypercubeSiteTopologyExtended:
    def test_power_of_two_unchanged(self):
        """N=16 still produces 4 buses with 8 elements each."""
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(16)
        assert len(buses) == 4
        for bus in buses:
            assert len(bus.src) == 8
            assert len(bus.dst) == 8

    def test_n17_produces_5_buses(self):
        """N=17 rounds up to 32 (5 dims), producing 5 buses."""
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(17)
        assert len(buses) == 5

    def test_n17_lower_dims_are_full(self):
        """Dimensions 0-3 have full 8-element buses (sites 0-15)."""
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(17)
        for bus in buses[:4]:
            assert len(bus.src) == 8
            assert len(bus.dst) == 8

    def test_n17_site16_has_single_connection(self):
        """Site 16 only appears in dimension 4, connecting to site 0."""
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(17)
        dim4_bus = buses[4]
        assert dim4_bus.src == [0]
        assert dim4_bus.dst == [16]

    def test_n1_produces_no_buses(self):
        """N=1 has no dimensions, no buses."""
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(1)
        assert len(buses) == 0

    def test_n3_produces_2_buses(self):
        """N=3 rounds up to 4 (2 dims). Dim 1 bus is filtered."""
        topo = HypercubeSiteTopology()
        buses = topo.generate_site_buses(3)
        assert len(buses) == 2
        # Dim 0: sites 0↔1 (site 2 has bit0=0, partner 3 is out of range)
        assert buses[0].src == [0]
        assert buses[0].dst == [1]
        # Dim 1: site 0↔2 (site 1 partner is 3, out of range)
        assert buses[1].src == [0]
        assert buses[1].dst == [2]


# ── Gemini full architecture ──


@pytest.fixture(scope="module")
def gemini_full():
    return get_arch()


class TestGeminiFull:
    def test_word_count(self, gemini_full):
        # Shared template: all 3 zones reuse ONE 10-word list (5 rows x 2
        # cols), addressed zone-locally.
        assert len(gemini_full.arch.words) == 10

    def test_site_count(self, gemini_full):
        # Template sites: 10 words x 17 sites.
        total_sites = sum(len(w.sites) for w in gemini_full.arch.words)
        assert total_sites == 170

    def test_zone_count(self, gemini_full):
        assert len(gemini_full.arch.zones) == 3

    def test_zone_names(self, gemini_full):
        names = [z.name for z in gemini_full.arch.zones]
        assert names == ["storage_top", "entangling", "storage_bottom"]

    def test_storage_zones_have_no_buses(self, gemini_full):
        for i in (0, 2):
            zone = gemini_full.arch.zones[i]
            assert len(zone.site_buses) == 0
            assert len(zone.word_buses) == 0

    def test_storage_zones_have_no_entangling_pairs(self, gemini_full):
        for i in (0, 2):
            zone = gemini_full.arch.zones[i]
            assert len(zone.entangling_pairs) == 0

    def test_entangling_zone_site_buses(self, gemini_full):
        zone = gemini_full.arch.zones[1]
        assert len(zone.site_buses) == 5

    def test_entangling_zone_word_buses(self, gemini_full):
        zone = gemini_full.arch.zones[1]
        assert len(zone.word_buses) == 9

    def test_entangling_zone_pairs(self, gemini_full):
        zone = gemini_full.arch.zones[1]
        assert len(zone.entangling_pairs) == 5

    def test_zone_buses(self, gemini_full):
        assert len(gemini_full.arch.zone_buses) == 2
        for zb in gemini_full.arch.zone_buses:
            assert len(zb.src) == 10

    def test_feed_forward(self, gemini_full):
        assert gemini_full.arch.feed_forward is True

    def test_atom_reloading(self, gemini_full):
        assert gemini_full.arch.atom_reloading is True

    def test_blockade_radius(self, gemini_full):
        assert gemini_full.arch.blockade_radius == 10.0

    def test_zone_grids_metadata(self, gemini_full):
        assert set(gemini_full.zone_grids.keys()) == {
            "storage_top",
            "entangling",
            "storage_bottom",
        }

    def test_zone_indices(self, gemini_full):
        assert gemini_full.zone_indices == {
            "storage_top": 0,
            "entangling": 1,
            "storage_bottom": 2,
        }
