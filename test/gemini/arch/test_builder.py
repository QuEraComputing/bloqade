"""Tests for the architecture builder."""

import pytest
from bloqade.lanes.arch.spec import ArchSpec

from bloqade.gemini.arch.build.topology import (
    MatchingTopology,
    HypercubeSiteTopology,
    HypercubeWordTopology,
)
from bloqade.gemini.arch.build.blueprint import (
    ZoneSpec,
    ArchResult,
    DeviceLayout,
    ArchBlueprint,
    build_arch,
)


def _has_site_buses(arch: ArchSpec) -> frozenset[int]:
    return frozenset(w for z in arch.zones for w in z.words_with_site_buses)


def _has_word_buses(arch: ArchSpec) -> frozenset[int]:
    return frozenset(s for z in arch.zones for s in z.sites_with_word_buses)


def _two_zone_blueprint(
    entangling_word_topo: bool = True,
    entangling_site_topo: bool = True,
) -> ArchBlueprint:
    """Helper: 2-zone blueprint (proc + mem), 2x2 grid, 4 sites per word."""
    return ArchBlueprint(
        zones={
            "proc": ZoneSpec(
                num_rows=2,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology() if entangling_word_topo else None,
                site_topology=HypercubeSiteTopology() if entangling_site_topo else None,
            ),
            "mem": ZoneSpec(num_rows=2, num_cols=2),
        },
        layout=DeviceLayout(sites_per_word=4, site_spacing=10.0),
    )


class TestBuildArchSingleZone:
    def test_minimal_no_topology(self) -> None:
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_rows=1, num_cols=2, entangling=True)},
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        assert isinstance(result, ArchResult)
        assert len(result.arch.words) == 2
        assert "proc" in result.zone_grids
        assert result.zone_indices == {"proc": 0}

    def test_with_word_topology(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=2,
                    num_cols=2,
                    entangling=True,
                    word_topology=HypercubeWordTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 2x2 grid → 1 row dim + 1 col dim = 2 word buses (1 zone, no split)
        assert len(result.arch.word_buses) == 2

    def test_with_site_topology(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 4 sites → log2(4) = 2 site buses (1 zone, no split)
        assert len(result.arch.site_buses) == 2
        assert _has_site_buses(result.arch) == frozenset({0, 1})


class TestBuildArchTwoZones:
    def test_two_zones_no_connection(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        # Shared word template: both zones reuse the SAME 4-word list
        # (zone-local IDs 0..3), so the global list has 4 entries, not 8.
        assert len(result.arch.words) == 4
        assert result.zone_indices == {"proc": 0, "mem": 1}
        # Only proc has word topology → 2 word buses (1 zone, no split)
        assert len(result.arch.word_buses) == 2

    def test_two_zones_with_matching(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(
            bp,
            connections={
                ("proc", "mem"): MatchingTopology(),
            },
        )
        # Proc zone: 2 hypercube word buses (no split, 1 zone)
        # Inter-zone connections go to zone_buses, not word_buses
        assert len(result.arch.word_buses) == 2
        assert len(result.arch.zone_buses) == 1

    def test_zone_grids_correct(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        proc_grid = result.zone_grids["proc"]
        mem_grid = result.zone_grids["mem"]
        # Shared template: both zones expose the same zone-local word IDs.
        assert list(proc_grid.all_word_ids) == list(mem_grid.all_word_ids)
        assert proc_grid.num_rows == 2
        assert mem_grid.num_rows == 2

    def test_entangling_pairs(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        # Proc is 1 zone with entangling pairs, mem is 1 zone without
        zones = result.arch._inner.zones
        assert len(zones) == 2
        assert len(zones[0].entangling_pairs) == 2
        assert zones[0].entangling_pairs == [(0, 1), (2, 3)]
        assert zones[1].entangling_pairs == []

    def test_measurement_zones(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        # Both blueprint zones have measurement=True (default).
        # 1:1 mapping: proc=zone 0, mem=zone 1.
        # Modes: "all" (zones [0,1]), "proc" (zones [0]), "mem" (zones [1])
        modes = result.arch.modes
        assert len(modes) == 3
        assert modes[0].name == "all"
        assert modes[0].zones == [0, 1]
        assert modes[1].name == "proc"
        assert modes[1].zones == [0]
        assert modes[2].name == "mem"
        assert modes[2].zones == [1]

    def test_has_word_buses_all_sites(self) -> None:
        bp = _two_zone_blueprint()
        result = build_arch(bp)
        assert _has_word_buses(result.arch) == frozenset({0, 1, 2, 3})

    def test_has_site_buses_union(self) -> None:
        bp = _two_zone_blueprint(entangling_site_topo=True)
        result = build_arch(bp)
        # Only proc has site_topology → only proc word IDs
        assert _has_site_buses(result.arch) == frozenset({0, 1, 2, 3})


class TestBuildArchThreeZones:
    def test_three_zones(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=2,
                    num_cols=2,
                    entangling=True,
                    word_topology=HypercubeWordTopology(),
                ),
                "buffer": ZoneSpec(num_rows=2, num_cols=2),
                "mem": ZoneSpec(num_rows=2, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(
            bp,
            connections={
                ("proc", "buffer"): MatchingTopology(),
                ("buffer", "mem"): MatchingTopology(),
            },
        )
        # Shared template: 3 zones reuse the SAME 4-word list (2x2).
        assert len(result.arch.words) == 4
        # 1:1 mapping: proc=0, buffer=1, mem=2
        assert result.zone_indices == {"proc": 0, "buffer": 1, "mem": 2}
        # Proc: 2 hypercube word buses (no split)
        # Inter-zone connections go to zone_buses
        assert len(result.arch.word_buses) == 2
        assert len(result.arch.zone_buses) == 2


class TestBuildArchValidation:
    def test_unknown_zone_in_connection_raises(self) -> None:
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_rows=1, num_cols=2)},
            layout=DeviceLayout(sites_per_word=4),
        )
        with pytest.raises(ValueError, match="Unknown zone 'mem'"):
            build_arch(bp, connections={("proc", "mem"): MatchingTopology()})

    def test_rust_validation_passes(self) -> None:
        """Ensure the generated ArchSpec passes Rust validation."""
        bp = _two_zone_blueprint()
        result = build_arch(
            bp,
            connections={
                ("proc", "mem"): MatchingTopology(),
            },
        )
        # from_components calls _inner.validate() — no exception = pass
        assert result.arch is not None

    def test_self_connection_raises(self) -> None:
        bp = _two_zone_blueprint()
        with pytest.raises(ValueError, match="Self-connection not allowed"):
            build_arch(bp, connections={("proc", "proc"): MatchingTopology()})

    def test_no_site_topology_anywhere(self) -> None:
        """All zones without site_topology → zero site buses, still valid."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_rows=1, num_cols=2, entangling=True),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        assert len(result.arch.site_buses) == 0
        assert _has_site_buses(result.arch) == frozenset()

    def test_non_measurement_zone(self) -> None:
        """Zone with measurement=False is excluded from per-zone modes."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_rows=1, num_cols=2, entangling=True),
                "buffer": ZoneSpec(num_rows=1, num_cols=2, measurement=False),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # Modes: "all" (zones [0,1,2]) + "proc" (zones [0,1]), buffer excluded
        modes = result.arch.modes
        assert len(modes) == 2
        assert modes[0].name == "all"
        assert modes[1].name == "proc"


class TestBuildArchPerBusWords:
    def test_mixed_site_topologies(self) -> None:
        """Two zones with different site topologies get separate buses."""
        from bloqade.gemini.arch.build.topology import AllToAllSiteTopology

        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    site_topology=AllToAllSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        arch = result.arch

        # Hypercube(4 sites) = 2 buses (1 zone, no split)
        # AllToAll(4 sites) = 6 buses for mem → 8 total
        assert len(arch.site_buses) == 8

        # Shared template: both zones address zone-local words [0, 1].
        assert arch.zones[0].words_with_site_buses == [0, 1]
        assert arch.zones[1].words_with_site_buses == [0, 1]

        # Union across zones collapses to the shared zone-local IDs.
        assert _has_site_buses(arch) == frozenset({0, 1})

    def test_single_zone_site_buses_have_words(self) -> None:
        """Single zone with site topology → all words have site buses."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # 1 zone, both words have site buses
        assert len(result.arch.zones) == 1
        assert result.arch.zones[0].words_with_site_buses == [0, 1]

    def test_zone_without_site_topology_excluded(self) -> None:
        """Zones without site_topology don't contribute to site buses."""
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(bp)
        # Only proc has site buses (2 from hypercube on 4 sites), mem has none
        assert len(result.arch.site_buses) == 2
        # Proc zone has both words
        assert result.arch.zones[0].words_with_site_buses == [0, 1]
        # Mem zone has no site buses
        assert result.arch.zones[1].words_with_site_buses == []
        # has_site_buses = only proc words (zone-local)
        assert _has_site_buses(result.arch) == frozenset({0, 1})


class TestPathFinderIntegration:
    def test_per_zone_site_bus_scoping(self) -> None:
        """PathFinder graph only connects site bus edges for the bus's words."""
        from bloqade.lanes.arch.path import PathFinder
        from bloqade.lanes.bytecode.encoding import LocationAddress

        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        # No connections — zones are isolated
        result = build_arch(bp)
        pf = PathFinder(result.arch)

        # proc word 0, site 0 → site 1: reachable (proc has site buses)
        assert pf.find_path(LocationAddress(0, 0), LocationAddress(0, 1)) is not None

        # mem word 2, site 0 → site 1: NOT reachable (no site buses, no connections)
        assert pf.find_path(LocationAddress(2, 0), LocationAddress(2, 1)) is None

    def test_cross_zone_reachable_via_word_bus(self) -> None:
        """PathFinder can route across zones via inter-zone zone buses."""
        from bloqade.lanes.arch.path import PathFinder
        from bloqade.lanes.bytecode.encoding import LocationAddress

        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(
                    num_rows=1,
                    num_cols=2,
                    entangling=True,
                    site_topology=HypercubeSiteTopology(),
                ),
                "mem": ZoneSpec(num_rows=1, num_cols=2),
            },
            layout=DeviceLayout(sites_per_word=4),
        )
        result = build_arch(
            bp,
            connections={
                ("proc", "mem"): MatchingTopology(),
            },
        )
        pf = PathFinder(result.arch)

        # proc word 0 (zone 0) → mem word 0 (zone 1, same zone-local ID and
        # site): reachable via the matching zone bus. Under the shared
        # template the destination word_id is zone-local (0), not global (2).
        proc_zone_id = result.zone_indices["proc"]
        mem_zone_id = result.zone_indices["mem"]
        assert (
            pf.find_path(
                LocationAddress(0, 0, proc_zone_id),
                LocationAddress(0, 0, mem_zone_id),
            )
            is not None
        )
