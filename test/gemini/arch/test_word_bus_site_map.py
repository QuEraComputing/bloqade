import json

import pytest
from bloqade.lanes.bytecode import SiteBus
from bloqade.lanes.bytecode.encoding import Direction, LocationAddress, WordLaneAddress

from bloqade.gemini.arch import ArchBuilder
from bloqade.gemini.visualize.arch import ArchVisualizer


def make_builder(*, sites=(0, 1, 2)):
    builder = ArchBuilder(grid_shape=(2, 4), word_shape=(1, 3))
    builder.add_word(rows=0, columns=[0, 1, 2])
    builder.add_word(rows=1, columns=[1, 2, 3])
    builder.add_zone(
        "z",
        rows=[0, 20],
        columns=[0, 10, 20, 30],
        x_clearance=5,
        y_clearance=5,
        sites_with_word_buses=sites,
    )
    return builder


def test_mapped_paths_and_round_trip():
    builder = make_builder()
    site_map = SiteBus(src=[1, 2], dst=[0, 1])
    builder.add_word_bus("z", [0], [1], site_map=site_map)
    spec = builder.build()
    assert json.loads(spec.to_json())["zones"][0]["word_buses"][0]["site_map"] == {
        "src": [1, 2],
        "dst": [0, 1],
    }
    assert set(spec.paths) == set(spec.iter_all_lanes())
    forward_paths = []
    for src, dst in zip(site_map.src, site_map.dst):
        lane = WordLaneAddress(0, src, 0, Direction.FORWARD, 0)
        endpoints = (LocationAddress(0, src, 0), LocationAddress(1, dst, 0))
        assert spec.get_endpoints(lane) == endpoints
        assert spec.get_endpoints(lane.reverse()) == endpoints[::-1]
        path = spec.paths[lane]
        assert path[0] == (src * 10, 0) and path[-1] == ((dst + 1) * 10, 20)
        assert spec.paths[lane.reverse()] == path[::-1]
        forward_paths.append(path)
    assert list(ArchVisualizer(spec).iter_word_bus_paths([0])) == forward_paths
    assert ArchBuilder.from_spec(spec).build().to_json() == spec.to_json()
    rebuilt = ArchBuilder.from_spec(spec, x_clearance=5, y_clearance=5)
    assert rebuilt.build(recompute_paths=True).paths == spec.paths


def test_explicit_empty_map_transports_no_sites():
    builder = make_builder()
    builder.add_word_bus("z", [0], [1], site_map=SiteBus([], []))
    spec = builder.build()
    assert list(spec.iter_all_lanes()) == []
    assert not spec.paths
    assert ArchBuilder.from_spec(spec).build().zones[0].word_buses[
        0
    ].site_map == SiteBus([], [])


def test_word_buses_with_distinct_maps_can_share_word_endpoints():
    builder = make_builder()
    for source, destination in [(1, 0), (2, 1)]:
        builder.add_word_bus("z", [0], [1], site_map=SiteBus([source], [destination]))
    with pytest.raises(ValueError, match="word bus 0 already"):
        builder.add_word_bus("z", [0], [1], site_map=SiteBus([1], [0]))
    assert len(builder.build().zones[0].word_buses) == 2


def test_unmapped_paths_only_include_participating_sites():
    builder = make_builder(sites=[1])
    builder.add_word_bus("z", [0], [1])
    spec = builder.build()
    assert set(spec.paths) == set(spec.iter_all_lanes())
    assert spec.zones[0].word_buses[0].site_map is None


@pytest.mark.parametrize(
    "src,dst,error",
    [
        ([1], [], "length"),
        ([1], [2], "sites_with_word_buses"),
        ([0, 1], [1, 0], "cycle"),
        ([1], [3], "sites_per_word"),
    ],
)
def test_invalid_site_map_leaves_builder_usable(src, dst, error):
    builder = make_builder(sites=[0, 1])
    with pytest.raises(ValueError, match=error):
        builder.add_word_bus("z", [0], [1], site_map=SiteBus(src, dst))
    builder.add_word_bus("z", [0], [1], site_map=SiteBus([1], [0]))
    assert len(builder.build().zones[0].word_buses) == 1
