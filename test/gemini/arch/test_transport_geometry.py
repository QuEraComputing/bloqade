"""Transport path and blockade regressions for the shared architecture geometry."""

import pytest
from bloqade.lanes.bytecode.encoding import MoveType, Direction, LaneAddress

from bloqade.gemini.arch import ArchBuilder
from bloqade.gemini.arch.build._geometry import enumerate_safe_positions


@pytest.mark.parametrize("offsets", [[0], [0, 2000]])
def test_safe_positions_clear_every_grid_line(offsets):
    grid = [0, 10000, 20000, 30000]
    safe = enumerate_safe_positions(grid, offsets, 3000)
    assert safe
    assert all(abs(p + off - g) >= 3000 for p in safe for off in offsets for g in grid)


class TestComputePaths:
    def _make_grid(self, rows):
        builder = ArchBuilder(grid_shape=(rows, 4), word_shape=(1, 2))
        for row in range(rows):
            for col in (0, 2):
                builder.add_word(rows=row, columns=[col, col + 1])
        builder.add_zone(
            "z",
            rows=[20.0 * row for row in range(rows)],
            columns=[0.0, 10.0, 20.0, 30.0],
            x_clearance=5.0,
            y_clearance=5.0,
        )
        return builder

    def _make_zone_with_site_bus(self):
        builder = self._make_grid(1)
        builder.add_site_bus("z", [0], [1])
        return builder

    def _make_zone_with_word_bus(self):
        builder = self._make_grid(2)
        builder.add_word_bus("z", [0, 1], [2, 3])
        return builder

    def test_site_bus_forward_path(self):
        builder = self._make_zone_with_site_bus()
        paths = builder.build().paths
        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((0.0, 0.0), (10.0, 0.0))

    def test_site_bus_backward_is_reversed(self):
        builder = self._make_zone_with_site_bus()
        paths = builder.build().paths
        fwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        bwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.BACKWARD, 0)
        assert paths[bwd] == paths[fwd][::-1]

    def test_site_bus_applies_to_all_words(self):
        builder = self._make_zone_with_site_bus()
        paths = builder.build().paths
        lane = LaneAddress(MoveType.SITE, 1, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((20.0, 0.0), (30.0, 0.0))

    def test_word_bus_forward_path(self):
        builder = self._make_zone_with_word_bus()
        paths = builder.build().paths
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((0.0, 0.0), (0.0, 20.0))

    def test_word_bus_backward_is_reversed(self):
        builder = self._make_zone_with_word_bus()
        paths = builder.build().paths
        fwd = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        bwd = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.BACKWARD, 0)
        assert paths[bwd] == paths[fwd][::-1]

    def test_word_bus_all_sites_get_paths(self):
        builder = self._make_zone_with_word_bus()
        paths = builder.build().paths
        for site_id in range(2):
            lane = LaneAddress(MoveType.WORD, 0, site_id, 0, Direction.FORWARD, 0)
            assert lane in paths


@pytest.mark.parametrize(
    ("columns", "sites", "radius", "error"),
    [
        ([0.0, 10.0, 2.0, 30.0], 2, 3.0, "partial blockade"),
        ([0.0, 100.0, 101.0, 5.0], 2, 2.0, "crossed-index"),
        ([0.0, 1.0, 2.0], 1, 1.5, "multiple blockade partners"),
        ([0.0, 1.0], 2, 1.5, "intra-word sites"),
    ],
)
def test_invalid_blockade_geometry(columns, sites, radius, error):
    builder = ArchBuilder(grid_shape=(1, len(columns)), word_shape=(1, sites))
    for col in range(0, len(columns), sites):
        builder.add_word(rows=0, columns=range(col, col + sites))
    builder.add_zone(
        "z", rows=[0.0], columns=columns, x_clearance=0.25, y_clearance=0.25
    )
    with pytest.raises(ValueError, match=error):
        builder.set_blockade_radius(radius)


def test_failed_blockade_scan_preserves_previous_pairs():
    builder = ArchBuilder(grid_shape=(1, 3), word_shape=(1, 1))
    for col in range(3):
        builder.add_word(rows=0, columns=col)
    builder.add_zone(
        "storage",
        rows=[0.0],
        columns=[0.0, 10.0, 20.0],
        x_clearance=0.25,
        y_clearance=0.25,
    )
    builder.add_zone(
        "gate",
        rows=[10.0],
        columns=[0.0, 1.0, 2.0],
        x_clearance=0.25,
        y_clearance=0.25,
    )
    builder.add_entangling_pairs("storage", [0], [1])
    with pytest.raises(ValueError, match="multiple blockade partners"):
        builder.set_blockade_radius(1.5)
    spec = builder.build()
    assert spec.blockade_radius is None
    assert list(spec.zones[0].entangling_pairs) == [(0, 1)]
