"""Tests for ZoneBuilder and ArchBuilder."""

import warnings

import pytest
from bloqade.lanes.bytecode._native import Grid
from bloqade.lanes.bytecode.encoding import MoveType, Direction, LaneAddress

from bloqade.gemini.arch.build.imperative import (
    ArchBuilder,
    ZoneBuilder,
    _SiteGridQuery,
    _validate_aod_rectangle,
)

# Default clearance used by most tests.  The unit-spaced test grids have
# spacings of 1.0, so 0.25 is comfortably below half-spacing.
_DEFAULT_CL = 0.25


def _make_grid(nx: int, ny: int, *, y_offset: float = 0.0) -> Grid:
    """Create a simple grid with unit spacing."""
    xs = [float(i) for i in range(nx)]
    ys = [y_offset + float(j) for j in range(ny)]
    return Grid.from_positions(xs, ys)


# ── _SiteGridQuery ──


class TestSiteGridQuery:
    def test_full_slice(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[:, :] == [0, 1, 2, 3, 4, 5]

    def test_single_row(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[:, 0] == [0, 1, 2]

    def test_second_row(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[:, 1] == [3, 4, 5]

    def test_x_slice(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[slice(0, 2), :] == [0, 1, 3, 4]

    def test_explicit_list(self):
        q = _SiteGridQuery(word_shape=(4, 2))
        assert q[[0, 2], 0] == [0, 2]

    def test_single_int(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[0, 0] == [0]


# ── AOD validation ──


class TestAODValidation:
    def test_valid_rectangle(self):
        _validate_aod_rectangle([(0, 0), (1, 0), (0, 1), (1, 1)], "test")

    def test_missing_corner_raises(self):
        with pytest.raises(ValueError, match="Cartesian product"):
            _validate_aod_rectangle([(0, 0), (1, 0), (0, 1)], "test")

    def test_single_point_valid(self):
        _validate_aod_rectangle([(0, 0)], "test")

    def test_empty_valid(self):
        _validate_aod_rectangle([], "test")

    def test_row_valid(self):
        _validate_aod_rectangle([(0, 0), (1, 0), (2, 0)], "test")

    def test_col_valid(self):
        _validate_aod_rectangle([(0, 0), (0, 1), (0, 2)], "test")


# ── ZoneBuilder: from_positions ──


class TestZoneBuilderFromPositions:
    def test_builds_equivalent_to_default_constructor(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0]
        zone = ZoneBuilder.from_positions(
            "gate",
            xs,
            ys,
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert isinstance(zone, ZoneBuilder)
        assert zone.name == "gate"
        assert zone.word_shape == (2, 1)
        assert zone.x_clearance == _DEFAULT_CL
        assert zone.y_clearance == _DEFAULT_CL
        assert list(zone._grid.x_positions) == xs
        assert list(zone._grid.y_positions) == ys

    def test_accepts_int_coordinates(self):
        zone = ZoneBuilder.from_positions(
            "gate",
            [0, 1, 2, 3],
            [0, 1],
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert list(zone._grid.x_positions) == [0.0, 1.0, 2.0, 3.0]
        assert list(zone._grid.y_positions) == [0.0, 1.0]

    def test_result_is_usable(self):
        zone = ZoneBuilder.from_positions(
            "gate",
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0],
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.add_word(x_sites=slice(0, 2), y_sites=slice(0, 1)) == 0

    def test_invalid_clearance_raises(self):
        with pytest.raises(ValueError, match="x_clearance must be positive"):
            ZoneBuilder.from_positions(
                "gate",
                [0.0, 1.0],
                [0.0],
                word_shape=(2, 1),
                x_clearance=0.0,
                y_clearance=_DEFAULT_CL,
            )


# ── ZoneBuilder: add_word ──


class TestZoneBuilderAddWord:
    def test_add_single_word(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        wid = zone.add_word(x_sites=slice(0, 2), y_sites=slice(0, 1))
        assert wid == 0

    def test_add_multiple_words(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.add_word(slice(0, 2), slice(0, 1)) == 0
        assert zone.add_word(slice(2, 4), slice(0, 1)) == 1

    def test_shape_mismatch_x_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        with pytest.raises(ValueError, match="word_shape requires 2"):
            zone.add_word(slice(0, 3), slice(0, 1))

    def test_shape_mismatch_y_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        with pytest.raises(ValueError, match="word_shape requires 1"):
            zone.add_word(slice(0, 2), slice(0, 2))

    def test_overlap_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 1))
        with pytest.raises(ValueError, match="already belongs to word 0"):
            zone.add_word(slice(1, 3), slice(0, 1))

    def test_out_of_bounds_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        with pytest.raises(IndexError):
            zone.add_word([3, 4], [0])

    def test_list_indices(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        wid = zone.add_word([0, 2], [0])
        assert wid == 0

    def test_num_words(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.num_words == 0
        zone.add_word(slice(0, 2), [0])
        assert zone.num_words == 1

    def test_sites_per_word(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 4),
            word_shape=(2, 3),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.sites_per_word == 6

    def test_has_site_bus_default_true(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        assert zone._word_has_site_bus == [True, True]

    def test_has_site_bus_explicit_false(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0], has_site_bus=False)
        zone.add_word(slice(2, 4), [0])
        assert zone._word_has_site_bus == [False, True]


# ── ZoneBuilder: grid queries ──


class TestZoneBuilderQueries:
    def test_words_query_all(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        assert zone.words[:, :] == [0, 1, 2, 3]

    def test_words_query_first_column(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        assert zone.words[slice(0, 2), :] == [0, 2]

    def test_zone_getitem_returns_name_qualified(self):
        zone = ZoneBuilder(
            "proc",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        name, ids = zone[:, :]
        assert name == "proc"
        assert ids == [0]

    def test_zone_getitem_matches_words_query(self):
        """zone[key] and zone.words[key] return compatible views."""
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        name, ids = zone[slice(0, 2), :]
        assert name == "gate"
        assert ids == zone.words[slice(0, 2), :]

    def test_sites_query(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(3, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.sites[:, 0] == [0, 1, 2]
        assert zone.sites[:, 1] == [3, 4, 5]
        assert zone.sites[:, :] == [0, 1, 2, 3, 4, 5]


# ── ZoneBuilder: buses ──


class TestZoneBuilderBuses:
    def test_add_site_bus_valid(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 2))
        zone.add_site_bus(src=[0, 1], dst=[2, 3])

    def test_add_site_bus_invalid_rectangle(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(3, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 3), slice(0, 2))
        # src=[0, 5] → positions (0,0) and (2,1) — diagonal, not a rectangle
        with pytest.raises(ValueError, match="Cartesian product"):
            zone.add_site_bus(src=[0, 5], dst=[1, 4])

    def test_add_site_bus_out_of_range(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 2))
        with pytest.raises(ValueError, match="out of range"):
            zone.add_site_bus(src=[0, 4], dst=[1, 5])

    def test_add_word_bus_valid(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])

    def test_add_word_bus_out_of_range(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        with pytest.raises(ValueError, match="out of range"):
            zone.add_word_bus(src=[0], dst=[1])

    def test_add_entangling_pairs(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_entangling_pairs([0], [1])

    def test_add_entangling_pairs_out_of_range(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        with pytest.raises(ValueError, match="out of range"):
            zone.add_entangling_pairs([0], [1])


# ── ArchBuilder ──


class TestArchBuilder:
    def test_single_zone(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        zid = arch.add_zone(zone)
        assert zid == 0
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert len(spec.zones) == 1
        assert len(spec.words) == 2

    def test_zone_name_preserved(self):
        zone = ZoneBuilder(
            "my_zone",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["my_zone"])
        spec = arch.build()
        assert spec.zones[0].name == "my_zone"

    def test_duplicate_zone_name_raises(self):
        z1 = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z1.add_word(slice(0, 2), [0])
        z2 = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z2.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(z1)
        with pytest.raises(ValueError, match="Duplicate zone name"):
            arch.add_zone(z2)

    def test_sites_per_word_mismatch_raises(self):
        z1 = ZoneBuilder(
            "a",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z1.add_word(slice(0, 2), [0])
        z2 = ZoneBuilder(
            "b",
            _make_grid(4, 1),
            word_shape=(4, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z2.add_word(slice(0, 4), [0])
        arch = ArchBuilder()
        arch.add_zone(z1)
        with pytest.raises(ValueError, match="sites_per_word"):
            arch.add_zone(z2)

    def test_multi_zone_with_connection(self):
        proc = ZoneBuilder(
            "proc",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        mem = ZoneBuilder(
            "mem",
            _make_grid(4, 2, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.connect(src=proc[:, :], dst=mem[:, :])
        arch.add_mode("all", ["proc", "mem"])
        spec = arch.build()
        assert len(spec.zones) == 2
        # Shared template: both zones reuse the same 2-word list.
        assert len(spec.words) == 2
        assert len(spec.zone_buses) == 1

    def test_unknown_zone_in_connect_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        with pytest.raises(ValueError, match="Unknown zone"):
            arch.connect(src=("gate", [0]), dst=("missing", [0]))

    def test_unknown_zone_in_mode_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        with pytest.raises(ValueError, match="Unknown zone"):
            arch.add_mode("test", ["missing"])

    def test_shared_template_word_ids_assigned(self):
        z1 = ZoneBuilder(
            "a",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z1.add_word(slice(0, 2), [0])
        z1.add_word(slice(2, 4), [0])
        z2 = ZoneBuilder(
            "b",
            _make_grid(4, 1, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z2.add_word(slice(0, 2), [0])
        z2.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        arch.add_zone(z1)
        arch.add_zone(z2)
        arch.add_mode("all", ["a", "b"])
        spec = arch.build()
        # Both zones share ONE 2-word template addressed as 0..1.
        assert len(spec.words) == 2

    def test_entangling_pairs_in_single_zone(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_entangling_pairs([0], [1])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec.zones[0].entangling_pairs == [(0, 1)]

    def test_words_with_site_buses_default_all(self):
        """Default has_site_bus=True → every word is in words_with_site_buses."""
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_site_bus(src=[0], dst=[1])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec.zones[0].words_with_site_buses == [0, 1]

    def test_words_with_site_buses_respects_opt_out(self):
        """has_site_bus=False excludes that word from words_with_site_buses."""
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0], has_site_bus=False)
        zone.add_site_bus(src=[0], dst=[1])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        # Only word 0 opted in.
        assert spec.zones[0].words_with_site_buses == [0]

    def test_rust_validation_passes(self):
        """Build a realistic arch and verify Rust validation."""
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus(src=[0], dst=[1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        zone.add_entangling_pairs([0, 2], [1, 3])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec is not None


class TestArchBuilderSharedTemplate:
    """Zones share ONE word template; word_id is zone-local (0..Nw-1).

    This is the LocationAddress convention the Rust model and docs assume:
    ``words`` is a single global template addressed by every zone as
    ``0..Nw-1``, disambiguated by ``zone_id``.  See
    ``docs/superpowers/plans/2026-07-29-word-id-shared-template-fix.md``.
    """

    @staticmethod
    def _bus_pairs(zone) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        return [(tuple(b.src), tuple(b.dst)) for b in zone.word_buses]

    def _make_two_zone_arch(self):
        proc = ZoneBuilder(
            "proc",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        proc.add_word(slice(0, 2), [1])
        proc.add_word(slice(2, 4), [1])
        proc.add_word_bus(src=[0, 1], dst=[2, 3])
        proc.add_site_bus(src=[0], dst=[1])
        proc.add_entangling_pairs([0, 2], [1, 3])

        mem = ZoneBuilder(
            "mem",
            _make_grid(4, 2, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        # Identical word template to proc (same slices / grid indices).
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(2, 4), [0])
        mem.add_word(slice(0, 2), [1])
        mem.add_word(slice(2, 4), [1])
        mem.add_word_bus(src=[0, 1], dst=[2, 3])
        mem.add_entangling_pairs([0, 2], [1, 3])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.connect(src=proc[:, :], dst=mem[:, :])
        arch.add_mode("all", ["proc", "mem"])
        return arch.build()

    def test_single_shared_word_template(self):
        """Two zones with an identical 4-word template → ONE 4-word list."""
        spec = self._make_two_zone_arch()
        assert len(spec.words) == 4
        assert len(spec.zones) == 2

    def test_word_buses_are_zone_local(self):
        spec = self._make_two_zone_arch()
        for zone in spec.zones:
            for bus in zone.word_buses:
                for w in list(bus.src) + list(bus.dst):
                    assert 0 <= w < 4, f"word bus id {w} must be zone-local (0..3)"

    def test_both_zones_word_buses_identical(self):
        """Shared template ⇒ every zone's word_buses reference the same IDs."""
        spec = self._make_two_zone_arch()
        assert self._bus_pairs(spec.zones[0]) == self._bus_pairs(spec.zones[1])

    def test_entangling_pairs_are_zone_local(self):
        spec = self._make_two_zone_arch()
        for zone in spec.zones:
            for a, b in zone.entangling_pairs:
                assert 0 <= a < 4
                assert 0 <= b < 4

    def test_zone_bus_word_ids_are_zone_local(self):
        spec = self._make_two_zone_arch()
        assert len(spec.zone_buses) == 1
        zb = spec.zone_buses[0]
        for zone_id, word_id in zb.src:
            assert zone_id == 0
            assert 0 <= word_id < 4
        for zone_id, word_id in zb.dst:
            assert zone_id == 1
            # zone-local under the shared template, NOT global 4..7
            assert 0 <= word_id < 4

    def test_words_with_site_buses_are_zone_local(self):
        spec = self._make_two_zone_arch()
        # proc has a site bus → all four template words opt in (zone-local).
        assert spec.zones[0].words_with_site_buses == [0, 1, 2, 3]
        # mem has no site bus → empty.
        assert spec.zones[1].words_with_site_buses == []

    def test_rust_validation_passes_multi_zone(self):
        spec = self._make_two_zone_arch()
        assert spec is not None
        assert len(spec.zones) == 2
        assert len(spec.words) == 4

    def test_mismatched_templates_raise(self):
        """Zones declaring different word templates are rejected at build()."""
        proc = ZoneBuilder(
            "proc",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])

        mem = ZoneBuilder(
            "mem",
            _make_grid(4, 2, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        # Different template: second word occupies a different grid region.
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(0, 2), [1])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.add_mode("all", ["proc", "mem"])
        with pytest.raises(ValueError, match="same word template"):
            arch.build()


class TestBuilderEdgeCases:
    def test_add_entangling_pairs_length_mismatch(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        with pytest.raises(ValueError, match="entries"):
            zone.add_entangling_pairs([0], [1, 0])

    def test_site_bus_src_dst_length_mismatch(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 2))
        with pytest.raises(ValueError, match="entries"):
            zone.add_site_bus(src=[0, 1], dst=[2])

    def test_word_bus_src_dst_length_mismatch(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        with pytest.raises(ValueError, match="entries"):
            zone.add_word_bus(src=[0, 1], dst=[2])

    def test_empty_word_query(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        # Query a region with no words
        assert zone.words[:, 1] == []
        # Name-qualified form still returns empty list
        assert zone[:, 1] == ("z", [])


# ── Path computation helpers ──


def _make_spaced_grid(
    nx: int, ny: int, *, x_spacing: float = 10.0, y_spacing: float = 20.0
) -> Grid:
    """Create a grid with explicit spacing (more realistic than unit grid)."""
    xs = [i * x_spacing for i in range(nx)]
    ys = [j * y_spacing for j in range(ny)]
    return Grid.from_positions(xs, ys)


# ── Safe-position enumeration ──


class TestEnumerateSafePositions:
    def test_single_source_valid_positions_clear(self):
        """Every returned position must be >= x_clearance (nm) from every grid line."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        min_cl_nm = zone._x_clearance_nm
        safe = zone._enumerate_safe_positions(zone._grid_x_nm, [0], min_cl_nm)
        assert len(safe) > 0
        for p in safe:
            for g in zone._grid_x_nm:
                assert abs(p - g) >= min_cl_nm

    def test_multiple_sources_preserve_spacing(self):
        """Multi-source: every shifted atom must clear every grid line."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        min_cl_nm = zone._x_clearance_nm
        safe = zone._enumerate_safe_positions(zone._grid_x_nm, [0, 2000], min_cl_nm)
        assert all(
            all(
                abs(p + off - g) >= min_cl_nm
                for g in zone._grid_x_nm
                for off in (0, 2000)
            )
            for p in safe
        )


# ── Search integration ──


class TestSearchPath:
    def test_straight_1d_move(self):
        """cd=1 shift in 1D row → straight line."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        # Coordinates in nm integers.
        path = zone._search_path((0, 0), (10000, 0), [(0, 0)])
        assert path == ((0, 0), (10000, 0))

    def test_cross_column_uses_clearance_row(self):
        """Cross-column move uses y-clearance (3-seg)."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        path = zone._search_path((0, 0), (20000, 0), [(0, 0)])
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (20000, 0)

    def test_diagonal_has_middle_waypoint(self):
        """Diagonal (col+row diff > 0) cannot be 2-segment with grid middle."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        path = zone._search_path((0, 0), (10000, 20000), [(0, 0)])
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (10000, 20000)


class TestAODCompatibility:
    """A bus is realizable when it is separable and order-preserving.

    An AOD sweeps whole x- and y-tones independently, so an atom may not
    leave its own row/column and tones may not cross.  Displacement need
    *not* be uniform: compressing or expanding a rectangle is one AOD
    operation.
    """

    def _row_zone(self, cols: int = 4) -> ZoneBuilder:
        """``cols`` words per row, two rows, one site per word."""
        grid = Grid.from_positions([float(i) for i in range(cols)], [0.0, 10.0])
        zone = ZoneBuilder("z", grid, (1, 1), x_clearance=0.25, y_clearance=3.0)
        for y in range(2):
            for x in range(cols):
                zone.add_word([x], [y])
        return zone

    def test_reversal_is_rejected(self):
        """The permutation bus: tones would have to cross."""
        zone = self._row_zone()
        with pytest.raises(ValueError, match="cannot .*cross|stay on its own"):
            zone.add_word_bus(src=[0, 1, 2, 3], dst=[7, 6, 5, 4])
        assert zone._word_buses == []

    def test_swapping_two_columns_is_rejected(self):
        zone = self._row_zone()
        with pytest.raises(ValueError, match="stay on its own"):
            zone.add_word_bus(src=[0, 1], dst=[5, 4])

    def test_shear_is_rejected(self):
        """Two atoms sharing a source column sent to different columns."""
        grid = Grid.from_positions([0.0, 1.0], [0.0, 1.0])
        zone = ZoneBuilder("z", grid, (1, 1), x_clearance=0.25, y_clearance=0.25)
        for y in range(2):
            for x in range(2):
                zone.add_word([x], [y])
        # words 0,1 (row 0) hold still; words 2,3 (row 1) swap columns.
        with pytest.raises(ValueError, match="stay on its own"):
            zone.add_word_bus(src=[0, 1, 2, 3], dst=[0, 1, 3, 2])

    def test_uniform_translation_is_accepted(self):
        zone = self._row_zone()
        zone.add_word_bus(src=[0, 1, 2, 3], dst=[4, 5, 6, 7])
        assert zone._word_buses == [([0, 1, 2, 3], [4, 5, 6, 7])]

    def test_non_uniform_but_separable_is_accepted(self):
        """Compressing a rectangle gives each column its own displacement."""
        # row 0 at x = 0, 10, 20; row 1 at x = 0, 5, 10 -> a compression.
        grid = Grid.from_positions([0.0, 5.0, 10.0, 20.0], [0.0, 30.0])
        zone = ZoneBuilder("z", grid, (1, 1), x_clearance=0.25, y_clearance=3.0)
        for x in (0, 2, 3):  # x = 0, 10, 20
            zone.add_word([x], [0])
        for x in (0, 1, 2):  # x = 0, 5, 10
            zone.add_word([x], [1])
        zone.add_word_bus(src=[0, 1, 2], dst=[3, 4, 5])
        assert zone._word_buses == [([0, 1, 2], [3, 4, 5])]

    def test_site_bus_with_mixed_pitch_is_accepted_then_declined_by_search(self):
        """Word 0's sites 10 µm apart, word 1's 30 µm apart.

        ``add_site_bus`` does not judge realizability — the atoms a site bus
        carries are ``words_with_site_buses x src``, and ``add_word`` can
        extend that set afterwards — so the bus is accepted here.  The path
        search still declines it, and its displacement check iterates every
        participating word, which is what surfaces the mismatch at all.
        """
        grid = Grid.from_positions([0.0, 10.0, 100.0, 130.0], [0.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=0.25, y_clearance=0.25)
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        zone.add_site_bus([0], [1])
        assert zone._site_buses == [([0], [1])]

        with pytest.warns(UserWarning, match="inconsistent site displacements"):
            paths = zone._compute_paths(zone_id=0)
        assert not any(k.move_type == MoveType.SITE for k in paths)

    def test_tone_count_must_match(self):
        """Two source columns collapsing onto one destination column.

        Endpoints are all distinct, so this reaches the tone-count check
        rather than the duplicate-endpoint one.
        """
        grid = Grid.from_positions([0.0, 1.0], [0.0, 10.0, 20.0])
        zone = ZoneBuilder("z", grid, (1, 1), x_clearance=0.25, y_clearance=3.0)
        zone.add_word([0], [0])  # word 0 at (0, 0)
        zone.add_word([1], [0])  # word 1 at (1, 0)
        zone.add_word([0], [1])  # word 2 at (0, 10)
        zone.add_word([0], [2])  # word 3 at (0, 20)
        # src spans 2 x-tones and 1 y-tone; dst spans 1 and 2.
        with pytest.raises(ValueError, match="add or drop a tone"):
            zone.add_word_bus(src=[0, 1], dst=[2, 3])

    def test_duplicate_endpoints_are_rejected(self):
        """Repeats survive every downstream check, which collapses to sets.

        A repeated source shadows later pairs under positional first-match
        resolution; a repeated destination asks for two transports into one
        place.  Rust catches both at ``build()``; fail at the call instead.
        """
        grid = Grid.from_positions([0.0, 1.0, 2.0, 3.0], [0.0, 10.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=0.25, y_clearance=3.0)
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        with pytest.raises(ValueError, match=r"src word \[0\] repeated"):
            zone.add_word_bus([0, 0], [1, 1])
        with pytest.raises(ValueError, match=r"dst word \[1\] repeated"):
            zone.add_word_bus([0, 1], [1, 1])
        assert zone._word_buses == []
        with pytest.raises(ValueError, match=r"src site \[0\] repeated"):
            zone.add_site_bus([0, 0], [1, 1])
        assert zone._site_buses == []

    def test_duplicate_site_indices_are_rejected(self):
        """A repeat passes the count check but collapses two sites onto one.

        The overlap check cannot catch it: it runs before any of the word's
        own positions is recorded, so the second copy does not yet look
        taken.
        """
        grid = Grid.from_positions([0.0, 1.0, 2.0, 3.0], [0.0, 10.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=0.25, y_clearance=3.0)
        with pytest.raises(ValueError, match=r"x_sites \[0\] repeated"):
            zone.add_word([0, 0], [0])
        assert zone.num_words == 0
        with pytest.raises(ValueError, match=r"y_sites \[1\] repeated"):
            ZoneBuilder("z", grid, (2, 2), x_clearance=0.25, y_clearance=3.0).add_word(
                [0, 1], [1, 1]
            )

    def test_word_opting_out_does_not_join(self):
        """Same layout, but the new word declines site-bus transport."""
        grid = Grid.from_positions([0.0, 1.0, 2.0, 3.0], [0.0, 10.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=0.25, y_clearance=3.0)
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        zone.add_site_bus([0], [1])
        zone.add_word([0, 1], [1], has_site_bus=False)
        assert zone.num_words == 3
        assert zone._site_bus_words() == [0, 1]

    def test_path_search_still_declines_a_non_uniform_bus(self):
        """Accepted by the hardware rule, but the search cannot route it.

        ``_compute_paths`` searches one reference path and applies its
        deltas to every lane, so it only handles uniform translations.  The
        bus is legal and simply lands unrouted.
        """
        grid = Grid.from_positions([0.0, 5.0, 10.0, 20.0], [0.0, 30.0])
        zone = ZoneBuilder("z", grid, (1, 1), x_clearance=0.25, y_clearance=3.0)
        for x in (0, 2, 3):
            zone.add_word([x], [0])
        for x in (0, 1, 2):
            zone.add_word([x], [1])
        zone.add_word_bus(src=[0, 1, 2], dst=[3, 4, 5])
        with pytest.warns(UserWarning, match="inconsistent word displacements"):
            paths = zone._compute_paths(zone_id=0)
        assert not any(k.move_type == MoveType.WORD for k in paths)


class TestSiteBusMoverConsistency:
    """The mover set must govern reference, obstacles and emitted lanes.

    Only words flagged ``has_site_bus`` are carried when a site bus fires.
    Using that set for some of the path computation and word 0 for the rest
    applies a delta from a word the bus does not move.
    """

    def _zone(self) -> ZoneBuilder:
        # word 0 opts out and has a 100 µm site pitch; words 1 and 2 opt in
        # with a 10 µm pitch.
        xs = [0.0, 100.0, 200.0, 210.0, 300.0, 310.0]
        zone = ZoneBuilder(
            "z",
            Grid.from_positions(xs, [0.0]),
            (2, 1),
            x_clearance=0.25,
            y_clearance=0.25,
        )
        zone.add_word([0, 1], [0], has_site_bus=False)
        zone.add_word([2, 3], [0])
        zone.add_word([4, 5], [0])
        zone.add_site_bus([0], [1])
        return zone

    def test_paths_land_on_their_destinations(self):
        zone = self._zone()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            paths = zone._compute_paths(zone_id=0)
        for lane, waypoints in paths.items():
            if lane.direction is Direction.FORWARD:
                dst_x, dst_y = zone._site_nm(lane.word_id, 1)
                assert waypoints[-1] == (dst_x / 1000, dst_y / 1000)

    def test_no_lane_for_a_word_that_opted_out(self):
        paths = self._zone()._compute_paths(zone_id=0)
        assert sorted({lane.word_id for lane in paths}) == [1, 2]

    def test_reference_follows_participation_not_word_zero(self):
        """Word 0 opts out and sets a pitch the movers do not share.

        The shipped physical spec has ``words_with_site_buses`` = the odd
        words, so word 0 opting out is production shape; it only fails to
        misbehave there because words 0 and 1 share a pitch.
        """
        grid = Grid.from_positions([0.0, 10.0, 100.0, 130.0, 200.0, 230.0], [0.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=0.5, y_clearance=0.5)
        zone.add_word([0, 1], [0], has_site_bus=False)  # pitch 10
        zone.add_word([2, 3], [0])  # pitch 30
        zone.add_word([4, 5], [0])  # pitch 30
        zone.add_site_bus([0], [1])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            paths = zone._compute_paths(zone_id=0)

        assert sorted({lane.word_id for lane in paths}) == [1, 2]
        for lane, waypoints in paths.items():
            if lane.direction is Direction.FORWARD:
                dst_x, dst_y = zone._site_nm(lane.word_id, 1)
                assert waypoints[-1] == (dst_x / 1000, dst_y / 1000)

    def test_empty_mover_set_is_skipped(self):
        """No word takes part, so there is no reference atom to route from."""
        grid = Grid.from_positions([0.0, 1.0, 2.0, 3.0], [0.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=0.25, y_clearance=0.25)
        zone.add_word([0, 1], [0], has_site_bus=False)
        zone.add_word([2, 3], [0], has_site_bus=False)
        zone.add_site_bus([0], [1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert zone._compute_paths(zone_id=0) == {}


class TestSearchFailureWarning:
    def test_no_valid_delta_warns_and_skips(self):
        """Very tight grid with high clearance → search fails."""
        # Dense grid: x=[0,1,2,3,4], y=[0,1,2,3,4], min_cl=2.5
        # Internal gaps half=0.5 < 2.5 → no safe internal positions.
        # Edges at ±2.5 → might work for a single source but not for multi-
        # word bus spanning multiple sources.
        grid = Grid.from_positions([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=2.5, y_clearance=2.5)
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        zone.add_word([0, 1], [2])
        zone.add_word([2, 3], [2])
        # Cross-column + cross-row bus (needs routing)
        zone.add_word_bus(src=[0], dst=[3])

        # Search may fail because no safe x (intra-pair sources span 0,1
        # with min_cl=2.5 → inter-offset shifts always collide).
        import warnings as _w

        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            paths = zone._compute_paths(zone_id=0)
            # Either warning fires (search failed) or paths found
            if any("no valid path" in str(warning.message) for warning in w):
                assert not any(k.move_type == MoveType.WORD for k in paths)


# ── _compute_paths (legacy + clearance) ──


class TestComputePaths:
    def _make_zone_with_site_bus(self):
        """Zone with 2 words on one row, one site bus shifting site 0→1."""
        grid = _make_spaced_grid(4, 1, x_spacing=10.0)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_site_bus(src=[0], dst=[1])
        return zone

    def _make_zone_with_word_bus(self):
        """Zone with 4 words (2×2 grid), one word bus moving row 0→1."""
        grid = _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        return zone

    def test_site_bus_forward_path(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=0)
        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((0.0, 0.0), (10.0, 0.0))

    def test_site_bus_backward_is_reversed(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=0)
        fwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        bwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.BACKWARD, 0)
        assert paths[bwd] == paths[fwd][::-1]

    def test_site_bus_applies_to_all_words(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=0)
        lane = LaneAddress(MoveType.SITE, 1, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((20.0, 0.0), (30.0, 0.0))

    def test_word_bus_forward_path(self):
        zone = self._make_zone_with_word_bus()
        paths = zone._compute_paths(zone_id=0)
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((0.0, 0.0), (0.0, 20.0))

    def test_word_bus_backward_is_reversed(self):
        zone = self._make_zone_with_word_bus()
        paths = zone._compute_paths(zone_id=0)
        fwd = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        bwd = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.BACKWARD, 0)
        assert paths[bwd] == paths[fwd][::-1]

    def test_word_bus_all_sites_get_paths(self):
        zone = self._make_zone_with_word_bus()
        paths = zone._compute_paths(zone_id=0)
        for site_id in range(2):
            lane = LaneAddress(MoveType.WORD, 0, site_id, 0, Direction.FORWARD, 0)
            assert lane in paths

    def test_paths_use_zone_local_word_ids(self):
        # Word IDs in lane addresses are zone-local (0..Nw-1); zone_id
        # disambiguates. No per-zone offset is applied.
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=1)
        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 1)
        assert lane in paths

    def test_no_buses_returns_empty(self):
        grid = _make_spaced_grid(2, 1)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        paths = zone._compute_paths(zone_id=0)
        assert paths == {}

    def test_cross_column_word_bus_uses_clearance(self):
        grid = _make_spaced_grid(4, 1, x_spacing=10.0)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word_bus(src=[0], dst=[1])
        paths = zone._compute_paths(zone_id=0)
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        path = paths[lane]
        # col_diff=2 > 1 → routing via y-clearance
        assert len(path) >= 2
        assert path[0] == (0.0, 0.0)
        assert path[-1] == (20.0, 0.0)

    def test_adjacent_word_bus_with_clearance_is_straight(self):
        """row_diff=1, col_diff=0 → straight line even with clearance."""
        grid = _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0)
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=5.0, y_clearance=5.0)
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        paths = zone._compute_paths(zone_id=0)
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        # col_diff=0, row_diff=1 → straight
        assert paths[lane] == ((0.0, 0.0), (0.0, 20.0))


class TestArchBuilderPaths:
    """Integration: ArchBuilder.build() produces paths on the ArchSpec."""

    def test_build_populates_paths(self):
        zone = ZoneBuilder(
            "gate",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus(src=[0], dst=[1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert len(spec.paths) > 0

    def test_get_path_returns_stored_path(self):
        zone = ZoneBuilder(
            "gate",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_site_bus(src=[0], dst=[1])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()

        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        path = spec.get_path(lane)
        assert path == ((0.0, 0.0), (10.0, 0.0))

    def test_multi_zone_paths_use_zone_local_ids(self):
        proc = ZoneBuilder(
            "proc",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        proc.add_site_bus(src=[0], dst=[1])

        mem = ZoneBuilder(
            "mem",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(2, 4), [0])
        mem.add_site_bus(src=[0], dst=[1])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.connect(src=proc[:, :], dst=mem[:, :])
        arch.add_mode("all", ["proc", "mem"])
        spec = arch.build()

        # Both zones' site-bus lanes address zone-local word 0; only the
        # zone_id differs (shared template).
        lane_proc = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        assert lane_proc in spec.paths
        lane_mem = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 1)
        assert lane_mem in spec.paths

    def test_no_buses_no_paths(self):
        zone = ZoneBuilder(
            "gate",
            _make_spaced_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        zone.add_word(slice(0, 2), [0])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert len(spec.paths) == 0


# ── ZoneBuilder: blockade-radius inference ──


def _gate_zone_grid() -> Grid:
    """Gate-zone style grid: 3 CZ pairs on one row, intra-pair 2 µm, inter-pair 8 µm."""
    # x=[0, 2, 10, 12, 20, 22], y=[0]
    xs = [0.0, 2.0, 10.0, 12.0, 20.0, 22.0]
    ys = [0.0]
    return Grid.from_positions(xs, ys)


def _make_gate_zone() -> ZoneBuilder:
    """Zone with 6 interleaved CZ-pair words (word_shape (1, 1))."""
    zone = ZoneBuilder(
        "gate",
        _gate_zone_grid(),
        word_shape=(1, 1),
        x_clearance=_DEFAULT_CL,
        y_clearance=_DEFAULT_CL,
    )
    # 6 words, one per grid x-position.
    for x in range(6):
        zone.add_word([x], [0])
    return zone


class TestZoneBuilderBlockadeRadius:
    def test_derives_interleaved_cz_pairs(self):
        zone = _make_gate_zone()
        zone.set_blockade_radius(2.0)
        # Pairs: (0,1) at dist 2, (2,3) at 2, (4,5) at 2.
        # Inter-pair dist = 8, outside radius.
        assert zone._entangling_pairs == [(0, 1), (2, 3), (4, 5)]

    def test_blockade_radius_property(self):
        zone = _make_gate_zone()
        assert zone.blockade_radius is None
        zone.set_blockade_radius(2.0)
        assert zone.blockade_radius == 2.0

    def test_overwrites_explicit_pairs(self):
        zone = _make_gate_zone()
        # Bogus pre-existing pair — should be replaced by the scan.
        zone.add_entangling_pairs([0], [3])
        zone.set_blockade_radius(2.0)
        assert zone._entangling_pairs == [(0, 1), (2, 3), (4, 5)]

    def test_no_pairs_when_radius_too_tight(self):
        zone = _make_gate_zone()
        zone.set_blockade_radius(1.0)  # all gaps > 1 µm
        assert zone._entangling_pairs == []
        assert zone.blockade_radius == 1.0

    def test_partial_blockade_raises(self):
        """2-site words where only one matching pair is within radius."""
        # Grid: x=[0, 10, 2, 12] giving two words with sites at x-indices
        # (0, 1) and (2, 3) → positions [(0, 10)] and [(2, 12)]. Same-index
        # distances: |0-2|=2, |10-12|=2 → all within radius=3 → clean pair.
        # For the partial case, make word B's second site far:
        # x=[0, 10, 2, 30] → B = [2, 30]. Same-index dist: 2, 20.
        grid = Grid.from_positions([0.0, 10.0, 2.0, 30.0], [0.0])
        zone = ZoneBuilder(
            "z",
            grid,
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        with pytest.raises(ValueError, match="partial blockade"):
            zone.set_blockade_radius(3.0)

    def test_crossed_index_raises(self):
        """Two words where site 0 of A is near site 1 of B (not site 0)."""
        # Grid x=[0, 100, 101, 5]. Word A at indices [0, 1] → sites (0, 100).
        # Word B at indices [2, 3] → sites (101, 5).
        # Matching: |0-101|=101, |100-5|=95 → both outside radius 2.
        # Crossed: |0-5|=5 → outside, |100-101|=1 → INSIDE.
        # So site_a[0] paired with site_b[1] — crossed index.
        grid = Grid.from_positions([0.0, 100.0, 101.0, 5.0], [0.0])
        zone = ZoneBuilder(
            "z",
            grid,
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        with pytest.raises(ValueError, match="crossed-index"):
            zone.set_blockade_radius(2.0)

    def test_multi_partner_raises(self):
        """Three words all mutually within radius — ambiguous pairing."""
        # 3 words at x=[0, 1, 2], single site each. Radius 1.5: all within.
        grid = Grid.from_positions([0.0, 1.0, 2.0], [0.0])
        zone = ZoneBuilder(
            "z",
            grid,
            word_shape=(1, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        for x in range(3):
            zone.add_word([x], [0])
        with pytest.raises(ValueError, match="multiple blockade partners"):
            zone.set_blockade_radius(1.5)

    def test_intra_word_blockade_raises(self):
        """Two sites of the same word within radius — invalid layout."""
        # Word with 2 sites at x=0 and x=1; radius 1.5 puts them within
        # blockade of each other. That would entangle atoms inside the
        # same word, which isn't allowed.
        grid = Grid.from_positions([0.0, 1.0], [0.0])
        zone = ZoneBuilder(
            "z",
            grid,
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word([0, 1], [0])
        with pytest.raises(ValueError, match="intra-word sites"):
            zone.set_blockade_radius(1.5)

    def test_non_positive_radius_raises(self):
        zone = _make_gate_zone()
        with pytest.raises(ValueError, match="must be positive"):
            zone.set_blockade_radius(0.0)
        with pytest.raises(ValueError, match="must be positive"):
            zone.set_blockade_radius(-1.0)

    def test_non_nm_precise_radius_raises(self):
        zone = _make_gate_zone()
        with pytest.raises(ValueError, match="1 nm precision"):
            zone.set_blockade_radius(1.2345)  # sub-nm


class TestArchBuilderBlockadeRadius:
    def test_applies_to_all_zones(self):
        gate_a = _make_gate_zone()
        gate_b = _make_gate_zone()
        # Second zone needs a distinct name to avoid clashes
        gate_b._name = "gate_b"
        arch = ArchBuilder()
        arch.add_zone(gate_a)
        arch.add_zone(gate_b)
        arch.set_blockade_radius(2.0)
        assert gate_a._entangling_pairs == [(0, 1), (2, 3), (4, 5)]
        assert gate_b._entangling_pairs == [(0, 1), (2, 3), (4, 5)]
        assert arch.blockade_radius == 2.0

    def test_arch_spec_exposes_blockade_radius(self):
        zone = _make_gate_zone()
        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        arch.set_blockade_radius(2.0)
        spec = arch.build()
        assert spec.blockade_radius == 2.0

    def test_build_without_blockade_radius_is_none(self):
        zone = _make_gate_zone()
        zone.add_entangling_pairs([0, 2, 4], [1, 3, 5])
        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec.blockade_radius is None

    def test_set_before_add_zone_has_no_effect(self):
        """set_blockade_radius operates on currently-added zones; zones added
        later are not automatically scanned."""
        arch = ArchBuilder()
        arch.set_blockade_radius(2.0)  # no zones yet
        zone = _make_gate_zone()
        arch.add_zone(zone)
        # The radius is stored; zones added afterward were not scanned.
        assert zone._entangling_pairs == []
        # Caller can re-set to apply:
        arch.set_blockade_radius(2.0)
        assert zone._entangling_pairs == [(0, 1), (2, 3), (4, 5)]

    def test_nan_radius_raises(self):
        """NaN / Inf on ArchBuilder.set_blockade_radius are rejected up-front."""
        import math

        arch = ArchBuilder()  # no zones: validation must still fire
        with pytest.raises(ValueError, match="must be positive"):
            arch.set_blockade_radius(0.0)
        with pytest.raises(ValueError, match="must be finite"):
            arch.set_blockade_radius(math.nan)
        with pytest.raises(ValueError, match="must be finite"):
            arch.set_blockade_radius(math.inf)

    def test_rollback_on_partial_failure(self):
        """If a later zone's scan fails, earlier zones must stay untouched."""
        good = _make_gate_zone()
        # Give `good` an existing explicit pairing so we can see whether
        # the failed set_blockade_radius call overwrote it.
        good.add_entangling_pairs([0, 2, 4], [1, 3, 5])
        # A zone whose layout fails the scan under radius=1.5.
        bad = ZoneBuilder(
            "bad",
            Grid.from_positions([0.0, 1.0, 2.0], [0.0]),
            (1, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        for x in range(3):
            bad.add_word([x], [0])
        # 3 mutually-within-radius words → multi-partner ValueError on bad.
        arch = ArchBuilder()
        arch.add_zone(good)
        arch.add_zone(bad)
        with pytest.raises(ValueError, match="multiple blockade partners"):
            arch.set_blockade_radius(1.5)
        # `good` must still have its original pairs.
        assert good._entangling_pairs == [(0, 1), (2, 3), (4, 5)]
        # And no radius recorded on the builder (the call failed).
        assert arch.blockade_radius is None

    def test_zone_radius_flows_to_spec(self):
        """A radius set only on a ZoneBuilder propagates to ArchSpec."""
        zone = _make_gate_zone()
        zone.set_blockade_radius(2.0)
        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec.blockade_radius == 2.0

    def test_inconsistent_zone_radii_raises(self):
        """Two zones with different radii cannot be composed into one spec."""
        z1 = _make_gate_zone()
        z1.set_blockade_radius(2.0)
        z2 = _make_gate_zone()
        z2._name = "gate_b"
        z2.set_blockade_radius(3.0)
        arch = ArchBuilder()
        arch.add_zone(z1)
        arch.add_zone(z2)
        arch.add_mode("all", ["gate", "gate_b"])
        with pytest.raises(ValueError, match="disagree on blockade_radius"):
            arch.build()

    def test_mixed_zone_radii_raises(self):
        """If some zones have a radius and others don't, ArchBuilder.build fails."""
        z1 = _make_gate_zone()
        z1.set_blockade_radius(2.0)
        z2 = _make_gate_zone()
        z2._name = "gate_b"
        # z2 has no radius.
        arch = ArchBuilder()
        arch.add_zone(z1)
        arch.add_zone(z2)
        arch.add_mode("all", ["gate", "gate_b"])
        with pytest.raises(ValueError, match="some zones but not others"):
            arch.build()


class TestZoneBuilderBlockadeMetadata:
    def test_add_entangling_pairs_clears_radius(self):
        """Manual append after set_blockade_radius clears the cached radius."""
        zone = _make_gate_zone()
        zone.set_blockade_radius(2.0)
        assert zone.blockade_radius == 2.0
        zone.add_entangling_pairs([0], [5])  # manual override
        assert zone.blockade_radius is None


# ── Grid query traversal order ──


def _interleaved_zone(rows: int = 2, cols: int = 4) -> ZoneBuilder:
    """A zone whose word IDs are not monotone in the grid.

    Mirrors the shipped physical architecture: each word's sites are
    interleaved with its neighbours' along x, so scanning x visits words
    0, 1, ..., cols-1, 0, 1, ... and a partial x-selection reaches them
    out of ID order.
    """
    xs = [float(i) for i in range(cols * 2)]
    ys = [float(10 * j) for j in range(rows)]
    zone = ZoneBuilder(
        "z",
        Grid.from_positions(xs, ys),
        word_shape=(2, 1),
        x_clearance=_DEFAULT_CL,
        y_clearance=3.0,
    )
    for row in range(rows):
        for col in range(cols):
            zone.add_word([col, col + cols], [row])
    return zone


class TestWordQueryTraversalOrder:
    def test_partial_selection_follows_the_grid_not_word_ids(self):
        zone = _interleaved_zone()
        # x index 3 is a site of word 3; x index 4 is a site of word 0.
        assert zone.words[[3, 4], [0, 1]] == [3, 0, 7, 4]

    def test_rows_are_the_outer_loop(self):
        assert _interleaved_zone().words[:, :] == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_axis_order_is_the_callers(self):
        """Like ``arr[np.ix_(ys, xs)].ravel()``: nesting is row-major, the
        order within each axis is the caller's."""
        zone = _interleaved_zone()
        assert zone.words[[3, 4], [0, 1]] == [3, 0, 7, 4]
        assert zone.words[[4, 3], [0, 1]] == [0, 3, 4, 7]
        assert zone.words[[3, 4], [1, 0]] == [7, 4, 3, 0]
        assert zone.words[[4, 3], [1, 0]] == [4, 7, 0, 3]

    def test_reverse_slice_expands_descending(self):
        zone = _interleaved_zone()
        assert zone.words[::-1, 0] == [3, 2, 1, 0]
        assert zone.words[3::-1, 0] == [3, 2, 1, 0]

    def test_repeated_index_raises(self):
        zone = _interleaved_zone()
        with pytest.raises(ValueError, match=r"x grid index \[0\] repeated"):
            zone.words[[0, 0, 1], 0]
        with pytest.raises(ValueError, match=r"y grid index \[1\] repeated"):
            zone.words[0, [1, 1]]

    def test_word_reached_twice_is_appended_once(self):
        zone = _interleaved_zone()
        # Both x=0 and x=4 are sites of word 0.
        assert zone.words[[0, 4], 0] == [0]

    def test_two_dimensional_words(self):
        """A word spanning a block is appended at its first arrival."""
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 4),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word([0, 1], [0, 1])
        zone.add_word([2, 3], [0, 1])
        zone.add_word([0, 1], [2, 3])
        zone.add_word([2, 3], [2, 3])
        assert zone.words[:, :] == [0, 1, 2, 3]

    def test_staggered_layout_needs_no_grid_of_words(self):
        zone = ZoneBuilder(
            "z",
            Grid.from_positions([float(i) for i in range(12)], [0.0, 10.0]),
            word_shape=(1, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=3.0,
        )
        for x in (0, 4, 8):
            zone.add_word([x], [0])
        for x in (2, 6, 10):
            zone.add_word([x], [1])
        assert zone.words[:, :] == [0, 1, 2, 3, 4, 5]
        assert zone.words[:, 0] == [0, 1, 2]
        assert zone.words[:, 1] == [3, 4, 5]

    def test_selected_endpoints_pair_into_a_routable_bus(self):
        """The property the traversal order exists to guarantee.

        ``src`` and ``dst`` are related by a uniform AOD translation, and a
        translation preserves traversal order, so selecting both endpoints
        with this query pairs ``src[i]`` with ``dst[i]`` correctly and the
        bus satisfies the single-shift invariant.
        """
        zone = _interleaved_zone()
        src, dst = zone.words[:, 0], zone.words[:, 1]
        assert (src, dst) == ([0, 1, 2, 3], [4, 5, 6, 7])
        zone.add_word_bus(src, dst)
        builder = ArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spec = builder.build()
        assert len(spec.paths) == 16

    def test_empty_region(self):
        zone = _interleaved_zone(rows=1)
        assert zone.words[:, 0] == [0, 1, 2, 3]


class TestGridQueryIndexValidation:
    def test_out_of_range_raises(self):
        zone = _interleaved_zone()
        with pytest.raises(IndexError, match=r"x grid index \[99\]"):
            zone.words[[0, 99], 0]
        with pytest.raises(IndexError, match=r"y grid index \[9\]"):
            zone.words[0, [9]]

    def test_negative_raises(self):
        zone = _interleaved_zone()
        with pytest.raises(IndexError, match=r"x grid index \[-1\]"):
            zone.words[[-1], 0]

    def test_site_query_validates_too(self):
        zone = _interleaved_zone()
        with pytest.raises(IndexError, match=r"x site index \[9\]"):
            zone.sites[[9], 0]

    def test_slices_are_clamped_not_rejected(self):
        assert _interleaved_zone().words[0:99, 0] == [0, 1, 2, 3]

    def test_strided_slice_is_allowed(self):
        assert _interleaved_zone().words[0:8:2, 0] == [0, 2]

    def test_name_qualified_form_validates(self):
        zone = _interleaved_zone()
        with pytest.raises(IndexError, match=r"x grid index \[99\]"):
            zone[[99], 0]


# ── Empty buses and zone-bus index range ──


class TestEmptyBusRejected:
    """An empty bus carries nothing and breaks the path search.

    ``_compute_paths`` indexes ``src[0]`` for its reference atom, so an
    empty bus reaches ``build()`` as a bare ``IndexError`` naming neither
    the bus nor the zone.  Reachable from ordinary use: ``zone.words[0:0, 0]``
    legitimately returns ``[]``.
    """

    def _zone(self) -> ZoneBuilder:
        zone = ZoneBuilder(
            "z", _make_grid(4, 1), (2, 1), x_clearance=0.25, y_clearance=0.25
        )
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        return zone

    def test_empty_word_bus_raises(self):
        with pytest.raises(ValueError, match="Word bus on zone 'z' is empty"):
            self._zone().add_word_bus([], [])

    def test_empty_site_bus_raises(self):
        with pytest.raises(ValueError, match="Site bus on zone 'z' is empty"):
            self._zone().add_site_bus([], [])

    def test_empty_slice_selection_is_the_reachable_path(self):
        zone = self._zone()
        assert zone.words[0:0, 0] == []
        with pytest.raises(ValueError, match="is empty"):
            zone.add_word_bus(zone.words[0:0, 0], zone.words[0:0, 0])


class TestConnectIndexRange:
    """``connect`` resolved word indices without a range check.

    ``_word_origin`` indexes ``_words`` directly, so ``-1`` silently
    resolved to the last word: the Cartesian-product check then validated
    geometry the caller never named, and the bad index failed later with an
    unrelated error.  ``add_word_bus`` rejects the same index.
    """

    def _builder(self) -> ArchBuilder:
        builder = ArchBuilder()
        for name, y_off in (("a", 0.0), ("b", 10.0)):
            zone = ZoneBuilder(
                name,
                _make_grid(4, 1, y_offset=y_off),
                (2, 1),
                x_clearance=0.25,
                y_clearance=0.25,
            )
            zone.add_word([0, 1], [0])
            zone.add_word([2, 3], [0])
            builder.add_zone(zone)
        builder.add_mode("all", ["a", "b"])
        return builder

    def test_negative_index_raises(self):
        builder = self._builder()
        with pytest.raises(ValueError, match=r"src word index -1 out of range"):
            builder.connect(src=("a", [-1]), dst=("b", [0]))
        assert builder._connections == []

    def test_too_large_index_raises(self):
        builder = self._builder()
        with pytest.raises(ValueError, match=r"dst word index 9 out of range"):
            builder.connect(src=("a", [0]), dst=("b", [9]))

    def test_valid_indices_still_connect(self):
        builder = self._builder()
        builder.connect(src=("a", [0, 1]), dst=("b", [0, 1]))
        assert len(builder._connections) == 1
