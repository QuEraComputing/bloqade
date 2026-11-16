"""Tests for the spec-shaped ArchBuilder in ``arch.build.v2``."""

import json
import warnings
from dataclasses import dataclass

import pytest
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import ArchSpec as RustArchSpec
from bloqade.lanes.bytecode.encoding import MoveType
from bloqade.lanes.arch.gemini.logical.spec import get_arch_spec as logical_spec
from bloqade.lanes.arch.gemini.physical.spec import get_arch_spec as physical_spec

from bloqade.gemini.arch.build.v2 import ArchBuilder
from bloqade.gemini.arch.build.imperative import (
    ArchBuilder as LegacyArchBuilder,
    ZoneBuilder,
)


def as_comparable(spec: ArchSpec) -> dict:
    """The whole spec as plain data, with paths in a stable order.

    Comparing field by field only tests the fields someone thought to
    list; serializing compares everything the spec actually carries.
    """
    data = json.loads(spec.to_json())
    if data.get("paths"):
        data["paths"] = sorted(data["paths"], key=json.dumps)
    return data


def perturbed(spec: ArchSpec, mutate) -> ArchSpec:
    """A copy of *spec* with ``mutate`` applied to its JSON form."""
    data = json.loads(spec.to_json())
    mutate(data)
    return ArchSpec(RustArchSpec.from_json_validated(json.dumps(data)))


_CL = 0.25


def interleaved(rows: int = 2, cols: int = 4) -> ArchBuilder:
    """A template whose word IDs are not monotone in the grid.

    Mirrors the shipped physical architecture: each word's sites are
    interleaved with its neighbours' along x, so scanning x visits words
    0, 1, ..., cols-1, 0, 1, ... and a partial x-selection reaches them out
    of ID order.
    """
    b = ArchBuilder(grid_shape=(rows, cols * 2), word_shape=(1, 2))
    for row in range(rows):
        for col in range(cols):
            b.add_word(rows=[row], columns=[col, col + cols])
    return b


def with_zone(b: ArchBuilder, name: str = "z", **kwargs) -> ArchBuilder:
    """Attach a unit-spaced zone covering the builder's index space."""
    n_rows, n_cols = b.grid_shape
    b.add_zone(
        name,
        rows=[10.0 * j for j in range(n_rows)],
        columns=[float(i) for i in range(n_cols)],
        x_clearance=_CL,
        y_clearance=3.0,
        **kwargs,
    )
    return b


# ── The word template is spec-wide ──


class TestTemplate:
    def test_words_are_frozen_by_the_first_zone(self):
        b = with_zone(interleaved())
        with pytest.raises(ValueError, match="word template is frozen"):
            b.add_word(rows=[0], columns=[0, 1])

    def test_zone_needs_a_template_first(self):
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 2))
        with pytest.raises(ValueError, match="add_word before adding a zone"):
            with_zone(b)

    def test_overlapping_words_are_rejected(self):
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 2))
        b.add_word(rows=[0], columns=[0, 1])
        with pytest.raises(ValueError, match="already belongs to word 0"):
            b.add_word(rows=[0], columns=[1, 2])

    def test_word_shape_is_enforced(self):
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 2))
        with pytest.raises(ValueError, match="word_shape requires 2"):
            b.add_word(rows=[0], columns=[0])

    def test_every_zone_indexes_the_same_space(self):
        b = interleaved()
        with pytest.raises(ValueError, match="grid_shape is 2x8"):
            b.add_zone(
                "z",
                rows=[0.0],
                columns=[0.0, 1.0],
                x_clearance=_CL,
                y_clearance=_CL,
            )


class TestTemplateQueries:
    def test_partial_selection_follows_the_grid_not_word_ids(self):
        # x index 3 is a site of word 3; x index 4 is a site of word 0.
        assert interleaved().words[[0, 1], [3, 4]] == [3, 0, 7, 4]

    def test_axis_order_is_the_callers(self):
        b = interleaved()
        assert b.words[[0, 1], [4, 3]] == [0, 3, 4, 7]
        assert b.words[[1, 0], [3, 4]] == [7, 4, 3, 0]

    def test_rows_are_the_outer_loop(self):
        assert interleaved().words[:, :] == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_word_reached_twice_is_appended_once(self):
        # Both column 0 and column 4 are sites of word 0.
        assert interleaved().words[0, [0, 4]] == [0]

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError, match=r"column grid index \[99\]"):
            interleaved().words[0, [0, 99]]

    def test_negative_raises(self):
        with pytest.raises(IndexError, match=r"column grid index \[-1\]"):
            interleaved().words[0, [-1]]

    def test_repeated_index_raises(self):
        with pytest.raises(ValueError, match=r"column grid index \[0\] repeated"):
            interleaved().words[0, [0, 0, 1]]

    def test_slices_are_clamped_not_rejected(self):
        assert interleaved().words[0, 0:99] == [0, 1, 2, 3]

    def test_reverse_slice_expands_descending(self):
        assert interleaved().words[0, ::-1] == [3, 2, 1, 0]

    def test_site_query_validates_too(self):
        with pytest.raises(IndexError, match=r"column site index \[9\]"):
            interleaved().sites[0, [9]]


# ── Bus realizability ──


class TestBusRealizability:
    """A bus is realizable when it is separable and order-preserving.

    An AOD sweeps whole x- and y-tones independently, so an atom may not
    leave its own row/column and tones may not cross.  Displacement need
    *not* be uniform: compressing or expanding a rectangle is one AOD
    operation.
    """

    def _rows(self, cols: int = 4) -> ArchBuilder:
        b = ArchBuilder(grid_shape=(2, cols), word_shape=(1, 1))
        for y in range(2):
            for x in range(cols):
                b.add_word(rows=[y], columns=[x])
        return with_zone(b)

    def test_uniform_translation_is_accepted(self):
        b = self._rows()
        b.add_word_bus("z", src=[0, 1, 2, 3], dst=[4, 5, 6, 7])

    def test_reversal_is_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="stay on its own"):
            b.add_word_bus("z", src=[0, 1, 2, 3], dst=[7, 6, 5, 4])

    def test_swapping_two_columns_is_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="stay on its own"):
            b.add_word_bus("z", src=[0, 1], dst=[5, 4])

    def test_shear_is_rejected(self):
        b = ArchBuilder(grid_shape=(2, 2), word_shape=(1, 1))
        for y in range(2):
            for x in range(2):
                b.add_word(rows=[y], columns=[x])
        with_zone(b)
        # row 0 holds still; row 1 swaps columns.
        with pytest.raises(ValueError, match="stay on its own"):
            b.add_word_bus("z", src=[0, 1, 2, 3], dst=[0, 1, 3, 2])

    def test_tone_count_must_match(self):
        b = self._rows(cols=3)
        with pytest.raises(ValueError, match="add or drop a tone"):
            b.add_word_bus("z", src=[0, 1], dst=[3, 3])

    def test_non_uniform_but_separable_is_accepted(self):
        """Compressing a rectangle gives each column its own displacement."""
        b = ArchBuilder(grid_shape=(2, 4), word_shape=(1, 1))
        for x in (0, 2, 3):  # x = 0, 10, 20
            b.add_word(rows=[0], columns=[x])
        for x in (0, 1, 2):  # x = 0, 5, 10
            b.add_word(rows=[1], columns=[x])
        b.add_zone(
            "z",
            rows=[0.0, 30.0],
            columns=[0.0, 5.0, 10.0, 20.0],
            x_clearance=_CL,
            y_clearance=3.0,
        )
        b.add_word_bus("z", src=[0, 1, 2], dst=[3, 4, 5])

    def test_empty_bus_is_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="at least one word"):
            b.add_word_bus("z", src=[], dst=[])
        with pytest.raises(ValueError, match="at least one site"):
            b.add_site_bus("z", src=[], dst=[])

    def test_mismatched_lengths_are_rejected(self):
        b = self._rows()
        with pytest.raises(ValueError, match="src has 2 entries but dst has 1"):
            b.add_word_bus("z", src=[0, 1], dst=[4])

    def test_out_of_range_word_is_rejected(self):
        b = self._rows()
        with pytest.raises(IndexError, match=r"out of range \[0, 8\)"):
            b.add_word_bus("z", src=[0], dst=[99])

    def test_negative_word_index_is_rejected(self):
        """A negative index would otherwise wrap to a different word."""
        b = self._rows()
        with pytest.raises(IndexError, match=r"\[-1\] out of range"):
            b.add_word_bus("z", src=[0], dst=[-1])


class TestStowaways:
    """The AOD traps at every intersection of its tones.

    An intersection occupied by an atom the bus does not carry would be
    picked up too; an intersection no atom occupies is harmless.
    """

    def _l_shape(self, **zone_kwargs) -> ArchBuilder:
        b = ArchBuilder(grid_shape=(2, 4), word_shape=(1, 2))
        b.add_word(rows=[0], columns=[0, 1])
        b.add_word(rows=[0], columns=[2, 3])
        b.add_word(rows=[1], columns=[0, 1])
        return with_zone(b, **zone_kwargs)

    def test_empty_intersection_is_allowed(self):
        """No word sits at the open crossing, so nothing rides along."""
        b = self._l_shape()
        b.add_site_bus("z", src=[0], dst=[1])

    def test_occupied_intersection_is_rejected(self):
        """A fourth word fills the crossing but sits out the bus."""
        b = ArchBuilder(grid_shape=(2, 4), word_shape=(1, 2))
        for y in range(2):
            for x0 in (0, 2):
                b.add_word(rows=[y], columns=[x0, x0 + 1])
        with_zone(b, words_with_site_buses=[0, 1, 2])  # word 3 opts out
        with pytest.raises(ValueError, match="would be carried along"):
            b.add_site_bus("z", src=[0], dst=[1])

    def test_including_the_intruder_makes_it_legal(self):
        b = ArchBuilder(grid_shape=(2, 4), word_shape=(1, 2))
        for y in range(2):
            for x0 in (0, 2):
                b.add_word(rows=[y], columns=[x0, x0 + 1])
        with_zone(b)  # every word participates
        b.add_site_bus("z", src=[0], dst=[1])


class TestParticipation:
    def test_subset_reaches_the_spec(self):
        b = with_zone(interleaved(), words_with_site_buses=[1, 3])
        b.add_site_bus("z", src=[0], dst=[1])
        b.add_mode("all", ["z"])
        spec = b.build()
        assert list(spec.zones[0].words_with_site_buses) == [1, 3]

    def test_sites_with_word_buses_is_not_forced_to_every_site(self):
        """The legacy builder could not express a proper subset here."""
        b = ArchBuilder(grid_shape=(2, 4), word_shape=(1, 2))
        for y in range(2):
            for x0 in (0, 2):
                b.add_word(rows=[y], columns=[x0, x0 + 1])
        with_zone(b, sites_with_word_buses=[0])
        b.add_word_bus("z", src=[0, 1], dst=[2, 3])
        b.add_mode("all", ["z"])
        assert list(b.build().zones[0].sites_with_word_buses) == [0]

    def test_out_of_range_participant_is_rejected(self):
        with pytest.raises(IndexError, match="words_with_site_buses"):
            with_zone(interleaved(), words_with_site_buses=[99])

    def test_repeated_participant_is_rejected(self):
        with pytest.raises(ValueError, match="sites_with_word_buses"):
            with_zone(interleaved(), sites_with_word_buses=[0, 0])


# ── Assembly ──


class TestBuild:
    def test_selected_endpoints_pair_into_a_routable_bus(self):
        b = interleaved()
        with_zone(b)
        src, dst = b.words[0, :], b.words[1, :]
        assert (src, dst) == ([0, 1, 2, 3], [4, 5, 6, 7])
        b.add_word_bus("z", src, dst)
        b.add_mode("all", ["z"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spec = b.build()
        assert len(spec.paths) == 16

    def test_build_requires_words_and_zones(self):
        b = ArchBuilder(grid_shape=(1, 2), word_shape=(1, 1))
        with pytest.raises(ValueError, match="no words defined"):
            b.build()
        b.add_word(rows=[0], columns=[0])
        with pytest.raises(ValueError, match="no zones defined"):
            b.build()

    def test_blockade_radius_derives_pairs_per_zone(self):
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 1))
        for x in range(4):
            b.add_word(rows=[0], columns=[x])
        b.add_zone(
            "z",
            rows=[0.0],
            columns=[0.0, 1.0, 10.0, 11.0],
            x_clearance=_CL,
            y_clearance=_CL,
        )
        b.set_blockade_radius(2.0)
        b.add_mode("all", ["z"])
        spec = b.build()
        assert list(spec.zones[0].entangling_pairs) == [(0, 1), (2, 3)]
        assert spec.blockade_radius == 2.0

    def test_unknown_zone_is_rejected(self):
        b = with_zone(interleaved())
        with pytest.raises(ValueError, match="unknown zone: 'nope'"):
            b.add_word_bus("nope", src=[0], dst=[4])


# ── Round-trip ──


@pytest.mark.parametrize(
    "getter", [physical_spec, logical_spec], ids=["physical", "logical"]
)
class TestFromSpec:
    def test_round_trip_is_lossless(self, getter):
        """Rebuilding a shipped spec reproduces it exactly, in every field.

        Compared through the serialized form rather than a hand-picked set
        of attributes: a subset comparison only checks what someone thought
        to list, and silently passed while ``from_spec`` was dropping the
        capability flags and regenerating ``bitstring_order``.
        """
        spec = getter()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rebuilt = ArchBuilder.from_spec(spec).build()
        assert as_comparable(rebuilt) == as_comparable(spec)

    def test_inherited_paths_survive_because_the_search_cannot_recreate_them(
        self, getter
    ):
        """Bundled lane geometry is hardware-derived, not search output.

        No clearance reproduces it, and move durations are measured off
        these segment lengths, so a round-trip must not quietly swap them
        for search results.
        """
        spec = getter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searched = ArchBuilder.from_spec(
                spec, x_clearance=0.5, y_clearance=1.0
            ).build(recompute_paths=True)
        assert searched.paths != spec.paths
        assert set(searched.paths) == set(spec.paths)

    def test_editing_a_bus_without_clearances_raises(self, getter):
        spec = getter()
        b = ArchBuilder.from_spec(spec)
        name = spec._inner.zones[0].name
        b.add_word_bus(name, src=b.words[0, 0], dst=b.words[0, 0])
        with pytest.raises(ValueError, match="no clearances"):
            b.build()


# ── Parity with the legacy builder ──
#
# The two builders share one geometry layer, so an architecture expressed
# both ways must produce the same ArchSpec — words, zones, buses,
# participation, entangling pairs, modes and every transport path.


@dataclass(frozen=True)
class Layout:
    """One zone, described once and built through both builders."""

    rows: tuple[float, ...]  # y-coordinate of each grid row, µm
    columns: tuple[float, ...]  # x-coordinate of each grid column, µm
    word_shape: tuple[int, int]  # (num_rows, num_columns)
    words: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]  # (rows, columns)
    site_bus_words: tuple[int, ...] | None = None
    site_buses: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    word_buses: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    radius: float | None = None
    x_clearance: float = 0.25
    y_clearance: float = 3.0

    def legacy(self) -> ArchSpec:
        # The legacy builder is (x, y) / (num_x, num_y) throughout.
        zone = ZoneBuilder.from_positions(
            "z",
            list(self.columns),
            list(self.rows),
            (self.word_shape[1], self.word_shape[0]),
            x_clearance=self.x_clearance,
            y_clearance=self.y_clearance,
        )
        for i, (rs, cs) in enumerate(self.words):
            opts_in = self.site_bus_words is None or i in self.site_bus_words
            zone.add_word(list(cs), list(rs), has_site_bus=opts_in)
        for src, dst in self.site_buses:
            zone.add_site_bus(list(src), list(dst))
        for src, dst in self.word_buses:
            zone.add_word_bus(src=list(src), dst=list(dst))
        builder = LegacyArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        if self.radius is not None:
            builder.set_blockade_radius(self.radius)
        return builder.build()

    def modern(self) -> ArchSpec:
        b = ArchBuilder(
            grid_shape=(len(self.rows), len(self.columns)),
            word_shape=self.word_shape,
        )
        for rs, cs in self.words:
            b.add_word(rows=list(rs), columns=list(cs))
        b.add_zone(
            "z",
            rows=list(self.rows),
            columns=list(self.columns),
            x_clearance=self.x_clearance,
            y_clearance=self.y_clearance,
            words_with_site_buses=(
                None if self.site_bus_words is None else list(self.site_bus_words)
            ),
        )
        for src, dst in self.site_buses:
            b.add_site_bus("z", list(src), list(dst))
        for src, dst in self.word_buses:
            b.add_word_bus("z", list(src), list(dst))
        b.add_mode("all", ["z"])
        if self.radius is not None:
            b.set_blockade_radius(self.radius)
        return b.build()


def assert_specs_equal(a: ArchSpec, b: ArchSpec) -> None:
    """Compare two specs field by field, paths included."""
    assert a._inner.words == b._inner.words
    assert list(a._inner.zones) == list(b._inner.zones)
    assert list(a._inner.zone_buses) == list(b._inner.zone_buses)
    assert [(m.name, list(m.zones)) for m in a._inner.modes] == [
        (m.name, list(m.zones)) for m in b._inner.modes
    ]
    assert [list(m.bitstring_order) for m in a._inner.modes] == [
        list(m.bitstring_order) for m in b._inner.modes
    ]
    assert a.blockade_radius == b.blockade_radius
    assert (a.paths or {}) == (b.paths or {})


ROWS = Layout(
    rows=(0.0, 10.0),
    columns=(0.0, 1.0, 2.0, 3.0),
    word_shape=(1, 1),
    words=tuple(((y,), (x,)) for y in (0, 1) for x in range(4)),
    word_buses=(((0, 1, 2, 3), (4, 5, 6, 7)),),
)

INTERLEAVED = Layout(
    rows=(0.0, 10.0),
    columns=tuple(float(i) for i in range(8)),
    word_shape=(1, 2),
    words=tuple(((r,), (c, c + 4)) for r in (0, 1) for c in range(4)),
    site_buses=(((0,), (1,)),),
    word_buses=(((0, 1, 2, 3), (4, 5, 6, 7)),),
)

OPTED_OUT = Layout(
    rows=(0.0,),
    columns=(0.0, 10.0, 100.0, 130.0, 200.0, 230.0),
    word_shape=(1, 2),
    # Word 0 has a 10 µm site pitch and sits the bus out; words 1-2 use 30 µm.
    words=(((0,), (0, 1)), ((0,), (2, 3)), ((0,), (4, 5))),
    site_bus_words=(1, 2),
    site_buses=(((0,), (1,)),),
    x_clearance=0.5,
    y_clearance=0.5,
)

BLOCKADE = Layout(
    rows=(0.0,),
    columns=(0.0, 1.0, 10.0, 11.0),
    word_shape=(1, 1),
    words=tuple(((0,), (x,)) for x in range(4)),
    radius=2.0,
)


class TestParityWithLegacyBuilder:
    @pytest.mark.parametrize(
        "layout",
        [ROWS, INTERLEAVED, OPTED_OUT, BLOCKADE],
        ids=["rows", "interleaved", "opted-out", "blockade"],
    )
    def test_same_architecture_yields_the_same_spec(self, layout):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_specs_equal(layout.legacy(), layout.modern())

    def test_opted_out_word_gets_no_lanes_in_either_builder(self):
        """Word 0 sits the bus out, so it must not appear in any lane."""
        for spec in (OPTED_OUT.legacy(), OPTED_OUT.modern()):
            assert list(spec.zones[0].words_with_site_buses) == [1, 2]
            assert all(lane.word_id != 0 for lane in spec.paths)

    def test_multi_zone_with_a_connection_matches(self):
        x, y = [0.0, 1.0, 2.0, 3.0], [0.0]
        words = [([0], [c]) for c in range(4)]  # (rows, columns)

        zones = []
        for name, shift in (("a", 0.0), ("b", 50.0)):
            z = ZoneBuilder.from_positions(
                name,
                [v + shift for v in x],
                y,
                (1, 1),
                x_clearance=0.25,
                y_clearance=0.25,
            )
            for rs, cs in words:
                z.add_word(cs, rs)  # legacy is (x, y)
            z.add_word_bus(src=[0, 1], dst=[2, 3])
            zones.append(z)
        legacy_builder = LegacyArchBuilder()
        for z in zones:
            legacy_builder.add_zone(z)
        legacy_builder.connect(("a", [0, 1]), ("b", [0, 1]))
        legacy_builder.add_mode("all", ["a", "b"])

        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 1))
        for rs, cs in words:
            b.add_word(rows=rs, columns=cs)
        for name, shift in (("a", 0.0), ("b", 50.0)):
            b.add_zone(
                name,
                rows=y,
                columns=[v + shift for v in x],
                x_clearance=0.25,
                y_clearance=0.25,
            )
            b.add_word_bus(name, src=[0, 1], dst=[2, 3])
        b.connect(("a", [0, 1]), ("b", [0, 1]))
        b.add_mode("all", ["a", "b"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_specs_equal(legacy_builder.build(), b.build())


# ── Extending an existing architecture ──


def _extendable(spec):
    """A builder over a shipped spec, with clearances so it can route."""
    return ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=1.0), (
        spec._inner.zones[0].name
    )


@pytest.mark.parametrize(
    "getter", [physical_spec, logical_spec], ids=["physical", "logical"]
)
class TestExtendingWordBuses:
    def test_appending_preserves_every_inherited_lane(self, getter):
        """The whole point: a new bus must not re-route the existing ones.

        Bundled lane geometry is hardware-derived, so recomputing it as a
        side effect of adding a bus would silently shift move durations.
        """
        spec = getter()
        b, zone = _extendable(spec)
        existing = spec._inner.zones[0].word_buses[0]
        # The reverse transport: legal, and not a duplicate of any bus.
        b.add_word_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        assert all(out.paths[k] == v for k, v in spec.paths.items())

    def test_appending_adds_lanes_for_the_new_bus_only(self, getter):
        spec = getter()
        b, zone = _extendable(spec)
        n_before = len(spec._inner.zones[0].word_buses)
        existing = spec._inner.zones[0].word_buses[0]
        # The reverse transport: legal, and not a duplicate of any bus.
        b.add_word_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        assert len(out.zones[0].word_buses) == n_before + 1
        new_lanes = {k for k in out.paths if k not in spec.paths}
        assert new_lanes and all(
            k.move_type == MoveType.WORD and k.bus_id == n_before for k in new_lanes
        )

    def test_unrealizable_extension_is_rejected_at_the_call(self, getter):
        """A reversal cannot be swept by an AOD, so it never reaches build."""
        spec = getter()
        b, zone = _extendable(spec)
        src = list(spec._inner.zones[0].word_buses[0].src)
        with pytest.raises(ValueError, match="stay on its own|add or drop a tone"):
            b.add_word_bus(zone, src=src, dst=list(reversed(src)))
        assert len(b._zones[0].word_buses) == len(spec._inner.zones[0].word_buses)

    def test_extending_without_clearances_raises(self, getter):
        spec = getter()
        b = ArchBuilder.from_spec(spec)
        zone = spec._inner.zones[0].name
        existing = spec._inner.zones[0].word_buses[0]
        # The reverse transport: legal, and not a duplicate of any bus.
        b.add_word_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with pytest.raises(ValueError, match="no clearances"):
            b.build()


@pytest.mark.parametrize("getter", [physical_spec], ids=["physical"])
class TestExtendingSiteBuses:
    def test_appending_preserves_every_inherited_lane(self, getter):
        spec = getter()
        b, zone = _extendable(spec)
        existing = spec._inner.zones[0].site_buses[0]
        b.add_site_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        assert all(out.paths[k] == v for k, v in spec.paths.items())

    def test_appending_adds_lanes_for_the_new_bus_only(self, getter):
        spec = getter()
        b, zone = _extendable(spec)
        n_before = len(spec._inner.zones[0].site_buses)
        existing = spec._inner.zones[0].site_buses[0]
        b.add_site_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        assert len(out.zones[0].site_buses) == n_before + 1
        new_lanes = {k for k in out.paths if k not in spec.paths}
        assert new_lanes and all(
            k.move_type == MoveType.SITE and k.bus_id == n_before for k in new_lanes
        )

    def test_new_site_bus_only_carries_participating_words(self, getter):
        """The shipped spec runs site buses on odd words only."""
        spec = getter()
        movers = list(spec._inner.zones[0].words_with_site_buses)
        b, zone = _extendable(spec)
        existing = spec._inner.zones[0].site_buses[0]
        b.add_site_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        new_lanes = [k for k in out.paths if k not in spec.paths]
        assert {k.word_id for k in new_lanes} == set(movers)

    def test_unrealizable_extension_is_rejected_at_the_call(self, getter):
        spec = getter()
        b, zone = _extendable(spec)
        src = list(spec._inner.zones[0].site_buses[0].src)
        with pytest.raises(ValueError, match="stay on its own|add or drop a tone"):
            b.add_site_bus(zone, src=src, dst=list(reversed(src)))


class TestPathInheritanceBoundaries:
    def test_recompute_paths_discards_inherited_geometry(self):
        spec = physical_spec()
        b = ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build(recompute_paths=True)
        assert set(out.paths) == set(spec.paths)
        assert out.paths != spec.paths

    def test_changed_participation_invalidates_inherited_site_lanes(self):
        """Participation decides which words get site lanes."""
        spec = physical_spec()
        b = ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=1.0)
        zone = b._zones[0]
        zone.words_with_site_buses = zone.words_with_site_buses[:-1]
        reusable = b._reusable_inherited(0, zone)
        assert reusable, "word-bus lanes should still be inheritable"
        assert all(k.move_type == MoveType.WORD for k in reusable)

    def test_a_fresh_builder_inherits_nothing(self):
        b = with_zone(interleaved())
        b.add_word_bus("z", src=[0, 1, 2, 3], dst=[4, 5, 6, 7])
        assert b._reusable_inherited(0, b._zones[0]) == {}


# ── The contract from bloqade-internal#445 ──
#
# "add buses given an architecture": select words by grid region, append
# ordered buses, leave the calibrated paths alone.


class TestAddBusesGivenAnArchitecture:
    """Behaviours the reference implementation in #445 specifies."""

    def test_selection_preserves_order_so_src_i_maps_to_dst_i(self):
        b = interleaved()
        assert b.words[0, :] == [0, 1, 2, 3]
        assert b.words[0, [3, 2, 1, 0]] == [3, 2, 1, 0]

    def test_whole_axis_selects_everything_in_order(self):
        b = interleaved()
        assert b.words[:, :] == list(range(8))

    def test_duplicate_indices_are_rejected(self):
        with pytest.raises(ValueError, match="repeated"):
            interleaved().words[0, [0, 0]]

    def test_out_of_range_indices_are_rejected(self):
        with pytest.raises(IndexError, match="out of range"):
            interleaved().words[0, [99]]

    def test_mismatched_selection_sizes_are_rejected(self):
        b = with_zone(interleaved())
        with pytest.raises(ValueError, match="src has 4 entries but dst has 2"):
            b.add_word_bus("z", src=b.words[0, :], dst=b.words[1, [0, 1]])

    def test_a_duplicate_word_bus_is_rejected(self):
        b = with_zone(interleaved())
        src, dst = b.words[0, :], b.words[1, :]
        b.add_word_bus("z", src, dst)
        with pytest.raises(ValueError, match="word bus 0 already has this exact"):
            b.add_word_bus("z", src, dst)
        assert len(b._zones[0].word_buses) == 1

    def test_a_duplicate_site_bus_is_rejected(self):
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 2))
        b.add_word(rows=[0], columns=[0, 1])
        b.add_word(rows=[0], columns=[2, 3])
        with_zone(b)
        b.add_site_bus("z", src=[0], dst=[1])
        with pytest.raises(ValueError, match="site bus 0 already has this exact"):
            b.add_site_bus("z", src=[0], dst=[1])

    def test_the_reverse_of_a_bus_is_not_a_duplicate(self):
        b = with_zone(interleaved())
        src, dst = b.words[0, :], b.words[1, :]
        b.add_word_bus("z", src, dst)
        b.add_word_bus("z", dst, src)
        assert len(b._zones[0].word_buses) == 2

    def test_existing_bus_ids_are_stable_across_an_append(self):
        """A bus's index is its bus_id, and preserved paths key off it."""
        spec = physical_spec()
        before = [(list(x.src), list(x.dst)) for x in spec._inner.zones[0].word_buses]
        b, zone = _extendable(spec)
        existing = spec._inner.zones[0].word_buses[0]
        b.add_word_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        after = [(list(x.src), list(x.dst)) for x in out.zones[0].word_buses]
        assert after[: len(before)] == before

    def test_calibrated_paths_survive_adding_a_bus(self):
        """The headline requirement: extending must not re-route the rest."""
        spec = physical_spec()
        b, zone = _extendable(spec)
        existing = spec._inner.zones[0].word_buses[0]
        b.add_word_bus(zone, src=list(existing.dst), dst=list(existing.src))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = b.build()
        assert {k: out.paths[k] for k in spec.paths} == dict(spec.paths)


# ── Regressions from the Copilot review of PR #1012 ──


class TestSpecFidelity:
    """``from_spec`` must reproduce a spec, or refuse it — never reinterpret."""

    def test_shape_must_be_whole_numbers(self):
        """``int(2.5)`` would silently accept a malformed index space."""
        with pytest.raises(ValueError, match="grid_shape must be two positive"):
            ArchBuilder(grid_shape=(2.5, 1), word_shape=(1, 1))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="word_shape must be two positive"):
            ArchBuilder(grid_shape=(2, 1), word_shape=(1, 0))

    def test_capability_flags_survive_a_round_trip(self):
        """build() defaulted both to False, silently clearing them."""
        spec = perturbed(
            physical_spec(),
            lambda d: d.update(feed_forward=True, atom_reloading=True),
        )
        rebuilt = ArchBuilder.from_spec(spec).build()
        assert (rebuilt.feed_forward, rebuilt.atom_reloading) == (True, True)

    def test_explicit_capability_arguments_still_win(self):
        spec = perturbed(physical_spec(), lambda d: d.update(feed_forward=True))
        rebuilt = ArchBuilder.from_spec(spec).build(feed_forward=False)
        assert rebuilt.feed_forward is False

    def test_site_order_is_preserved_not_canonicalized(self):
        """Site IDs address bus endpoints and lanes; renumbering re-maps them."""

        def reverse_every_word(d):
            for w in d["words"]:
                w["sites"] = list(reversed(w["sites"]))

        spec = perturbed(physical_spec(), reverse_every_word)
        rebuilt = ArchBuilder.from_spec(spec).build()
        assert list(rebuilt._inner.words[0].sites) == list(spec._inner.words[0].sites)

    def test_custom_bitstring_order_is_preserved(self):
        """An ArchSpec may carry any measurement ordering."""

        def reverse_order(d):
            for mode in d["modes"]:
                mode["bitstring_order"] = list(reversed(mode["bitstring_order"]))

        spec = perturbed(physical_spec(), reverse_order)
        rebuilt = ArchBuilder.from_spec(spec).build()
        assert list(rebuilt._inner.modes[0].bitstring_order) == list(
            spec._inner.modes[0].bitstring_order
        )

    def test_empty_participation_is_not_read_as_everything(self):
        """[] alongside buses is a real value, not an unset one."""
        spec = perturbed(
            physical_spec(), lambda d: d["zones"][0].update(words_with_site_buses=[])
        )
        with pytest.raises(ValueError, match="no words_with_site_buses"):
            ArchBuilder.from_spec(spec)

    def test_absent_participation_still_leaves_a_zone_extensible(self):
        """With no buses of that kind, [] carries no information."""
        spec = logical_spec()
        assert not spec._inner.zones[0].site_buses
        b = ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=1.0)
        assert b._zones[0].words_with_site_buses == tuple(range(b.num_words))


class TestConnectContract:
    def test_same_zone_endpoints_are_rejected_at_the_call(self):
        """Rust rejects these at build; phase 3 promises to catch them here."""
        spec = physical_spec()
        b = ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=1.0)
        name = spec._inner.zones[0].name
        with pytest.raises(ValueError, match="inter-zone bus"):
            b.connect((name, [0, 1]), (name, [2, 3]))

    def test_zone_bus_endpoints_spanning_zones_are_rejected(self):
        """connect addresses one zone per side; a mixed endpoint cannot map."""
        x, y = [0.0, 1.0, 2.0, 3.0], [0.0]
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 1))
        for c in range(4):
            b.add_word(rows=[0], columns=[c])
        for name, shift in (("a", 0.0), ("b", 50.0)):
            b.add_zone(
                name,
                rows=y,
                columns=[v + shift for v in x],
                x_clearance=0.25,
                y_clearance=0.25,
            )
        b.connect(("a", [0, 1]), ("b", [0, 1]))
        b.add_mode("all", ["a", "b"])
        spec = b.build()

        def mix_endpoint(d):
            # Every pair still crosses a boundary, which is all Rust asks,
            # but the src endpoint now names both zones.
            d["zone_buses"][0]["src"] = [[0, 0], [1, 1]]
            d["zone_buses"][0]["dst"] = [[1, 0], [0, 1]]

        with pytest.raises(ValueError, match="spans zones"):
            ArchBuilder.from_spec(perturbed(spec, mix_endpoint))

    def test_empty_zone_bus_endpoint_is_rejected(self):
        b = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 1))
        for c in range(4):
            b.add_word(rows=[0], columns=[c])
        for name, shift in (("a", 0.0), ("b", 50.0)):
            b.add_zone(
                name,
                rows=[0.0],
                columns=[v + shift for v in (0.0, 1.0, 2.0, 3.0)],
                x_clearance=0.25,
                y_clearance=0.25,
            )
        b.connect(("a", [0, 1]), ("b", [0, 1]))
        b.add_mode("all", ["a", "b"])
        spec = b.build()
        # Rust accepts a bus with both endpoints empty, so this is reachable.
        with pytest.raises(ValueError, match="empty src endpoint"):
            ArchBuilder.from_spec(
                perturbed(spec, lambda d: d["zone_buses"][0].update(src=[], dst=[]))
            )


# ── Word templates ArchSpec allows but (rows x columns) cannot describe ──


def busless_spec() -> ArchSpec:
    """Two 2x2 words on a 4x4 grid, no buses — isolates word restoration.

    Rows 2 and 3 are left empty so a test can move a site somewhere free
    without tripping the overlap guard.
    """
    b = ArchBuilder(grid_shape=(4, 4), word_shape=(2, 2))
    for rs, cs in (([0, 1], [0, 1]), ([0, 1], [2, 3])):
        b.add_word(rows=rs, columns=cs)
    b.add_zone(
        "z",
        rows=[0.0, 10.0, 20.0, 30.0],
        columns=[0.0, 1.0, 2.0, 3.0],
        x_clearance=0.25,
        y_clearance=3.0,
    )
    b.add_mode("all", ["z"])
    return b.build()


class TestVerbatimTemplateRestore:
    """A word is an ordered list of positions; its index is the site ID.

    ``ArchSpec`` is more general than ``(rows x columns)`` here, so the
    template is restored as-is rather than regenerated — regenerating would
    renumber sites and re-map every bus endpoint and inherited lane.
    """

    def test_column_major_word_round_trips(self):
        spec = perturbed(
            busless_spec(),
            lambda d: d["words"][0].update(sites=[[0, 0], [0, 1], [1, 0], [1, 1]]),
        )
        assert as_comparable(ArchBuilder.from_spec(spec).build()) == as_comparable(spec)

    def test_non_rectangular_word_round_trips(self):
        """Four sites that are not a Cartesian product at all."""
        spec = perturbed(
            busless_spec(),
            lambda d: d["words"][0].update(sites=[[0, 0], [1, 0], [0, 1], [2, 2]]),
        )
        assert as_comparable(ArchBuilder.from_spec(spec).build()) == as_comparable(spec)

    def test_words_of_differing_shapes_round_trip(self):
        """Rust's only cross-word rule is an equal site count."""

        def reshape(d):
            d["words"][1]["sites"] = [[0, 3], [1, 3], [2, 3], [3, 3]]  # 1x4

        spec = perturbed(busless_spec(), reshape)
        b = ArchBuilder.from_spec(spec)
        assert b.word_shape is None
        assert as_comparable(b.build()) == as_comparable(spec)

    def test_word_shape_survives_a_uniform_template(self):
        b = ArchBuilder.from_spec(busless_spec())
        assert b.word_shape == (2, 2)
        assert b.sites[0, :] == [0, 1]

    def test_sites_query_is_unavailable_without_one_shape(self):
        spec = perturbed(
            busless_spec(),
            lambda d: d["words"][0].update(sites=[[0, 0], [0, 1], [1, 0], [1, 1]]),
        )
        b = ArchBuilder.from_spec(spec)
        assert b.word_shape is None
        with pytest.raises(ValueError, match="no single row-major"):
            _ = b.sites[0, :]

    def test_word_sites_reports_positions_for_any_template(self):
        spec = perturbed(
            busless_spec(),
            lambda d: d["words"][0].update(sites=[[0, 0], [0, 1], [1, 0], [1, 1]]),
        )
        b = ArchBuilder.from_spec(spec)
        # (row, column) per site ID, matching the spec's own ordering.
        assert b.word_sites(0) == [(0, 0), (1, 0), (0, 1), (1, 1)]

    def test_sites_per_word_follows_the_restored_template(self):
        assert ArchBuilder.from_spec(busless_spec()).sites_per_word == 4


class TestTemplateGuardsRustDoesNotMake:
    """Rust accepts these; both put two atoms in one place."""

    def test_overlapping_words_are_rejected(self):
        spec = perturbed(
            busless_spec(),
            lambda d: d["words"][1].update(sites=list(d["words"][0]["sites"])),
        )
        with pytest.raises(ValueError, match="both occupy grid position"):
            ArchBuilder.from_spec(spec)

    def test_a_word_listing_a_position_twice_is_rejected(self):
        spec = perturbed(
            busless_spec(),
            lambda d: d["words"][1].update(sites=[[2, 0], [2, 0], [2, 1], [3, 1]]),
        )
        with pytest.raises(ValueError, match="more than once"):
            ArchBuilder.from_spec(spec)
