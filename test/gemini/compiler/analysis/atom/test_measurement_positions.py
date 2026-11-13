"""Tests for ``AtomInterpreter.get_measurement_positions``.

The three scopes only differ when some atoms sit outside the measured
zones, and some inside them go unread. The bundled Gemini specs have a
single zone and kernels that read everything they fill, so these use a
hand-built two-zone architecture where all three genuinely diverge.
"""

import pytest
from bloqade.lanes.prelude import kernel
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    Mode as RustMode,
    Zone as RustZone,
    SiteBus,
    WordBus,
)


def _two_zone_arch() -> ArchSpec:
    """Two words of two sites each, in two non-overlapping zones."""

    def zone(x_positions):
        return RustZone(
            name="z",
            grid=RustGrid.from_positions(x_positions, [0.0, 1.0]),
            site_buses=[SiteBus(src=[0], dst=[1])],
            word_buses=[WordBus(src=[0], dst=[1])],
            words_with_site_buses=[0, 1],
            sites_with_word_buses=[0],
            entangling_pairs=[(0, 1)],
        )

    return ArchSpec.from_components(
        # Sites are (x_idx, y_idx).
        words=(Word(sites=((0, 0), (1, 0))), Word(sites=((0, 1), (1, 1)))),
        zones=(zone([0.0, 10.0]), zone([100.0, 110.0])),
        modes=[RustMode(name="all", zones=[0, 1], bitstring_order=[])],
    )


_ARCH = _two_zone_arch()


@kernel
def _partial_readout():
    """Fill four atoms across both zones, measure only zone 0, and read
    back only one of the two atoms that zone 0 holds."""
    state0 = move.load()
    state1 = move.fill(
        state0,
        location_addresses=(
            move.LocationAddress(0, 0, 0),  # zone 0, word 0, site 0
            move.LocationAddress(1, 0, 0),  # zone 0, word 1, site 0
            move.LocationAddress(0, 0, 1),  # zone 1, word 0, site 0
            move.LocationAddress(1, 0, 1),  # zone 1, word 1, site 0
        ),
    )
    future = move.end_measure(state1, zone_addresses=(move.ZoneAddress(0),))
    return move.get_future_result(
        future,
        zone_address=move.ZoneAddress(0),
        location_address=move.LocationAddress(0, 0, 0),
    )


@kernel
def _both_zones_measured():
    """Same fill, but the measurement covers both zones — so zone 1's
    addresses land in the second block of the frame."""
    state0 = move.load()
    state1 = move.fill(
        state0,
        location_addresses=(
            move.LocationAddress(0, 0, 0),
            move.LocationAddress(1, 0, 0),
            move.LocationAddress(0, 0, 1),
            move.LocationAddress(1, 0, 1),
        ),
    )
    future = move.end_measure(
        state1, zone_addresses=(move.ZoneAddress(0), move.ZoneAddress(1))
    )
    return move.get_future_result(
        future,
        zone_address=move.ZoneAddress(1),
        location_address=move.LocationAddress(1, 0, 1),
    )


def _positions():
    interp = atom.AtomInterpreter(kernel, arch_spec=_ARCH)
    return interp.get_measurement_positions(_partial_readout)


def _both_zone_positions():
    interp = atom.AtomInterpreter(kernel, arch_spec=_ARCH)
    return interp.get_measurement_positions(_both_zones_measured)


def _frame_of(atoms):
    """``{(word, site, zone): frame_index}`` for readability in asserts."""
    return {
        (
            a.location_address.word_id,
            a.location_address.site_id,
            a.location_address.zone_id,
        ): a.frame_index
        for a in atoms
    }


def test_one_snapshot_per_measurement_statement():
    result = _positions()
    assert len(result.measurements) == 1
    assert result.measurements[0].index == 0
    assert [z.zone_id for z in result.measurements[0].zone_addresses] == [0]


def test_three_scopes_are_genuinely_nested():
    """readout ⊆ measured_zones ⊆ qpu_state, with each strictly smaller."""
    snapshot = _positions().measurements[0]

    assert len(snapshot.readout) == 1
    assert len(snapshot.measured_zones) == 2
    assert len(snapshot.qpu_state) == 4

    readout_addrs = {a.location_address for a in snapshot.readout}
    zone_addrs = {a.location_address for a in snapshot.measured_zones}
    qpu_addrs = {a.location_address for a in snapshot.qpu_state}
    assert readout_addrs < zone_addrs < qpu_addrs


def test_measured_zones_includes_unread_atoms():
    """A zone readout measures every atom present, not just the ones the
    kernel bothers to read."""
    snapshot = _positions().measurements[0]
    unread = [
        atom_position
        for atom_position in snapshot.measured_zones
        if atom_position.location_address
        not in {a.location_address for a in snapshot.readout}
    ]
    assert len(unread) == 1
    assert unread[0].location_address.zone_id == 0
    assert unread[0].measurement_id is None


def test_qpu_state_includes_atoms_outside_the_measured_zones():
    snapshot = _positions().measurements[0]
    other_zone = [a for a in snapshot.qpu_state if a.location_address.zone_id == 1]
    assert len(other_zone) == 2
    assert all(a.measurement_id is None for a in other_zone)


def test_only_readout_atoms_carry_a_measurement_id():
    snapshot = _positions().measurements[0]
    assert [a.measurement_id for a in snapshot.readout] == [0]
    assert all(a.measurement_id is None for a in snapshot.qpu_state)


def test_positions_resolve_through_the_arch_spec():
    snapshot = _positions().measurements[0]
    for atom_position in snapshot.qpu_state:
        assert atom_position.position == _ARCH.get_position(
            atom_position.location_address
        )
    # Zone 1's grid starts at x=100, so its atoms are far from zone 0's.
    zone1 = [a for a in snapshot.qpu_state if a.location_address.zone_id == 1]
    assert all(a.position[0] >= 100.0 for a in zone1)


def test_flat_readout_is_ordered_by_measurement_id():
    result = _positions()
    assert [a.measurement_id for a in result.readout] == [0]
    assert result.positions == tuple(a.position for a in result.readout)


def test_readout_is_stable_across_repeated_runs():
    """The interpreter clears its bookkeeping on initialize, so running
    twice must not double-count."""
    interp = atom.AtomInterpreter(kernel, arch_spec=_ARCH)
    first = interp.get_measurement_positions(_partial_readout)
    second = interp.get_measurement_positions(_partial_readout)
    assert len(first.measurements) == len(second.measurements) == 1
    assert first.readout == second.readout
    assert first.measurements[0].qpu_state == second.measurements[0].qpu_state


def test_collection_is_a_pure_function_of_the_converged_frame():
    """Snapshots are read out of the frame after the analysis converges,
    not accumulated while it runs. Abstract interpretation may visit a
    statement any number of times before reaching a fixpoint, so anything
    accumulated during the walk has to be made idempotent by hand;
    collecting afterwards makes repeat visits irrelevant by construction.
    """
    interp = atom.AtomInterpreter(kernel, arch_spec=_ARCH)
    frame, _ = interp.run(_partial_readout)

    first = interp.collect_measurement_positions(_partial_readout, frame)
    second = interp.collect_measurement_positions(_partial_readout, frame)
    assert first == second
    assert len(first.measurements) == 1
    assert len(first.measurements[0].readout) == 1


def test_a_measurement_executing_twice_is_rejected():
    """One snapshot per measurement *statement* only models a program
    where each statement runs once. A second visit mints a fresh
    ``measurement_id``, the two ``MeasureResult`` values are incomparable
    so the lattice joins them to ``Unknown``, and the readout drops out of
    the walk — leaving a record set that no longer covers 0..n-1. That is
    a genuine modelling failure (overlapping snapshots), so it must be
    reported rather than silently producing partial snapshots."""
    interp = atom.AtomInterpreter(kernel, arch_spec=_ARCH)
    frame, _ = interp.run(_partial_readout)

    (readout_stmt,) = [
        stmt
        for stmt in _partial_readout.callable_region.walk()
        if isinstance(stmt, move.GetFutureResult)
    ]
    first_visit = frame.get(readout_stmt.result)
    assert isinstance(first_visit, atom.MeasureResult)

    # A second visit mints the next id for the same location; joining the
    # two is what the fixpoint would store.
    second_visit = atom.MeasureResult(
        first_visit.measurement_id + 1,
        first_visit.qubit_id,
        first_visit.location_address,
    )
    frame.set(readout_stmt.result, first_visit.join(second_visit))
    assert isinstance(frame.get(readout_stmt.result), atom.Unknown)

    with pytest.raises(ValueError, match="executed more than once"):
        interp.collect_measurement_positions(_partial_readout, frame)


def test_readouts_attach_to_their_own_measurement_statement():
    """A readout finds its snapshot through
    ``GetFutureResult.measurement_future.owner``, so the association holds
    however many times the analysis visited either statement and whatever
    the interpreter's counters happen to read."""
    interp = atom.AtomInterpreter(kernel, arch_spec=_ARCH)
    positions = interp.get_measurement_positions(_both_zones_measured)

    (snapshot,) = positions.measurements
    (readout,) = snapshot.readout
    # The kernel reads zone 1's word 1, and that is what comes back —
    # attribution is structural, not positional.
    assert readout.location_address == move.LocationAddress(1, 0, 1)
    assert readout.measurement_id == 0


# ── frame_index ──────────────────────────────────────────────────


def test_frame_index_is_word_major_site_minor():
    """Within a zone the frame walks addresses word-major / site-minor,
    so it equals ``word_id * sites_per_word + site_id``."""
    snapshot = _positions().measurements[0]
    # 2 words x 2 sites: word 0 site 0 -> 0, word 1 site 0 -> 2.
    assert _frame_of(snapshot.measured_zones) == {(0, 0, 0): 0, (1, 0, 0): 2}


def test_frame_index_matches_get_zone_index_for_a_single_zone_frame():
    """A one-zone frame has zero offset, so it must agree exactly with
    the arch spec's own within-zone index."""
    snapshot = _positions().measurements[0]
    for atom_position in snapshot.measured_zones:
        assert atom_position.frame_index == _ARCH.get_zone_index(
            atom_position.location_address,
            move.ZoneAddress(atom_position.location_address.zone_id),
        )


def test_frame_index_is_always_populated_inside_the_frame():
    """``readout`` and ``measured_zones`` are built from the
    measurement's own zones, so a ``None`` there would be a bug, not a
    legitimately absent value. Only ``qpu_state`` may carry ``None``,
    and only for atoms outside the covered zones."""
    for result in (_positions(), _both_zone_positions()):
        for snapshot in result.measurements:
            covered = {z.zone_id for z in snapshot.zone_addresses}
            assert all(a.frame_index is not None for a in snapshot.readout)
            assert all(a.frame_index is not None for a in snapshot.measured_zones)
            for atom_position in snapshot.qpu_state:
                in_frame = atom_position.location_address.zone_id in covered
                assert (atom_position.frame_index is not None) is in_frame


def test_atom_position_requires_both_optional_fields():
    """The optional fields have no default, so they cannot be silently
    left unpopulated at a construction site."""
    with pytest.raises(TypeError):
        atom.AtomPosition(  # type: ignore[call-arg]
            qubit_id=0,
            location_address=move.LocationAddress(0, 0, 0),
            position=(0.0, 0.0),
        )


def test_frame_index_is_none_outside_the_measured_zones():
    """Zone 1 isn't measured here, so its atoms have no slot in the
    frame even though they show up in ``qpu_state``."""
    snapshot = _positions().measurements[0]
    outside = [a for a in snapshot.qpu_state if a.location_address.zone_id == 1]
    assert outside and all(a.frame_index is None for a in outside)
    inside = [a for a in snapshot.qpu_state if a.location_address.zone_id == 0]
    assert inside and all(a.frame_index is not None for a in inside)


def test_frame_index_offsets_later_zones():
    """With both zones measured, zone 1's block starts one stride in."""
    snapshot = _both_zone_positions().measurements[0]
    # Each zone contributes 2 words x 2 sites = 4 slots.
    assert snapshot.frame_size == 8
    assert _frame_of(snapshot.qpu_state) == {
        (0, 0, 0): 0,  # zone 0 block: word*2 + site
        (1, 0, 0): 2,
        (0, 0, 1): 4,  # zone 1 block: 4 + word*2 + site
        (1, 0, 1): 6,
    }


def test_frame_index_follows_zone_address_order_not_zone_id():
    """The offset comes from a zone's position in ``zone_addresses``,
    so the same atom's frame index depends on how the measurement was
    declared."""
    single = _positions().measurements[0]
    both = _both_zone_positions().measurements[0]
    assert single.frame_size == 4
    assert both.frame_size == 8
    # Zone 0 is first in both, so its atoms keep the same indices...
    assert _frame_of(single.measured_zones)[(0, 0, 0)] == 0
    assert _frame_of(both.measured_zones)[(0, 0, 0)] == 0
    # ...while zone 1 only gains a slot once it is actually measured.
    assert _frame_of(single.qpu_state)[(0, 0, 1)] is None
    assert _frame_of(both.qpu_state)[(0, 0, 1)] == 4


def test_frame_indices_are_within_frame_size_and_unique():
    for snapshot in (
        _positions().measurements[0],
        _both_zone_positions().measurements[0],
    ):
        indices = [
            a.frame_index for a in snapshot.qpu_state if a.frame_index is not None
        ]
        assert len(set(indices)) == len(indices)
        assert all(0 <= i < snapshot.frame_size for i in indices)


def test_frame_index_is_independent_of_measurement_id():
    """The readout atom here is the *second* zone's second word, so its
    frame index and its measurement id must not coincide."""
    snapshot = _both_zone_positions().measurements[0]
    assert len(snapshot.readout) == 1
    readout = snapshot.readout[0]
    assert readout.measurement_id == 0
    assert readout.frame_index == 6


# ── Against the bundled Gemini spec ──────────────────────────────


@pytest.mark.slow
def test_matches_shot_remapping_ordering_end_to_end():
    """``readout[k]`` and ``get_shot_remapping().mapping[k]`` must
    describe the same atom — one geometrically, one as a bit index."""
    from bloqade.lanes.transform import LogicalPipeline
    from bloqade.lanes.arch.gemini import physical

    from bloqade import qubit, squin
    from bloqade.gemini import logical as gemini_logical

    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        reg = qubit.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        return gemini_logical.terminal_measure(reg)

    arch_spec = physical.get_arch_spec()
    physical_move = LogicalPipeline(transversal_rewrite=True).emit(main)

    interp = atom.AtomInterpreter(physical_move.dialects, arch_spec=arch_spec)
    positions = interp.get_measurement_positions(physical_move)

    remapping = atom.AtomInterpreter(
        physical_move.dialects, arch_spec=arch_spec
    ).get_shot_remapping(physical_move)
    assert isinstance(remapping, atom.ShotRemappingOk)

    assert len(positions.readout) == len(remapping.mapping)
    assert [a.measurement_id for a in positions.readout] == list(
        range(len(positions.readout))
    )
    # Same atom, described two ways: the position must be the one the
    # remapping's bit index points at.
    zone0_locations = list(arch_spec.yield_zone_locations(move.ZoneAddress(0)))
    for atom_position, bit in zip(positions.readout, remapping.mapping):
        assert atom_position.position == arch_spec.get_position(zone0_locations[bit])
