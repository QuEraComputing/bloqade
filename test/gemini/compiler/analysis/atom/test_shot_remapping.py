"""Unit tests for ``get_shot_remapping``.

``mapping[k]`` must be the frame slot of measurement record ``k``, so
that ``frame[:, mapping]`` yields the per-measurement array that
post-processing indexes by ``measurement_id``.

The old fixtures here gave every record ``measurement_id == 0`` and
explicitly disclaimed the field, so the ordering contract was never
exercised (issue #967). These build records with distinct, non-identity
ids on purpose.
"""

import pytest
from kirin.dialects import ilist
from bloqade.lanes.prelude import kernel
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.analysis.atom import (
    AtomPosition,
    AtomInterpreter,
    ShotRemappingOk,
    ShotRemappingErr,
    MeasurementSnapshot,
    MeasurementPositions,
    ShotRemappingDiagnostic,
)
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    Mode as RustMode,
    Zone as RustZone,
    SiteBus,
    WordBus,
)
from bloqade.lanes.bytecode.encoding import ZoneAddress, LocationAddress
from bloqade.lanes.analysis.atom._shot_remapping import get_shot_remapping

from bloqade import squin
from bloqade.gemini import logical
from bloqade.gemini.post_processing import build_post_processing


def _atom(measurement_id: int | None, frame_index: int | None, site: int = 0):
    return AtomPosition(
        qubit_id=site,
        location_address=LocationAddress(0, site, 0),
        position=(float(site), 0.0),
        measurement_id=measurement_id,
        frame_index=frame_index,
    )


def _snapshot(readout, *, index: int = 0, frame_size: int = 8):
    return MeasurementSnapshot(
        index=index,
        zone_addresses=(ZoneAddress(0),),
        frame_size=frame_size,
        readout=tuple(readout),
        measured_zones=tuple(readout),
        qpu_state=tuple(readout),
    )


def _positions(*snapshots):
    return MeasurementPositions(measurements=tuple(snapshots))


# ── ordering: the contract issue #967 is about ───────────────────


def test_mapping_is_indexed_by_measurement_id_not_traversal_order():
    """Records arriving in a scrambled order must still land at their
    own record index."""
    result = get_shot_remapping(
        _positions(
            _snapshot(
                [
                    _atom(measurement_id=2, frame_index=50, site=2),
                    _atom(measurement_id=0, frame_index=10, site=0),
                    _atom(measurement_id=1, frame_index=30, site=1),
                ]
            )
        )
    )
    assert isinstance(result, ShotRemappingOk)
    assert result.mapping == [10, 30, 50]


def test_mapping_covers_every_record():
    readout = [_atom(measurement_id=i, frame_index=i * 7, site=i) for i in range(5)]
    result = get_shot_remapping(_positions(_snapshot(readout)))
    assert isinstance(result, ShotRemappingOk)
    assert result.mapping == [0, 7, 14, 21, 28]


def test_reports_frame_size_and_measurement_index():
    result = get_shot_remapping(
        _positions(_snapshot([_atom(0, 3)], index=4, frame_size=64))
    )
    assert isinstance(result, ShotRemappingOk)
    assert result.frame_size == 64
    assert result.measurement_index == 4


def test_no_measurement_statement_returns_diagnostic():
    result = get_shot_remapping(_positions())
    assert isinstance(result, ShotRemappingErr)
    assert isinstance(result.diagnostic, ShotRemappingDiagnostic)
    assert "no measurement statement" in result.diagnostic.message


def test_empty_readout_is_valid():
    """A measurement whose results the kernel never reads yields an
    empty mapping, not an error."""
    result = get_shot_remapping(_positions(_snapshot([])))
    assert isinstance(result, ShotRemappingOk)
    assert result.mapping == []


def test_non_contiguous_record_ids_return_diagnostic():
    """Holes would leave uninitialised slots in ``mapping``, so they are
    rejected rather than silently filled."""
    result = get_shot_remapping(
        _positions(_snapshot([_atom(0, 10), _atom(2, 30, site=2)]))
    )
    assert isinstance(result, ShotRemappingErr)
    assert "contiguous" in result.diagnostic.message


def test_record_without_a_frame_slot_returns_diagnostic():
    result = get_shot_remapping(
        _positions(_snapshot([_atom(measurement_id=0, frame_index=None)]))
    )
    assert isinstance(result, ShotRemappingErr)
    assert "no slot" in result.diagnostic.message


# ── multiple frames ──────────────────────────────────────────────


def test_uses_the_last_frame():
    """The terminal readout is the frame the machine returns."""
    result = get_shot_remapping(
        _positions(
            _snapshot([], index=0, frame_size=8),
            _snapshot([_atom(0, 5)], index=1, frame_size=16),
        )
    )
    assert isinstance(result, ShotRemappingOk)
    assert result.mapping == [5]
    assert result.measurement_index == 1
    assert result.frame_size == 16


def test_readouts_in_an_earlier_frame_return_diagnostic():
    """One flat projection cannot span two separately-returned frames."""
    result = get_shot_remapping(
        _positions(
            _snapshot([_atom(0, 5)], index=0),
            _snapshot([_atom(1, 6)], index=1),
        )
    )
    assert isinstance(result, ShotRemappingErr)
    assert "multiple frames" in result.diagnostic.message


# ── Unresolvable locations ───────────────────────────────────────


@kernel
def _fills_an_address_outside_the_arch():
    """Word 7 doesn't exist in the two-word arch below."""
    state0 = move.load()
    state1 = move.fill(state0, location_addresses=(move.LocationAddress(7, 0, 0),))
    future = move.end_measure(state1, zone_addresses=(move.ZoneAddress(0),))
    return move.get_future_result(
        future,
        zone_address=move.ZoneAddress(0),
        location_address=move.LocationAddress(7, 0, 0),
    )


def _tiny_arch():
    zone = RustZone(
        name="test",
        grid=RustGrid.from_positions([0.0, 10.0], [0.0, 1.0]),
        site_buses=[SiteBus(src=[0], dst=[1])],
        word_buses=[WordBus(src=[0], dst=[1])],
        words_with_site_buses=[0, 1],
        sites_with_word_buses=[0],
        entangling_pairs=[(0, 1)],
    )
    return ArchSpec.from_components(
        words=(Word(sites=((0, 0), (1, 0))), Word(sites=((0, 1), (1, 1)))),
        zones=(zone,),
        modes=[RustMode(name="all", zones=[0], bitstring_order=[])],
    )


@kernel
def _reads_one_measurement():
    state0 = move.load()
    state1 = move.fill(state0, location_addresses=(move.LocationAddress(0, 0, 0),))
    future = move.end_measure(state1, zone_addresses=(move.ZoneAddress(0),))
    return move.get_future_result(
        future,
        zone_address=move.ZoneAddress(0),
        location_address=move.LocationAddress(0, 0, 0),
    )


def test_a_crashed_analysis_becomes_a_diagnostic(monkeypatch):
    """``get_measurement_positions`` raises when the analysis fails, and
    this method contracts for a diagnostic, so it converts. Before #967
    a crashed run yielded a silently short mapping instead."""
    from bloqade.lanes.analysis.atom import lattice

    def failing_init(self, measurement_id, qubit_id, location_address):
        raise RuntimeError("simulated analysis crash")

    monkeypatch.setattr(lattice.MeasureResult, "__init__", failing_init)

    interp = AtomInterpreter(kernel, arch_spec=_tiny_arch())
    result = interp.get_shot_remapping(_reads_one_measurement)
    assert isinstance(result, ShotRemappingErr)
    assert "did not produce measurement positions" in result.diagnostic.message


def test_a_crashed_analysis_still_raises_when_no_raise_is_false(monkeypatch):
    """``no_raise=False`` is the debugging escape hatch."""
    from bloqade.lanes.analysis.atom import lattice

    def failing_init(self, measurement_id, qubit_id, location_address):
        raise RuntimeError("simulated analysis crash")

    monkeypatch.setattr(lattice.MeasureResult, "__init__", failing_init)

    interp = AtomInterpreter(kernel, arch_spec=_tiny_arch())
    with pytest.raises(RuntimeError, match="simulated analysis crash"):
        interp.get_shot_remapping(_reads_one_measurement, no_raise=False)


def test_unresolvable_location_returns_diagnostic_not_an_exception():
    """Resolving positions raises ``ValueError`` for an address the arch
    spec doesn't know, but this method contracts for a diagnostic — the
    likely cause is an arch spec that doesn't match the compiled
    program, which callers handle rather than crash on."""
    interp = AtomInterpreter(kernel, arch_spec=_tiny_arch())
    result = interp.get_shot_remapping(_fills_an_address_outside_the_arch)
    assert isinstance(result, ShotRemappingErr)
    # The underlying error is preserved in the message, so a compiler
    # developer can still see which address failed to resolve.
    assert "Invalid location address" in result.diagnostic.message


def test_unresolvable_location_still_raises_when_no_raise_is_false():
    """``no_raise=False`` is the debugging escape hatch and must keep
    letting the original exception through."""
    interp = AtomInterpreter(kernel, arch_spec=_tiny_arch())
    with pytest.raises(ValueError, match="Invalid location address"):
        interp.get_shot_remapping(_fills_an_address_outside_the_arch, no_raise=False)


# ── Integration: the reproductions from issue #967 ───────────────


def _mapping_for(kernel_method):
    arch_spec = physical.get_arch_spec()
    physical_move = LogicalPipeline(transversal_rewrite=True).emit(kernel_method)
    interp = AtomInterpreter(physical_move.dialects, arch_spec=arch_spec)
    result = interp.get_shot_remapping(physical_move)
    assert isinstance(
        result, ShotRemappingOk
    ), f"unexpected diagnostic: {getattr(result, 'diagnostic', None)}"
    return result, interp, physical_move


@pytest.mark.slow
def test_full_return_maps_every_record():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        return logical.terminal_measure(q)

    result, _, _ = _mapping_for(main)
    # 2 logical qubits x 7 physical.
    assert len(result.mapping) == 14
    assert len(set(result.mapping)) == 14
    assert all(0 <= idx < result.frame_size for idx in result.mapping)


@pytest.mark.slow
def test_subset_return_still_maps_every_record():
    """Returning one logical block used to yield a 7-entry mapping while
    post-processing read indices 7..13, raising IndexError."""

    @logical.kernel(aggressive_unroll=True)
    def main():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        m = logical.terminal_measure(q)
        return ilist.IList([m[1]])

    subset, _, _ = _mapping_for(main)

    @logical.kernel(aggressive_unroll=True)
    def full():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        return logical.terminal_measure(q)

    every, _, _ = _mapping_for(full)
    assert len(subset.mapping) == 14
    assert subset.mapping == every.mapping


@pytest.mark.slow
def test_permuted_return_does_not_permute_the_mapping():
    """Returning the blocks swapped used to swap the mapping too, so
    post-processing re-applied the permutation and silently returned the
    logical blocks in the wrong order."""

    @logical.kernel(aggressive_unroll=True)
    def permuted():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        m = logical.terminal_measure(q)
        return ilist.IList([m[1], m[0]])

    @logical.kernel(aggressive_unroll=True)
    def full():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        return logical.terminal_measure(q)

    assert _mapping_for(permuted)[0].mapping == _mapping_for(full)[0].mapping


@pytest.mark.slow
def test_permuted_return_round_trips_through_post_processing():
    """The end-to-end silent-corruption case: set only the second
    logical block's frame bits and check the values come back attached
    to the block that was actually measured."""

    @logical.kernel(aggressive_unroll=True)
    def permuted():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        m = logical.terminal_measure(q)
        return ilist.IList([m[1], m[0]])

    result, _, _ = _mapping_for(permuted)
    post_processing = build_post_processing(permuted)

    # mapping[7:] are the frame slots of the second logical block.
    hot = set(result.mapping[7:])
    frame = [[slot in hot for slot in range(result.frame_size)]]
    projected = [[shot[slot] for slot in result.mapping] for shot in frame]

    first, second = next(iter(post_processing.emit_return(projected)))
    assert all(first), "m[1] was measured, so it must come back all True"
    assert not any(second), "m[0] was not measured, so it must be all False"
