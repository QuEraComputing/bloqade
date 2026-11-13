"""Tests for logical position-based qubit allocation."""

from kirin.dialects import ilist
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome

import bloqade.gemini as gemini
from bloqade.gemini.device import GeminiLogicalSimulator


def test_qalloc_at_mixes_pinned_and_unpinned_allocations():
    """Integer positions pin their corresponding logical qubits; ``None`` does not."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        register = gemini.logical.qalloc_at(ilist.IList([None, 2, None, 4]))
        gemini.logical.terminal_measure(register)

    physical_move = LogicalPipeline(
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)

    fill = next(
        statement
        for statement in physical_move.callable_region.walk()
        if isinstance(statement, move.Fill)
    )

    assert fill.location_addresses[1] == LocationAddress(
        zone_id=0,
        word_id=4,
        site_id=0,
    )
    assert fill.location_addresses[3] == LocationAddress(
        zone_id=0,
        word_id=8,
        site_id=0,
    )
    assert len(set(fill.location_addresses)) == 4


def test_logical_simulator_task_accepts_qalloc_at_kernel():
    """The public simulator API compiles a kernel using the public helper."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        register = gemini.logical.qalloc_at(ilist.IList([0, None]))
        gemini.logical.terminal_measure(register)

    task = GeminiLogicalSimulator().task(kernel)

    assert task.logical_squin_kernel.is_structurally_equal(kernel)
