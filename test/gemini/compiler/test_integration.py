import math
from test.gemini._squin_to_move_helper import squin_to_move

import pytest
from kirin.dialects import ilist
from bloqade.lanes.utils import check_circuit
from bloqade.lanes.passes import ALAPPlacePass, ASAPPlacePass
from bloqade.lanes.transform import LogicalPipeline, MoveToSquinPhysical
from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.placement import PhysicalPlacementStrategy

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical


@pytest.mark.slow
def test_logical_compilation():
    from bloqade.rewrite.passes import AggressiveUnroll

    # This is a state-vector round-trip test, not a runnable Gemini task:
    # state-vector simulation intentionally has no terminal-measurement impl.
    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def main():
        reg = qubit.qalloc(5)
        squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_y_adj(reg)

    logical_move = LogicalPipeline().emit(main, no_raise=True)
    decompiled_squin = MoveToSquinPhysical(get_arch_spec()).emit(logical_move)

    AggressiveUnroll(main.dialects).fixpoint(main)

    assert check_circuit(main, decompiled_squin)


@pytest.mark.slow
def test_ghz_move_to_squin_roundtrip_state_vector():
    @squin.kernel(typeinfer=True, fold=True)
    def ghz():
        reg = squin.qalloc(7)
        squin.h(reg[0])
        for i in range(1, len(reg)):
            squin.cx(reg[0], reg[i])

    # GHZ is a state-prep circuit with no terminal measurement; use the
    # lower-level squin_to_move API which does not enforce that requirement.
    physical_move = squin_to_move(
        ghz,
        layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        placement_strategy=PalindromePlacementStrategy(
            inner=PhysicalPlacementStrategy()
        ),
        logical_initialize=False,
        no_raise=False,
    )
    roundtrip_squin = MoveToSquinPhysical(get_physical_arch_spec()).emit(
        physical_move, no_raise=False
    )

    assert check_circuit(ghz, roundtrip_squin)


@pytest.mark.slow
def test_asap_place_pass_roundtrip_state_vector():
    """ASAPPlacePass produces a physically equivalent circuit to SequentialPlacePass."""

    @squin.kernel(typeinfer=True, fold=True)
    def circuit():
        reg = squin.qalloc(4)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[0], reg[2])
        squin.cx(reg[0], reg[3])

    physical_move = squin_to_move(
        circuit,
        layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        placement_strategy=PalindromePlacementStrategy(
            inner=PhysicalPlacementStrategy()
        ),
        logical_initialize=False,
        place_opt_type=ASAPPlacePass,
        no_raise=False,
    )
    roundtrip_squin = MoveToSquinPhysical(get_physical_arch_spec()).emit(
        physical_move, no_raise=False
    )

    assert check_circuit(circuit, roundtrip_squin)


@pytest.mark.slow
def test_alap_place_pass_roundtrip_state_vector():
    """ALAPPlacePass produces a physically equivalent circuit to SequentialPlacePass.

    Mirrors the steane_demo main() pattern: a first CX layer followed by a second
    layer applied via a loop over (control, target) pairs.  CX(q1, q3) in the second
    layer depends on CX(q0, q1)'s post-H via q1, while CX(q0, q2) is independent
    of that post-H.  ASAP front-loads the pre-gates for the second layer into SP1's
    qubit footprint; ALAP defers them past CZ(q0,q1), shrinking SP1 to {q0, q1} and
    collapsing the second pair of CZs into a single layer.
    """

    @squin.kernel(typeinfer=True, fold=True)
    def circuit():
        reg = squin.qalloc(4)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        for i in range(2):
            squin.cx(reg[i], reg[i + 2])

    physical_move = squin_to_move(
        circuit,
        layout_heuristic=PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        placement_strategy=PalindromePlacementStrategy(
            inner=PhysicalPlacementStrategy()
        ),
        logical_initialize=False,
        place_opt_type=ALAPPlacePass,
        no_raise=False,
    )
    roundtrip_squin = MoveToSquinPhysical(get_physical_arch_spec()).emit(
        physical_move, no_raise=False
    )

    assert check_circuit(circuit, roundtrip_squin)
