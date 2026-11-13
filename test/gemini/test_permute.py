"""Tests for the movement-dialect permute(qubits, perm) feature."""

import bloqade.squin as squin
from kirin import ir, types, rewrite, lowering
from kirin.prelude import structural_no_opt
from kirin.analysis import const
from kirin.dialects import py, func, ilist
from bloqade.lanes.dialects import move as move_dialect, place, arrange as arrange
from bloqade.lanes.transform import PhysicalPipeline
from bloqade.lanes.dialects.arrange import dialect as arrange_dialect
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.validation.move_to import MoveToValidation
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.rewrite.circuit2place import RewritePlaceOperations
from bloqade.lanes.dialects.arrange.stmts import Permute
from bloqade.lanes.analysis.placement.strategy import _resolve_permute_locations

from bloqade.gemini import physical


def test_permute_statement_shape():
    assert issubclass(Permute, ir.Statement)
    assert Permute.name == "permute"
    # Lowered from a Python call; NOT pure (it mutates placement state).
    assert any(isinstance(t, lowering.FromPythonCall) for t in Permute.traits)
    assert not any(isinstance(t, ir.Pure) for t in Permute.traits)
    # Registered on the arrange dialect — catches wrong-dialect registration.
    assert Permute in arrange.dialect.stmts


def test_permute_callable_lowering_wrapper():
    # arrange.permute must be callable (it is the lowering wrapper for Permute).
    # The import itself exercises the re-export; this asserts the wrapper is usable.
    assert callable(arrange.permute)


def _loc(w, s):
    return LocationAddress(zone_id=0, word_id=w, site_id=s)


def test_place_permute_statement_shape():
    stmt = place.Permute(ir.TestValue(), qubits=(0, 1, 2), perm=(1, 2, 0))
    assert stmt.qubits == (0, 1, 2)
    assert stmt.perm == (1, 2, 0)


def test_resolve_permute_locations_cycle():
    layout = (_loc(0, 0), _loc(1, 0), _loc(2, 0))
    locations = _resolve_permute_locations(
        layout, qubits=(0, 1, 2), permutation=(1, 2, 0)
    )
    assert locations == (_loc(1, 0), _loc(2, 0), _loc(0, 0))


def test_resolve_permute_locations_identity():
    layout = (_loc(0, 0), _loc(1, 0))
    locations = _resolve_permute_locations(layout, qubits=(0, 1), permutation=(0, 1))
    assert locations == (_loc(0, 0), _loc(1, 0))


def test_resolve_permute_locations_remapped_indices():
    # qubits are global layout indices (StaticPlacement merging remaps them);
    # permutation indexes positions within the qubits tuple.
    layout = (_loc(9, 0), _loc(0, 0), _loc(1, 0), _loc(2, 0))
    locations = _resolve_permute_locations(
        layout, qubits=(1, 2, 3), permutation=(2, 0, 1)
    )
    assert locations == (_loc(2, 0), _loc(0, 0), _loc(1, 0))


def test_rewrite_permute_produces_place_permute():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    qubits_new = ilist.New(values=(q0, q1, q2))

    # perm [1, 2, 0] as a const-hinted ilist of ints
    p0, p1, p2 = py.Constant(1), py.Constant(2), py.Constant(0)
    perm_new = ilist.New(values=(p0.result, p1.result, p2.result))
    perm_new.result.hints["const"] = const.Value((1, 2, 0))

    stmt = arrange.stmts.Permute(qubits_new.result, perm_new.result)

    # Only statements go in the block; q0/q1/q2 are SSA operands (TestValues).
    block = ir.Block([qubits_new, p0, p1, p2, perm_new, stmt])
    region = ir.Region([block])

    rewrite.Walk(RewritePlaceOperations()).rewrite(region)

    permutes = [s for s in region.walk() if isinstance(s, place.Permute)]
    assert len(permutes) == 1
    assert permutes[0].qubits == (0, 1, 2)
    assert permutes[0].perm == (1, 2, 0)
    assert not any(isinstance(s, arrange.stmts.Permute) for s in region.walk())


# ---------------------------------------------------------------------------
# Validation tests (P1–P4)
# ---------------------------------------------------------------------------

_VAL_DIALECTS = structural_no_opt.union([arrange_dialect])


def _build_method(stmts):
    block = ir.Block(list(stmts) + [func.Return()])
    block.args.append_from(types.MethodType, name="self")
    region = ir.Region([block])
    fn = func.Function(
        sym_name="test_fn",
        signature=func.Signature(inputs=(), output=types.NoneType),
        body=region,
    )
    return ir.Method(
        dialects=_VAL_DIALECTS, sym_name="test_fn", arg_names=["self"], code=fn
    )


def _permute_stmts(qubit_vals, perm_data, *, stamp_const=True):
    qubits_new = ilist.New(values=tuple(qubit_vals))
    perm_consts = [py.Constant(p) for p in perm_data]
    perm_new = ilist.New(values=tuple(c.result for c in perm_consts))
    if stamp_const:
        perm_new.result.hints["const"] = const.Value(tuple(perm_data))
    stmt = arrange.stmts.Permute(qubits_new.result, perm_new.result)
    # Only statements; qubit_vals are SSA operands (TestValues), not statements.
    return [qubits_new, *perm_consts, perm_new, stmt]


def _errors(stmts):
    _, errs = MoveToValidation(arch_spec=get_arch_spec()).run(_build_method(stmts))
    return errs


def test_permute_valid_passes():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    assert _errors(_permute_stmts([q0, q1, q2], [1, 2, 0])) == []


def test_permute_p1_non_const_perm_rejected():
    q0, q1 = ir.TestValue(), ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q1], [1, 0], stamp_const=False))) >= 1


def test_permute_p2_length_mismatch_rejected():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q1, q2], [1, 0]))) >= 1


def test_permute_p3_not_a_bijection_rejected():
    q0, q1, q2 = ir.TestValue(), ir.TestValue(), ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q1, q2], [0, 0, 1]))) >= 1  # duplicate index
    assert len(_errors(_permute_stmts([q0, q1, q2], [0, 1, 3]))) >= 1  # out of range


def test_permute_p4_duplicate_qubit_rejected():
    q0 = ir.TestValue()
    assert len(_errors(_permute_stmts([q0, q0], [0, 1]))) >= 1


# ---------------------------------------------------------------------------
# End-to-end integration tests (physical pipeline)
# ---------------------------------------------------------------------------


def test_permute_identity_compiles():
    """An identity permutation is a valid no-op user-movement before a CZ."""

    @physical.kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        arrange.permute([q[0], q[1]], [0, 1])
        squin.cz(q[0], q[1])

    assert k is not None


def test_permute_then_cz_compiles():
    """permute (relabel-only, the default) followed by a CZ lowers without error.

    The default permute is a pure relabel — no atoms move and no routing is
    involved — so it always builds regardless of the permutation (a cycle here).
    """

    @physical.kernel(verify=False)
    def k():
        q = squin.qalloc(3)
        arrange.permute([q[0], q[1], q[2]], [1, 2, 0])
        squin.cz(q[0], q[1])

    assert k is not None


def test_permute_rejected_on_plain_logical_kernel():
    """A plain logical kernel (no arrange dialect) cannot lower permute."""
    import pytest
    from kirin.lowering.exception import BuildError

    from bloqade.gemini import logical

    with pytest.raises(BuildError, match="unsupported dialect"):

        @logical.kernel(aggressive_unroll=True)
        def k():
            q = squin.qalloc(2)
            arrange.permute([q[0], q[1]], [1, 0])
            squin.cz(q[0], q[1])


def test_permute_default_relabel_full_pipeline_emits_no_moves():
    """The default permute (relabel-only) compiled through the full pipeline
    lowers to *no* move IR — it is a free relabel — and leaves no residual
    place/movement Permute. Runs under the palindrome strategy, where
    relabel-only is allowed."""

    @physical.kernel(verify=False)
    def k():
        q = squin.qalloc(3)
        arrange.permute([q[0], q[1], q[2]], [1, 2, 0])

    strat = make_physical_placement_strategy(return_moves=True)  # palindrome
    out = PhysicalPipeline(placement_strategy=strat).emit(k)

    stmts = list(out.callable_region.walk())
    assert not any(isinstance(s, place.Permute) for s in stmts)
    assert not any(isinstance(s, arrange.stmts.Permute) for s in stmts)
    # relabel-only is free — the permute emits no move-dialect Move.
    assert not any(isinstance(s, move_dialect.Move) for s in stmts)


# ---------------------------------------------------------------------------
# insert_moves=True — commit the physical permutation (active)
# ---------------------------------------------------------------------------


def test_place_permute_insert_moves_attribute_defaults_false():
    assert (
        place.Permute(ir.TestValue(), qubits=(0, 1), perm=(1, 0)).insert_moves is False
    )
    stmt = place.Permute(ir.TestValue(), qubits=(0, 1), perm=(1, 0), insert_moves=True)
    assert stmt.insert_moves is True


def test_rewrite_permute_propagates_insert_moves():
    q0, q1 = ir.TestValue(), ir.TestValue()
    qubits_new = ilist.New(values=(q0, q1))
    p0, p1 = py.Constant(1), py.Constant(0)
    perm_new = ilist.New(values=(p0.result, p1.result))
    perm_new.result.hints["const"] = const.Value((1, 0))

    stmt = arrange.stmts.Permute(qubits_new.result, perm_new.result, insert_moves=True)
    block = ir.Block([qubits_new, p0, p1, perm_new, stmt])
    region = ir.Region([block])
    rewrite.Walk(RewritePlaceOperations()).rewrite(region)

    permutes = [s for s in region.walk() if isinstance(s, place.Permute)]
    assert len(permutes) == 1
    assert permutes[0].insert_moves is True


def test_permute_placements_relabel_only_vs_insert_moves():
    """Default (relabel-only): permuted layout, no moves, plain ConcreteState.
    insert_moves=True: committed physical moves with the layout pinned to the
    pre-permute layout (a Permuted state). Both leave qubit ``i`` referring to
    what was qubit ``perm[i]``."""
    from bloqade.lanes.analysis.placement import Permuted, ConcreteState
    from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategy

    strat = LogicalPlacementStrategy(arch_spec=get_arch_spec())
    layout = (_loc(0, 0), _loc(2, 0), _loc(4, 0))
    state = ConcreteState(occupied=frozenset(), layout=layout, move_count=(0, 0, 0))
    perm = (2, 1, 0)

    # Default: relabel only — no atoms move, layout is permuted, no move layers.
    relabel = strat.permute_placements(state, (0, 1, 2), perm, insert_moves=False)
    assert type(relabel) is ConcreteState  # neither Permuted nor UserMoved
    assert relabel.layout == (_loc(4, 0), _loc(2, 0), _loc(0, 0))  # permuted refs
    assert relabel.get_move_layers() == ()  # free

    # insert_moves=True: commit the physical permutation, pin the layout.
    active = strat.permute_placements(state, (0, 1, 2), perm, insert_moves=True)
    assert isinstance(active, Permuted)
    assert active.layout == layout  # pinned to the pre-permute layout
    assert active.get_move_layers()  # committed physical moves
    assert active.get_reverse_moves() == ()  # not palindrome-returned


def test_permute_insert_moves_full_pipeline_emits_moves():
    """insert_moves=True under a non-palindrome (no-return) strategy commits the
    permutation to move IR, with no residual place/movement Permute."""

    @physical.kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        arrange.permute([q[0], q[1]], [1, 0], insert_moves=True)
        squin.cz(q[0], q[1])

    strat = make_physical_placement_strategy(return_moves=False)  # no-return
    out = PhysicalPipeline(placement_strategy=strat).emit(k)

    stmts = list(out.callable_region.walk())
    assert not any(isinstance(s, place.Permute) for s in stmts)
    assert not any(isinstance(s, arrange.stmts.Permute) for s in stmts)
    assert any(isinstance(s, move_dialect.Move) for s in stmts)


def test_permute_insert_moves_survives_asap_place_pass():
    """Regression: the ASAP/ALAP place passes merge & reorder StaticPlacement
    blocks by *reconstructing* each inner statement's attributes
    (HoistNewQubitsUp / MergeStaticPlacement). That reconstruction rebuilt
    place.Permute with only ``qubits``/``perm`` and dropped ``insert_moves``, so a
    permute compiled with ``place_opt_type=ASAPPlacePass`` crashed at placement
    analysis with ``KeyError('insert_moves')``. The default pipeline place pass
    never merges/hoists a Permute, so this path was uncovered."""
    from bloqade.lanes.passes import ASAPPlacePass

    @physical.kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        arrange.permute([q[0], q[1]], [1, 0], insert_moves=True)
        squin.cz(q[0], q[1])

    strat = make_physical_placement_strategy(return_moves=False)  # no-return
    out = PhysicalPipeline(placement_strategy=strat, place_opt_type=ASAPPlacePass).emit(
        k
    )

    stmts = list(out.callable_region.walk())
    assert not any(isinstance(s, place.Permute) for s in stmts)
    assert not any(isinstance(s, arrange.stmts.Permute) for s in stmts)
    assert any(isinstance(s, move_dialect.Move) for s in stmts)


def test_permuted_state_is_measurable_but_user_move_is_not():
    """The committed Permuted state is measurable, whereas a palindrome-pending
    UserMoved is rejected at a terminal measure — Permuted is deliberately not a
    UserMoved, so palindrome never picks it up."""
    import pytest
    from bloqade.lanes.analysis.placement import (
        Permuted,
        UserMoved,
        ExecuteMeasure,
        PlacementError,
        PalindromePlacementStrategy,
    )
    from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategy

    strat = PalindromePlacementStrategy(
        inner=LogicalPlacementStrategy(arch_spec=get_arch_spec())
    )
    layout = (_loc(0, 0), _loc(2, 0))

    permuted = Permuted(
        occupied=frozenset(), layout=layout, move_count=(0, 0), move_layers=()
    )
    assert isinstance(strat.measure_placements(permuted, (0, 1)), ExecuteMeasure)

    user_moved = UserMoved(
        occupied=frozenset(),
        layout=layout,
        move_count=(0, 0),
        move_layers=(),
        accumulated_move_layers=(),
        pre_user_layout=layout,
    )
    with pytest.raises(PlacementError, match="pending user-directed move"):
        strat.measure_placements(user_moved, (0, 1))


def test_permute_insert_moves_rejected_under_palindrome():
    """PalindromePlacementStrategy rejects insert_moves=True (a committed
    permutation is incompatible with returning every move to the pre-move home),
    while the default relabel-only permute is still allowed."""
    import pytest
    from bloqade.lanes.analysis.placement import (
        ConcreteState,
        PalindromePlacementStrategy,
    )
    from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategy

    strat = PalindromePlacementStrategy(
        inner=LogicalPlacementStrategy(arch_spec=get_arch_spec())
    )
    layout = (_loc(0, 0), _loc(2, 0))
    state = ConcreteState(occupied=frozenset(), layout=layout, move_count=(0, 0))

    with pytest.raises(NotImplementedError, match="insert_moves=True"):
        strat.permute_placements(state, (0, 1), (1, 0), insert_moves=True)

    # relabel-only is fine under palindrome: permuted references, no moves.
    relabel = strat.permute_placements(state, (0, 1), (1, 0), insert_moves=False)
    assert isinstance(relabel, ConcreteState)
    assert relabel.layout == (_loc(2, 0), _loc(0, 0))
