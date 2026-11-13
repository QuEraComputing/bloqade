"""Tests for the EliminateRz rewrite rule.

Each case builds the input block and the expected output block explicitly and
compares them whole with ``assert_nodes``, so a stray statement the rule leaves
behind fails the test rather than slipping past a spot-check.
"""

import pytest
from kirin import ir, types as kirin_types, rewrite
from kirin.dialects import py, ilist
from bloqade.test_utils import assert_nodes
from bloqade.native.dialects import gate as native_gate
from bloqade.lanes.dialects.steane713 import stmts as operations
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz, EliminateRzError
from bloqade.lanes.dialects.steane713.extensions import stmts as extensions

from bloqade import qubit as squin_qubit, types as bloqade_types

QUBIT = bloqade_types.QubitType


def _rz_is_deleted():
    """A lone Rz is absorbed into the frame and removed. Its angle constant is
    left behind as dead code -- DCE later in the pipeline sweeps it, and this
    rule does not do reference counting."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    test = ir.Block([q, reg, quarter, native_gate.stmts.Rz(quarter.result, reg.result)])

    q2 = squin_qubit.stmts.New()
    reg2 = ilist.New(values=(q2.result,), elem_type=QUBIT)
    expected = ir.Block([q2, reg2, py.Constant(0.25)])
    return test, expected


def _r_axis_shifts_by_the_frame():
    """R(0) after Rz(0.25) becomes R(0 - 0.25). The rule emits a fresh register
    and gate before the original and deletes it, so the new pair lands where the
    old R was."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    q2 = squin_qubit.stmts.New()
    reg2 = ilist.New(values=(q2.result,), elem_type=QUBIT)
    quarter2 = py.Constant(0.25)
    zero2 = py.Constant(0.0)
    shifted = py.Constant(-0.25)
    expected = ir.Block(
        [
            q2,
            reg2,
            quarter2,
            zero2,
            shifted,
            native_gate.stmts.R(shifted.result, quarter2.result, reg2.result),
        ]
    )
    return test, expected


def _frames_accumulate():
    """Two Rz on one qubit sum before shifting: 0.0 - (0.25 + 0.5) = -0.75."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    half = py.Constant(0.5)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            half,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            native_gate.stmts.Rz(half.result, reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    q2 = squin_qubit.stmts.New()
    reg2 = ilist.New(values=(q2.result,), elem_type=QUBIT)
    quarter2 = py.Constant(0.25)
    half2 = py.Constant(0.5)
    zero2 = py.Constant(0.0)
    summed = py.Constant(0.75)
    shifted = py.Constant(-0.75)
    expected = ir.Block(
        [
            q2,
            reg2,
            quarter2,
            half2,
            zero2,
            summed,
            shifted,
            native_gate.stmts.R(shifted.result, quarter2.result, reg2.result),
        ]
    )
    return test, expected


def _untouched_qubit_is_left_alone():
    """An R on a qubit no Rz ever reached keeps its original statement -- the
    rule must not rebuild it."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    reg0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    reg1 = ilist.New(values=(q1.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            reg0,
            reg1,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg0.result),
            native_gate.stmts.R(zero.result, quarter.result, reg1.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    areg0 = ilist.New(values=(a0.result,), elem_type=QUBIT)
    areg1 = ilist.New(values=(a1.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    expected = ir.Block(
        [
            a0,
            a1,
            areg0,
            areg1,
            aquarter,
            azero,
            native_gate.stmts.R(azero.result, aquarter.result, areg1.result),
        ]
    )
    return test, expected


def _r_splits_when_frames_differ():
    """One pulse cannot carry two axis angles. Only q0 saw an Rz, so the shared
    R splits: q0 gets the shifted axis, q1 keeps the original."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    only_q0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    both = ilist.New(values=(q0.result, q1.result), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            only_q0,
            both,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, only_q0.result),
            native_gate.stmts.R(zero.result, quarter.result, both.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    a_only = ilist.New(values=(a0.result,), elem_type=QUBIT)
    a_both = ilist.New(values=(a0.result, a1.result), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    shifted = py.Constant(-0.25)
    shifted_reg = ilist.New(values=(a0.result,), elem_type=QUBIT)
    plain_reg = ilist.New(values=(a1.result,), elem_type=QUBIT)
    expected = ir.Block(
        [
            a0,
            a1,
            a_only,
            a_both,
            aquarter,
            azero,
            shifted,
            shifted_reg,
            native_gate.stmts.R(shifted.result, aquarter.result, shifted_reg.result),
            plain_reg,
            native_gate.stmts.R(azero.result, aquarter.result, plain_reg.result),
        ]
    )
    return test, expected


def _equal_frames_produce_equal_axes():
    """Two R statements shifted by the same frame get the same axis *value*,
    each as its own constant. Deduplicating them is not this rule's job --
    py.Constant is Pure, so the CommonSubexpressionElimination already in
    NativeToPlaceBase.emit merges them, and the fusion this protects is asserted
    end-to-end by test_equal_frames_still_fuse_after_lowering_to_place."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    both = ilist.New(values=(q0.result, q1.result), elem_type=QUBIT)
    reg0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    reg1 = ilist.New(values=(q1.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            both,
            reg0,
            reg1,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, both.result),
            native_gate.stmts.R(zero.result, quarter.result, reg0.result),
            native_gate.stmts.R(zero.result, quarter.result, reg1.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    a_both = ilist.New(values=(a0.result, a1.result), elem_type=QUBIT)
    areg0 = ilist.New(values=(a0.result,), elem_type=QUBIT)
    areg1 = ilist.New(values=(a1.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    shifted0 = py.Constant(-0.25)
    shifted1 = py.Constant(-0.25)
    expected = ir.Block(
        [
            a0,
            a1,
            a_both,
            areg0,
            areg1,
            aquarter,
            azero,
            shifted0,
            native_gate.stmts.R(shifted0.result, aquarter.result, areg0.result),
            shifted1,
            native_gate.stmts.R(shifted1.result, aquarter.result, areg1.result),
        ]
    )
    return test, expected


def _cz_passes_through():
    """CZ is diagonal, so the frame commutes past it and it is left alone."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    reg0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    reg1 = ilist.New(values=(q1.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    test = ir.Block(
        [
            q0,
            q1,
            reg0,
            reg1,
            quarter,
            native_gate.stmts.Rz(quarter.result, reg0.result),
            native_gate.stmts.CZ(reg0.result, reg1.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    areg0 = ilist.New(values=(a0.result,), elem_type=QUBIT)
    areg1 = ilist.New(values=(a1.result,), elem_type=QUBIT)
    expected = ir.Block(
        [
            a0,
            a1,
            areg0,
            areg1,
            py.Constant(0.25),
            native_gate.stmts.CZ(areg0.result, areg1.result),
        ]
    )
    return test, expected


def _measurement_discards_the_frame():
    """The residual is diagonal and the readout is in the Z basis, so the
    measurement drops it and a later R on that qubit is unshifted."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            operations.TerminalLogicalMeasurement(reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    expected = ir.Block(
        [
            a,
            areg,
            aquarter,
            azero,
            operations.TerminalLogicalMeasurement(areg.result),
            native_gate.stmts.R(azero.result, aquarter.result, areg.result),
        ]
    )
    return test, expected


def _non_constant_angle_stays_symbolic():
    """A parametrized angle has no literal to fold, so the shift is emitted as
    py.Sub over the SSA values -- this is why the frame holds SSA values rather
    than floats."""
    theta = ir.TestValue(type=kirin_types.Float)
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    zero = py.Constant(0.0)
    quarter = py.Constant(0.25)
    test = ir.Block(
        [
            q,
            reg,
            zero,
            quarter,
            native_gate.stmts.Rz(theta, reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    azero = py.Constant(0.0)
    aquarter = py.Constant(0.25)
    sub = py.Sub(azero.result, theta)
    expected = ir.Block(
        [
            a,
            areg,
            azero,
            aquarter,
            sub,
            native_gate.stmts.R(sub.result, aquarter.result, areg.result),
        ]
    )
    return test, expected


def _squin_measure_discards_the_frame():
    """The mid-circuit measurement the handler's comment argues about.

    ``squin.qubit.Measure`` is co-registered with the terminal measurement on
    the strength of a claim in that handler -- "safe for a mid-circuit
    measurement too [...] a Z measurement leaves the qubit in a computational
    basis state, where any leftover Z phase is a global phase on that branch".
    Nothing exercised it: the real pipeline never emits one (confirmed by
    instrumenting the rule across the gemini and integration suites), so the
    registration could be deleted with every test still green, after which a
    kernel that does measure mid-circuit would hit the raising default.
    """
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            squin_qubit.stmts.Measure(reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    expected = ir.Block(
        [
            a,
            areg,
            aquarter,
            azero,
            squin_qubit.stmts.Measure(areg.result),
            native_gate.stmts.R(azero.result, aquarter.result, areg.result),
        ]
    )
    return test, expected


def _split_preserves_the_source_elem_type():
    """Rebuilt registers inherit the element type of the one they replace.

    The real post-unroll IR mixes these: a two-qubit register arrives as
    ``!Any`` while a one-qubit register is ``!py.Qubit``. Hardcoding
    ``QubitType`` here would silently re-type every fragment of an ``!Any``
    register, leaving TypeInfer to re-derive what the rewrite already knew.
    """
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    only_q0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    both = ilist.New(values=(q0.result, q1.result), elem_type=kirin_types.Any)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            only_q0,
            both,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, only_q0.result),
            native_gate.stmts.R(zero.result, quarter.result, both.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    a_only = ilist.New(values=(a0.result,), elem_type=QUBIT)
    a_both = ilist.New(values=(a0.result, a1.result), elem_type=kirin_types.Any)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    shifted = py.Constant(-0.25)
    shifted_reg = ilist.New(values=(a0.result,), elem_type=kirin_types.Any)
    plain_reg = ilist.New(values=(a1.result,), elem_type=kirin_types.Any)
    expected = ir.Block(
        [
            a0,
            a1,
            a_only,
            a_both,
            aquarter,
            azero,
            shifted,
            shifted_reg,
            native_gate.stmts.R(shifted.result, aquarter.result, shifted_reg.result),
            plain_reg,
            native_gate.stmts.R(azero.result, aquarter.result, plain_reg.result),
        ]
    )
    return test, expected


CASES = {
    "rz_is_deleted": _rz_is_deleted,
    "split_preserves_the_source_elem_type": _split_preserves_the_source_elem_type,
    "r_axis_shifts_by_the_frame": _r_axis_shifts_by_the_frame,
    "frames_accumulate": _frames_accumulate,
    "untouched_qubit_is_left_alone": _untouched_qubit_is_left_alone,
    "r_splits_when_frames_differ": _r_splits_when_frames_differ,
    "equal_frames_produce_equal_axes": _equal_frames_produce_equal_axes,
    "cz_passes_through": _cz_passes_through,
    "measurement_discards_the_frame": _measurement_discards_the_frame,
    "squin_measure_discards_the_frame": _squin_measure_discards_the_frame,
    "non_constant_angle_stays_symbolic": _non_constant_angle_stays_symbolic,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_rewrite(case):
    test_block, expected_block = CASES[case]()
    rewrite.Walk(EliminateRz()).rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_rewrite_is_idempotent():
    """A second walk finds no Rz and reports no change."""
    test_block, _ = _r_axis_shifts_by_the_frame()
    rewrite.Walk(EliminateRz()).rewrite(test_block)
    assert not rewrite.Walk(EliminateRz()).rewrite(test_block).has_done_something


REGISTERED_STATEMENTS = {
    "bloqade.decoders.dialects.annotate.stmts.SetDetector",
    "bloqade.decoders.dialects.annotate.stmts.SetObservable",
    "bloqade.lanes.dialects.arch.stmts.NewAt",
    "bloqade.lanes.dialects.steane713.extensions.stmts.StarRz",
    "bloqade.lanes.dialects.steane713.stmts.Initialize",
    "bloqade.lanes.dialects.steane713.stmts.TerminalLogicalMeasurement",
    "bloqade.native.dialects.gate.stmts.CZ",
    "bloqade.native.dialects.gate.stmts.R",
    "bloqade.native.dialects.gate.stmts.Rz",
    "bloqade.qubit.stmts.Measure",
    "bloqade.qubit.stmts.New",
    "kirin.dialects.func.stmts.ConstantNone",
    "kirin.dialects.func.stmts.Function",
    "kirin.dialects.func.stmts.Return",
    "kirin.dialects.ilist.stmts.New",
    "kirin.dialects.py.binop.stmts.BinOp",
    "kirin.dialects.py.constant.Constant",
    "kirin.dialects.py.indexing.GetItem",
    "kirin.dialects.py.tuple.New",
}


def test_dispatch_table_is_what_it_claims_to_be():
    """Pin every registration, because deleting one is currently invisible.

    The table is load-bearing -- the default raises, so an unregistered type
    is a hard failure -- but most entries are reached by no test, and several
    only by kernels outside this file. Removing one would leave the suite
    green and fail later on a program nobody ran here.

    A new entry is not a problem to be silenced: update this set and say in
    the commit why that statement is phase-neutral.
    """
    registry = EliminateRz.__dict__["_rewrite"].dispatcher.registry
    registered = {
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in registry
        if cls is not object  # the raising default
    }

    assert registered == REGISTERED_STATEMENTS


# ---------------------------------------------------------------------------
# Preconditions. Dispatch is exhaustive, so anything unregistered is an error
# rather than a silent pass -- that is what keeps an Rz from being left behind.
# ---------------------------------------------------------------------------


def test_unregistered_statement_raises():
    """An unrecognised statement is one whose phase behaviour is unknown, so it
    is an error rather than a guess."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    block = ir.Block([q, reg, ilist.Push(reg.result, q.result)])
    with pytest.raises(EliminateRzError, match="no EliminateRz handler"):
        rewrite.Walk(EliminateRz()).rewrite(block)


def _constant_angle(block: ir.Block, value: float) -> ir.SSAValue:
    block.stmts.append(constant := py.Constant(value))
    return constant.result


def _symbolic_angle(block: ir.Block, _: float) -> ir.SSAValue:
    """An angle that arrives as a function argument, so it has no literal."""
    return block.args.append_from(kirin_types.Float, name="angle")


def _initialize_after_a_phase(make_angle, value: float = 0.25) -> ir.Block:
    """A mid-wire Initialize: invalid, and what the guard below exists to catch."""
    block = ir.Block()
    angle = make_angle(block, value)
    block.stmts.append(q := squin_qubit.stmts.New())
    block.stmts.append(reg := ilist.New(values=(q.result,), elem_type=QUBIT))
    block.stmts.append(zero := py.Constant(0.0))
    block.stmts.append(native_gate.stmts.Rz(angle, reg.result))
    block.stmts.append(
        operations.Initialize(zero.result, zero.result, zero.result, reg.result)
    )
    return block


@pytest.mark.parametrize(
    ("make_angle", "value"),
    [
        pytest.param(_constant_angle, 0.25, id="constant"),
        # Regression: the guard used to compare `constant_float(pending) != 0.0`, and
        # a non-constant has no literal, so an angle arriving as a function
        # argument raised here -- the very case the SSA-valued frame exists to
        # support.
        pytest.param(_symbolic_angle, 0.25, id="symbolic"),
        # Regression: two Z gates accumulate to exactly one turn, the identity,
        # and the old comparison against 0.0 rejected that too.
        pytest.param(_constant_angle, 1.0, id="identity"),
    ],
)
def test_initialize_with_a_pending_frame_raises(make_angle, value):
    """Initialize sits at the head of a wire; a mid-wire one would drop a phase.

    There is no logical reset statement, and ``_RewriteU3ToInitialize``
    documents that it assumes no later gate touches the qubit -- so a pending
    phase here is evidence that an upstream rewrite broke that assumption.
    Raising says so; popping the frame would quietly accept it.
    """
    with pytest.raises(EliminateRzError, match="head of a wire"):
        rewrite.Walk(EliminateRz()).rewrite(
            _initialize_after_a_phase(make_angle, value)
        )


def test_initialize_with_a_pending_frame_drains_under_no_raise():
    block = _initialize_after_a_phase(_constant_angle)

    rewrite.Walk(EliminateRz(no_raise=True)).rewrite(block)

    restored = [s for s in block.stmts if isinstance(s, native_gate.stmts.Rz)]
    assert len(restored) == 1, "the pending phase comes back rather than vanishing"


# ---------------------------------------------------------------------------
# Best-effort mode. Under `no_raise` a precondition failure drains the pending
# frame back into the IR as real `Rz` statements instead of raising. That is
# exact rather than merely conservative: the scan pushes each `Rz` *later* in
# the circuit, so a pending `alpha` means the emitted program followed by
# `Rz(alpha)` equals the original, and writing it back restores the original.
#
# Skipping the statement *without* draining is the failure mode these tests
# exist to rule out -- it leaves the program half commuted, structurally valid,
# and silently wrong.
# ---------------------------------------------------------------------------


def test_unregistered_statement_drains_the_frame_under_no_raise():
    """The absorbed Rz is written back out, not dropped."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    unknown = ilist.Push(reg.result, q.result)
    block = ir.Block(
        [q, reg, quarter, native_gate.stmts.Rz(quarter.result, reg.result), unknown]
    )

    rewrite.Walk(EliminateRz(no_raise=True)).rewrite(block)

    stmts = list(block.stmts)
    restored = [stmt for stmt in stmts if isinstance(stmt, native_gate.stmts.Rz)]
    assert len(restored) == 1, "the pending phase must come back as an Rz"
    assert restored[0].rotation_angle is quarter.result, "same angle, same SSA value"
    assert stmts.index(restored[0]) < stmts.index(unknown), "drained before giving up"


def test_drained_frame_is_not_reapplied_to_later_gates():
    """After a drain the frame is empty, which is the state the scan starts in.

    A later ``R`` must therefore keep its original axis: re-applying the phase
    that was just written back out would double-count it.
    """
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    axis = py.Constant(0.0)
    turn = py.Constant(0.5)
    later = native_gate.stmts.R(axis.result, turn.result, reg.result)
    block = ir.Block(
        [
            q,
            reg,
            quarter,
            axis,
            turn,
            native_gate.stmts.Rz(quarter.result, reg.result),
            ilist.Push(reg.result, q.result),
            later,
        ]
    )

    rewrite.Walk(EliminateRz(no_raise=True)).rewrite(block)

    assert later.axis_angle is axis.result


def test_star_rz_is_left_alone_silently(recwarn):
    """STAR puts an Rz on a subset of a block's physical qubits, which this
    rule's per-logical-qubit frame does not model. It passes through -- the
    commutation still holds -- but the phase reaches the backend.

    No warning is emitted for it. The condition is static and the caller
    cannot act on it, so it belongs in the class docstring; warning instead
    would fire once per gadget invocation after unrolling, and with no
    dedicated category, so silencing it would mean silencing every
    ``UserWarning`` the compiler emits.
    """
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    theta = py.Constant(0.03)
    star = extensions.StarRz(theta.result, reg.result)
    block = ir.Block([q, reg, theta, star])

    rewrite.Walk(EliminateRz()).rewrite(block)

    assert star in list(block.stmts)
    assert not recwarn.list


@pytest.mark.parametrize(
    ("with_pending_frame", "expected"),
    [
        pytest.param(False, False, id="empty_frame_changes_nothing"),
        pytest.param(True, True, id="pending_frame_is_written_back"),
    ],
)
def test_giving_up_reports_whether_it_actually_changed_the_ir(
    with_pending_frame, expected
):
    """`has_done_something` must describe the IR, not the decision to give up.

    Draining emits one `Rz` per pending phase, so it emits nothing at all when
    the frame is empty -- the common case, since the scan usually meets
    something it cannot handle before absorbing any `Rz`. Claiming a change
    there would tell a `Fixpoint` driver to go round again on an untouched
    program.
    """
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    stmts = [q, reg, quarter]
    if with_pending_frame:
        stmts.append(native_gate.stmts.Rz(quarter.result, reg.result))
    stmts.append(ilist.Push(reg.result, q.result))  # unregistered: forces a give-up
    block = ir.Block(stmts)

    before = len(list(block.stmts))
    result = rewrite.Walk(EliminateRz(no_raise=True)).rewrite(block)
    after = len(list(block.stmts))

    assert result.has_done_something is expected
    assert (after != before) is expected, "reported change must match reality"
