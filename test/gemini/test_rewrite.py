from kirin import ir, types, rewrite
from bloqade.types import MeasurementResultType
from kirin.dialects import py, func, ilist
from bloqade.test_utils import assert_nodes
from bloqade.squin.gate.stmts import U3, S, SqrtX
from bloqade.lanes.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.lanes.dialects.steane713.stmts import (
    Initialize,
    TerminalLogicalMeasurement,
)
from bloqade.decoders.dialects.annotate.stmts import SetDetector
from bloqade.lanes.rewrite.steane_transversal import (
    RewriteSteaneTransversalCliffordAdjoints,
)
from bloqade.lanes.rewrite.remove_postprocessing import RemovePostProcessing

from bloqade import squin
from bloqade.gemini import logical

TerminalMeasureRetType = ilist.IListType[
    ilist.IListType[MeasurementResultType, types.Any], types.Any
]


def test_rewrite_u3_to_initialize():
    theta = ir.TestValue()
    phi = ir.TestValue()
    qubits = ir.TestValue()
    test_block = ir.Block(
        [
            lam_stmt := py.Constant(1.0),
            U3(theta, phi, lam_stmt.result, qubits),
        ]
    )

    expected_block = ir.Block(
        [
            lam_stmt := py.Constant(1.0),
            Initialize(theta, phi, lam_stmt.result, qubits),
        ]
    )

    rewrite.Walk(_RewriteU3ToInitialize()).rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_rewrite_steane_transversal_clifford_adjoints():
    qubits = ir.TestValue()
    test_block = ir.Block(
        [
            SqrtX(qubits),
            SqrtX(qubits, adjoint=True),
            S(qubits),
            S(qubits, adjoint=True),
        ]
    )

    expected_block = ir.Block(
        [
            SqrtX(qubits, adjoint=True),
            SqrtX(qubits),
            S(qubits, adjoint=True),
            S(qubits),
        ]
    )

    result = rewrite.Walk(RewriteSteaneTransversalCliffordAdjoints()).rewrite(
        test_block
    )

    assert result.has_done_something
    assert_nodes(test_block, expected_block)


def test_remove_postprocessing():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        squin.broadcast.h(q)

        logical.terminal_measure(q)

    result = RemovePostProcessing(main.dialects)(main)

    assert result.has_done_something

    # check that calling twice doesn't do anything
    result = RemovePostProcessing(main.dialects)(main)
    assert not result.has_done_something

    assert main.return_type.is_subseteq(TerminalMeasureRetType)


def test_remove_postprocessing_with_uses():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        m = logical.terminal_measure(q)
        det = squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])
        return det

    # check that we have a detector there
    assert any(isinstance(stmt, SetDetector) for stmt in main.callable_region.stmts())

    result = RemovePostProcessing(main.dialects)(main)
    assert result.has_done_something

    assert main.return_type.is_subseteq(TerminalMeasureRetType)

    # check that calling twice doesn't do anything
    result = RemovePostProcessing(main.dialects)(main)
    assert not result.has_done_something

    assert not any(
        isinstance(stmt, SetDetector) for stmt in main.callable_region.stmts()
    )


def test_remove_postprocessing_and_terminal_measure():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        m = logical.terminal_measure(q)
        det = squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])
        return det

    # check that we have a detector there
    assert any(
        isinstance(stmt, (SetDetector, TerminalLogicalMeasurement))
        for stmt in main.callable_region.stmts()
    )

    result = RemovePostProcessing(main.dialects, delete_terminal_measure=True)(main)
    assert result.has_done_something
    # check if type is updated
    assert main.return_type.is_subseteq(types.NoneType)

    # check that calling twice doesn't do anything
    result = RemovePostProcessing(main.dialects, delete_terminal_measure=True)(main)
    assert not result.has_done_something
    assert (last_stmt := main.callable_region.blocks[-1].last_stmt) is not None
    assert isinstance(last_stmt, func.Return)
    assert isinstance(last_stmt.prev_stmt, func.ConstantNone)
    assert not any(
        isinstance(stmt, (SetDetector, TerminalLogicalMeasurement))
        for stmt in main.callable_region.stmts()
    )


def test_remove_postprocessing_and_terminal_measure_2():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        m = logical.terminal_measure(q)
        return m

    # rewrite removes terminal measurement
    result = RemovePostProcessing(main.dialects, delete_terminal_measure=True)(main)
    assert result.has_done_something

    assert not any(
        isinstance(stmt, TerminalLogicalMeasurement)
        for stmt in main.callable_region.stmts()
    )

    assert main.return_type.is_subseteq(types.NoneType)

    # check that calling twice doesn't do anything
    result = RemovePostProcessing(main.dialects, delete_terminal_measure=True)(main)
    assert not result.has_done_something
    assert (last_stmt := main.callable_region.blocks[-1].last_stmt) is not None
    assert isinstance(last_stmt, func.Return)
    assert isinstance(last_stmt.prev_stmt, func.ConstantNone)

    assert not any(
        isinstance(stmt, TerminalLogicalMeasurement)
        for stmt in main.callable_region.stmts()
    )


def test_remove_postprocessing_and_terminal_measure_3():
    # This fixture deliberately has no terminal measurement: the rewrite must
    # also be a no-op for an incomplete program constructed with verification
    # disabled.
    @logical.kernel(aggressive_unroll=True, verify=False)
    def main():
        squin.qalloc(2)

    # rewrite doesn't do anything
    result = RemovePostProcessing(main.dialects, delete_terminal_measure=True)(main)
    assert not result.has_done_something
    assert main.return_type.is_subseteq(types.NoneType)
