import cirq
import cirq.circuits
from kirin import ir
from bloqade import qasm2
from kirin.rewrite import cse, walk, fixpoint
from bloqade.qasm2.rewrite.native_gates import RydbergGateSetRewriteRule


def test_generate_2q_ctrl_gate_stmts():

    q = [cirq.LineQubit(i) for i in range(2)]

    qubits_ssa = [
        ir.TestValue(type=qasm2.QubitType),
        ir.TestValue(type=qasm2.QubitType),
    ]

    stmts = RydbergGateSetRewriteRule(qasm2.main)._generate_2q_ctrl_gate_stmts(
        cirq.CX(*q), qubits_ssa
    )

    block = ir.Block(stmts=stmts)

    fixpoint.Fixpoint(walk.Walk(cse.CommonSubexpressionElimination())).rewrite(block)

    expected_stmts = [
        (s0 := qasm2.expr.ConstFloat(value=1.5707963267948966)),
        (s1 := qasm2.expr.ConstFloat(value=3.141592653589793)),
        (qasm2.uop.UGate(qubits_ssa[1], s0.result, s1.result, s1.result)),
        (qasm2.uop.CZ(qubits_ssa[0], qubits_ssa[1])),
        (s4 := qasm2.expr.ConstFloat(value=6.283185307179586)),
        (s5 := qasm2.expr.ConstFloat(value=0.0)),
        (qasm2.uop.UGate(qubits_ssa[1], s0.result, s4.result, s5.result)),
    ]

    expected_block = ir.Block(stmts=expected_stmts)

    assert block.is_equal(expected_block)


def test2():

    node = qasm2.expr.ConstFloat(value=1.5707963267948966)
    node2 = qasm2.expr.ConstFloat(value=3.141592653589793)

    block = ir.Block(stmts=[node2, node])

    block.print()

    RydbergGateSetRewriteRule(qasm2.main)._rewrite_gate_stmts(...)

    block.print()
