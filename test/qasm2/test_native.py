import cirq
import cirq.circuits
from kirin import ir
from bloqade import qasm2
from kirin.rewrite import Walk, cse, walk, fixpoint
from bloqade.qasm2.rewrite.native_gates import RydbergGateSetRewriteRule


def test_native():

    @qasm2.main
    def qasm2_cx():
        reg = qasm2.qreg(3)
        qasm2.cx(reg[0], reg[1])

    qasm2_cx.print()

    Walk(RydbergGateSetRewriteRule(qasm2_cx.dialects)).rewrite(qasm2_cx.code)

    qasm2_cx.print()


def test_rewrite_gate_stmts():
    rule = RydbergGateSetRewriteRule(qasm2.main)

    block = ir.Block(
        [
            stmt1 := qasm2.expr.ConstFloat(value=0.2),
            stmt2 := qasm2.expr.ConstFloat(value=0.0),
            stmt3 := qasm2.expr.ConstFloat(value=6.283185307179586),
        ]
    )

    stmts = [
        in1 := qasm2.expr.ConstInt(value=5),
        in2 := qasm2.expr.ConstInt(value=6),
        in3 := qasm2.expr.ConstPI(),
    ]

    rule._rewrite_gate_stmts(stmts, stmt2)

    expected_block = ir.Block(
        [
            stmt1.from_stmt(stmt1),
            in1.from_stmt(in1),
            in2.from_stmt(in2),
            in3.from_stmt(in3),
            stmt3.from_stmt(stmt3),
        ]
    )

    assert block.is_equal(expected_block)


def test_generate_1q_gate_stmts():

    q = cirq.LineQubit.range(1)

    qubit_ssa = ir.TestValue(type=qasm2.QubitType)

    stmts = RydbergGateSetRewriteRule(qasm2.main)._generate_1q_gate_stmts(
        cirq.YPowGate(exponent=0.2)(q[0]), qubit_ssa
    )

    block = ir.Block(stmts=stmts)

    expected_stmts = [
        (s0 := qasm2.expr.ConstFloat(value=0.6283185307179588)),
        (s1 := qasm2.expr.ConstFloat(value=6.283185307179586)),
        (s2 := qasm2.expr.ConstFloat(value=0.0)),
        (qasm2.uop.UGate(qubit_ssa, s0.result, s1.result, s2.result)),
    ]

    expected_block = ir.Block(stmts=expected_stmts)

    assert block.is_equal(expected_block)


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
