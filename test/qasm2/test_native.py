import math

import cirq
import cirq.contrib
import cirq.testing
import cirq.contrib.qasm_import
import cirq.circuits.qasm_output
from kirin import ir
from bloqade import qasm2
from kirin.rewrite import Walk, cse, walk, fixpoint
from bloqade.test_utils import assert_nodes
from bloqade.qasm2.rewrite.native_gates import (
    RydbergGateSetRewriteRule,
    one_qubit_gate_to_u3_angles,
)


def test_one_qubit_gate_to_u3_angles():
    theta = 1.1 * math.pi
    phi = 0.2 * math.pi
    lam = 1.6 * math.pi

    op = cirq.circuits.qasm_output.QasmUGate(
        theta / math.pi, phi / math.pi, lam / math.pi
    )(cirq.LineQubit(0))

    theta, phi, lam = one_qubit_gate_to_u3_angles(op)
    op1 = cirq.circuits.qasm_output.QasmUGate(
        theta / math.pi, phi / math.pi, lam / math.pi
    )(cirq.LineQubit(0))

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(op), cirq.unitary(op1), atol=1e-8
    )
