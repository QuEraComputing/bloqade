import math
import textwrap

import cirq
import cirq.contrib
import cirq.testing
import cirq.contrib.qasm_import as qasm_import
import cirq.circuits.qasm_output as qasm_output
from pytest import mark
from bloqade import qasm2
from kirin.rewrite import walk
from bloqade.qasm2.rewrite.native_gates import (
    RydbergGateSetRewriteRule,
    one_qubit_gate_to_u3_angles,
)


def test_one_qubit_gate_to_u3_angles():
    theta = 1.1 * math.pi
    phi = 0.2 * math.pi
    lam = 1.6 * math.pi

    op = qasm_output.QasmUGate(theta / math.pi, phi / math.pi, lam / math.pi)(
        cirq.LineQubit(0)
    )

    theta, phi, lam = one_qubit_gate_to_u3_angles(op)
    op1 = qasm_output.QasmUGate(theta / math.pi, phi / math.pi, lam / math.pi)(
        cirq.LineQubit(0)
    )

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(op), cirq.unitary(op1), atol=1e-8
    )


def generator(n_tests: int):
    import numpy as np

    yield textwrap.dedent(
        """
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[1];

    tdg q[0];

    """
    )

    yield textwrap.dedent(
        """
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[1];

    sdg q[0];

    """
    )

    yield textwrap.dedent(
        """
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[1];

    sx q[0];

    """
    )

    yield textwrap.dedent(
        """
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[1];

    sxdg q[0];

    """
    )

    rgen = np.random.RandomState(128)
    for num in range(n_tests):
        # Generate a new instance:
        N = int(rgen.choice([4, 5, 6, 7, 8]))
        D = int(rgen.choice([1, 3, 5, 10, 15]))
        rho = float(rgen.random())
        circ: cirq.Circuit = cirq.testing.random_circuit(N, D, rho, random_state=rgen)
        while len(circ.all_qubits()) == 0:
            circ = cirq.testing.random_circuit(N, D, rho, random_state=rgen)
        yield circ.to_qasm()


@mark.parametrize(
    "qasm2_prog",
    generator(20),
)
def test_rewrite(qasm2_prog: str):
    @qasm2.main.add(qasm2.dialects.inline)
    def kernel():
        qasm2.inline(qasm2_prog)

    walk.Walk(RydbergGateSetRewriteRule(kernel.dialects)).rewrite(kernel.code)

    new_qasm2 = qasm2.emit.QASM2().emit_str(kernel)

    cirq_circuit = qasm_import.circuit_from_qasm(new_qasm2)
    old_circuit = qasm_import.circuit_from_qasm(qasm2_prog)

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(old_circuit), cirq.unitary(cirq_circuit), atol=1e-8
    )
