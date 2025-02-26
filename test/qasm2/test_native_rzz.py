import math

import numpy as np
from bloqade import qasm2
from cirq.testing import assert_allclose_up_to_global_phase
from kirin.rewrite import Walk
from cirq.contrib.qasm_import import circuit_from_qasm
from bloqade.qasm2.rewrite.native_gates import RydbergGateSetRewriteRule


def test():

    theta = math.pi / 2

    @qasm2.main
    def main():
        reg = qasm2.qreg(2)
        # qasm2.cu(reg[0], reg[1], theta = 0.1, phi = 0.2, lam = 0.3, gamma = 0.4)
        qasm2.rzz(reg[0], reg[1], theta=theta)

    # Convert to U + CZ
    Walk(RydbergGateSetRewriteRule(main.dialects)).rewrite(main.code)

    # Generate QASM
    target = qasm2.emit.QASM2(custom_gate=True)
    bloqade_qasm2_str = target.emit_str(main)
    print(bloqade_qasm2_str)

    # Load the QASM into Cirq
    bloqade_circuit = circuit_from_qasm(bloqade_qasm2_str)
    print(bloqade_circuit)

    # Build the circuit directly in Cirq
    ## for rzz (and probably rxx as well) there doesn't seem to exist a native QASM type \
    ## but I can try and use the raw definition
    cirq_circuit_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];

    cx q[0], q[1];
    u1(pi/3) q[1];
    cx q[0], q[1];
    """
    print(cirq_circuit_str)
    cirq_circuit = circuit_from_qasm(cirq_circuit_str)
    print(cirq_circuit)

    # Check equivalence of circuits

    np.set_printoptions(precision=5, suppress=True)
    print("Bloqade Unitary")
    print(bloqade_circuit.unitary())
    print("Cirq Unitary")
    print(cirq_circuit.unitary())
    assert_allclose_up_to_global_phase(
        actual=bloqade_circuit.unitary(), desired=cirq_circuit.unitary(), atol=1e-5
    )


test()
