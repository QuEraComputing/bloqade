import math

from bloqade import qasm2
from kirin.dialects import ilist


# simple linear depth impl of ghz state prep
def ghz_linear(n_qubits: int):

    @qasm2.main
    def ghz_linear_program():

        qreg = qasm2.qreg(n_qubits)
        qasm2.h(qreg[0])
        for i in range(1, n_qubits):
            qasm2.cx(qreg[i - 1], qreg[i])

    return ghz_linear_program


# 1/2 linear depth by re-arranging
def ghz_half(n_qubits: int):
    @qasm2.main
    def ghz_half_program():
        assert n_qubits % 2 == 0

        qreg = qasm2.qreg(n_qubits)

        # acting H on the middle qubit
        s = n_qubits // 2
        qasm2.h(qreg[s])

        # fan out the CX gate:
        qasm2.cx(qreg[s], qreg[s - 1])

        for i in range(s - 1, 0, -1):
            qasm2.cx(qreg[i], qreg[i - 1])
            qasm2.cx(qreg[n_qubits - i - 1], qreg[n_qubits - i])

    return ghz_half_program


# 1/2 linear depth by re-arranging, and using parallelism
def ghz_half_simd(n_qubits: int):
    @qasm2.main
    def ghz_half_simd_program():
        assert n_qubits % 2 == 0
        s = n_qubits // 2

        # create register
        qreg = qasm2.qreg(n_qubits)

        def get_qubit(i: int):
            return qreg[i]

        even_qubits = ilist.Map(fn=get_qubit, collection=range(0, n_qubits, 2))

        # acting parallel H = XRy^{pi/2} on even qubits and middle qubit
        initial_targets = even_qubits + [qreg[s]]
        # Ry(pi/2)
        qasm2.parallel.u(qargs=initial_targets, theta=math.pi / 2, phi=0.0, lam=0.0)
        # X
        qasm2.parallel.u(qargs=initial_targets, theta=math.pi, phi=0.0, lam=math.pi)

        # fan out the CZ gate:
        qasm2.cz(qreg[s], qreg[s - 1])

        for i in range(s - 1, 0, -1):
            qasm2.parallel.cz(
                ctrls=[qreg[i], qreg[n_qubits - i - 1]],
                qargs=[qreg[i - 1], qreg[n_qubits - i]],
            )

        # acting parallel H = Ry^{-pi/2}X on even qubits only:
        # Ry(pi/2)
        qasm2.parallel.u(qargs=even_qubits, theta=math.pi / 2, phi=0.0, lam=0.0)
        # X
        qasm2.parallel.u(qargs=even_qubits, theta=math.pi, phi=0.0, lam=math.pi)

    return ghz_half_simd_program


# Note:
# Since qasm2 does not allow main program with arguments, so we need to put the program in a closure.
# our kirin compiler toolchain can capture the global variable inside the closure.
# In this case, the n_qubits will be captured upon calling the `ghz_half_simd(n_qubits)` python function,
# As a result, the return qasm2 program will not have any arguments.
