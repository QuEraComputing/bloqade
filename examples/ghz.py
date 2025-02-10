import math

from bloqade import qasm2
from kirin.dialects import ilist

# In this example, we will consider the GHZ state preparation circuit with 2^n qubits


# > Simple linear depth impl of ghz state prep
# A simple GHZ state preparation circuit can be done with N CX gates and 1 H gate.
# This gives circuit execution depth of N+1.
def ghz_linear(n: int):
    n_qubits = int(2**n)

    @qasm2.main
    def ghz_linear_program():

        qreg = qasm2.qreg(n_qubits)
        qasm2.h(qreg[0])
        for i in range(1, n_qubits):
            qasm2.cx(qreg[i - 1], qreg[i])

    return ghz_linear_program


# > log depth impl of ghz state prep
# Let's take a look how we can rewrite the circuit toward more QuEra's hardware friendly circuit.
# We can rewrite the GHZ state preparation circuit with log(N) depth by rearranging the CX gates
# [citation](https://arxiv.org/abs/2101.08946 – Mooney, White, Hill, Hollenberg)


# Note it is important to separate the concept of circuit depth and circuit execution depth.
# For example, in the following implementation, each CX gate instruction inside the for loop are executed in sequence.
# So even thought the circuit depth is N/2 + 1. The circuit execution depth is still N + 1.
def ghz_log_depth(n: int):
    n_qubits = int(2**n)

    @qasm2.main
    def layer(i_layer: int, qreg: qasm2.QReg):
        step = n_qubits // (2**i_layer)
        for j in range(0, n_qubits, step):
            qasm2.cx(ctrl=qreg[j], qarg=qreg[j + step // 2])

    @qasm2.main
    def ghz_log_depth_program():

        qreg = qasm2.qreg(n_qubits)

        qasm2.h(qreg[0])
        for i in range(n):
            layer(i_layer=i, qreg=qreg)

    return ghz_log_depth_program


# > native gate set and parallelism
# On our digital quantum computer, by nature can execute native gate in parallel in an single instruction/ execution cycle.
# The concept is very similar to the SIMD (Single Instruction, Multiple Data) in classical computing.
# On our hardware, there are two important factor to be consider:
# 1. the native gate set is arbitrary (parallel) rotations and (parallel) CZ gates.
# 2. Our atom shuttling architecture allows arbitrary qubit connectivity. This means that our parallel instruction is not limited to certain hardware connectivity (for example nearest neighbor connectivity).
#
# Let's try to rewrite the `layer` subroutinme.
# We know that the CX gate can be decomposed into CZ gate with two single qubit gates Ry(-pi/2) and Ry(pi/2) acting on the target qubits.
# After such decomposition, we can now using our parallel gate instructions `parallel.u` and `parallel.cz`.
# With the following modification, we can further reduce the circuit execution depth to n (log of total qubit number N)
def ghz_log_simd(n: int):
    n_qubits = int(2**n)

    @qasm2.main
    def layer(i_layer: int, qreg: qasm2.QReg):
        step = n_qubits // (2**i_layer)

        def get_qubit(x: int):
            return qreg[x]

        ctrl_qubits = ilist.Map(fn=get_qubit, collection=range(0, n_qubits, step))
        targ_qubits = ilist.Map(
            fn=get_qubit, collection=range(step // 2, n_qubits, step)
        )

        # Ry(-pi/2)
        qasm2.parallel.u(qargs=targ_qubits, theta=-math.pi / 2, phi=0.0, lam=0.0)

        # CZ gates
        qasm2.parallel.cz(ctrls=ctrl_qubits, qargs=targ_qubits)

        # Ry(pi/2)
        qasm2.parallel.u(qargs=targ_qubits, theta=math.pi / 2, phi=0.0, lam=0.0)

    @qasm2.main
    def ghz_log_depth_program():

        qreg = qasm2.qreg(n_qubits)

        qasm2.h(qreg[0])
        for i in range(n):
            layer(i_layer=i, qreg=qreg)

    return ghz_log_depth_program


# Note on using closure to capture global variable:
# Since qasm2 does not allow main program with arguments, so we need to put the program in a closure.
# our kirin compiler toolchain can capture the global variable inside the closure.
# In this case, the n_qubits will be captured upon calling the `ghz_half_simd(n_qubits)` python function,
# As a result, the return qasm2 program will not have any arguments.
