import math

from bloqade import qasm2
from kirin.dialects import ilist


# > Simple linear depth impl of ghz state prep
# A simple GHZ state preparation circuit can be done with N CX gates and 1 H gate.
# This gives circuit execution depth of N+1.
def ghz_linear(n_qubits: int):

    @qasm2.main
    def ghz_linear_program():

        qreg = qasm2.qreg(n_qubits)
        qasm2.h(qreg[0])
        for i in range(1, n_qubits):
            qasm2.cx(qreg[i - 1], qreg[i])

    return ghz_linear_program


# > 1/2 linear depth by re-arranging
# Let's take a look how we can rewrite the circuit toward more QuEra's hardware friendly circuit.
# If we put the first haradamard gate on the middle qubit, and fan out the CX gate,
# we can effectively reduce the circuit depth to N/2 + 1. With each layer have two CX gates.


# Note it is important to separate the concept of circuit depth and circuit execution depth.
# For example, in the following implementation, the two CX gates instruction inside the for loop are executed in sequence.
# So even thought the circuit depth is N/2 + 1. The circuit execution depth is still N + 1.
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


# > 1/2 linear depth by re-arranging, and using parallelism
# Now lets see how we can ultilize QuEra's neutral atom's unique parallel feature to reduce the
# actual execution time of the circuit.
# It is important to know that on our digital quantum computer, by nature can execute native gate in parallel in an single instruction/ execution cycle.
# The concept is very similar to the SIMD (Single Instruction, Multiple Data) in classical computing.
# On our hardware, the native gate set is arbitrary (parallel) rotations and (parallel) CZ gates.
#
# We know that:
# 1. hadamard can be decomposed into a Ry(pi/2) rotation follow by a X gate.
# 2. the CX gate can be decomposed into CZ gate with two hadamards acting on the target qubit.
# 3. CZ gate commute with each other CZ gates.
# Base on the above observation, one can rewrite the circuit with only CZ gates in the middle,
# and some left-over parallel hadamard gates at the beginning and the end of the circuit


# We can now further rewrite the above circuit to using parallel gates.
# using parallel gates, we can further reduce the circuit execution depth to 6 (constant and does not scale with N)
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
        odd_qubits = ilist.Map(fn=get_qubit, collection=range(1, n_qubits, 2))

        # acting parallel H = XRy^{pi/2} on even qubits and middle qubit
        initial_targets = even_qubits + [qreg[s]]
        # Ry(pi/2)
        qasm2.parallel.u(qargs=initial_targets, theta=math.pi / 2, phi=0.0, lam=0.0)
        # X
        qasm2.parallel.u(qargs=initial_targets, theta=math.pi, phi=0.0, lam=math.pi)

        # two layer of parallel CZ gates
        qasm2.parallel.cz(ctrls=even_qubits, qargs=odd_qubits)
        qasm2.parallel.cz(ctrls=odd_qubits[:-1], qargs=even_qubits[1:])

        # acting parallel H = Ry^{-pi/2}X on even qubits only:
        # Ry(pi/2)
        qasm2.parallel.u(qargs=even_qubits, theta=math.pi / 2, phi=0.0, lam=0.0)
        # X
        qasm2.parallel.u(qargs=even_qubits, theta=math.pi, phi=0.0, lam=math.pi)

    return ghz_half_simd_program


# Note on using closure to capture global variable:
# Since qasm2 does not allow main program with arguments, so we need to put the program in a closure.
# our kirin compiler toolchain can capture the global variable inside the closure.
# In this case, the n_qubits will be captured upon calling the `ghz_half_simd(n_qubits)` python function,
# As a result, the return qasm2 program will not have any arguments.
