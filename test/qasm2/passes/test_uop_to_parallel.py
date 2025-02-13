from bloqade import qasm2
from bloqade.qasm2.passes import parallel


@qasm2.gate
def gate(q1: qasm2.Qubit, q2: qasm2.Qubit):
    qasm2.cx(q1, q2)


@qasm2.main
def test():
    q = qasm2.qreg(4)

    theta = 0.1
    phi = 0.2
    lam = 0.3

    qasm2.u(q[1], theta, phi, lam)
    qasm2.u(q[3], theta, phi, lam)
    qasm2.u(q[2], theta, phi, lam)


# test.print()
parallel.UOpToParallel(test.dialects)(test)
test.print()
