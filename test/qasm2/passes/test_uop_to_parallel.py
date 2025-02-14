from bloqade import qasm2
from bloqade.qasm2 import glob
from bloqade.qasm2.passes import parallel


def test():

    @qasm2.gate
    def gate(q1: qasm2.Qubit, q2: qasm2.Qubit):
        qasm2.cx(q1, q2)

    @qasm2.extended
    def test():
        q = qasm2.qreg(4)

        theta = 0.1
        phi = 0.2
        lam = 0.3

        qasm2.u(q[1], 0.1, 0.2, 0.3)
        qasm2.u(q[3], theta, phi, lam)
        gate(q[1], q[3])
        qasm2.u(q[2], theta, phi, lam)
        glob.u(theta=theta, phi=phi, lam=lam, registers=[q])
        qasm2.u(q[0], theta, phi, lam)

        gate(q[0], q[2])

    # test.print()
    parallel.UOpToParallel(test.dialects)(test)
    test.print()
