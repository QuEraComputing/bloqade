from bloqade import qasm2
from bloqade.qasm2 import glob
from bloqade.analysis import address
from bloqade.qasm2.passes import parallel
from bloqade.qasm2.rewrite import SimpleOptimalMergePolicy


def test_one():

    @qasm2.gate
    def gate(q1: qasm2.Qubit, q2: qasm2.Qubit):
        qasm2.cx(q1, q2)

    @qasm2.extended
    def test():
        q = qasm2.qreg(4)

        theta = 0.1
        phi = 0.2
        lam = 0.3

        qasm2.u(q[1], theta, phi, lam)
        qasm2.u(q[0], 0.4, phi, lam)
        qasm2.u(q[3], theta, phi, lam)
        qasm2.u(q[2], 0.4, phi, lam)

        gate(q[1], q[3])
        qasm2.barrier((q[1], q[2]))
        qasm2.u(q[2], theta, phi, lam)
        glob.u(theta=theta, phi=phi, lam=lam, registers=[q])
        qasm2.u(q[0], theta, phi, lam)

        gate(q[0], q[2])

    parallel.UOpToParallel(test.dialects)(test)
    test.print()

    # add this to raise error if there are broken ssa references
    _, _ = address.AddressAnalysis(test.dialects).run_analysis(test, no_raise=False)


def test_two():

    @qasm2.extended
    def test():
        q = qasm2.qreg(8)
        theta = 0.1
        theta2 = 0.4
        phi = 0.2
        lam = 0.3

        qasm2.rz(q[0], 0.8)
        qasm2.rz(q[1], 0.7)

        qasm2.u(q[1], theta, phi, lam)
        qasm2.u(q[0], theta2, phi, lam)

        qasm2.rz(q[2], 0.6)
        qasm2.rz(q[3], 0.5)

        qasm2.u(q[3], theta, phi, lam)
        qasm2.u(q[2], theta2, phi, lam)

        qasm2.rz(q[4], 0.5)
        qasm2.rz(q[5], 0.6)

        qasm2.u(q[4], theta, phi, lam)
        qasm2.u(q[5], 0.4, phi, lam)

        qasm2.rz(q[6], 0.7)
        qasm2.rz(q[7], 0.8)

    parallel.UOpToParallel(test.dialects, SimpleOptimalMergePolicy)(test)
    test.print()

    # add this to raise error if there are broken ssa references
    _, _ = address.AddressAnalysis(test.dialects).run_analysis(test, no_raise=False)


def test_three():

    @qasm2.extended
    def test():
        q1 = qasm2.qreg(1)
        qasm2.u(q1[0], 0.1, 0.2, 0.3)

        q2 = qasm2.qreg(1)
        qasm2.u(q2[0], 0.1, 0.2, 0.3)
        qasm2.cz(q2[0], q1[0])

        q3 = qasm2.qreg(1)
        qasm2.u(q3[0], 0.1, 0.2, 0.3)
        qasm2.cz(q3[0], q2[0])

    parallel.UOpToParallel(test.dialects)(test)
    test.print()

    # add this to raise error if there are broken ssa references
    _, _ = address.AddressAnalysis(test.dialects).run_analysis(test, no_raise=False)
