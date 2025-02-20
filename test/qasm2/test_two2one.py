from bloqade import qasm2


def test_221():

    @qasm2.main
    def testcirc():
        q = qasm2.qreg(3)
        qasm2.u(qarg=q[0], theta=0.1, phi=0.1, lam=0.1)

    @qasm2.main
    def testcirc2():
        q = qasm2.qreg(3)
        qasm2.u3(qarg=q[0], theta=0.1, phi=0.1, lam=0.1)

    assert testcirc.callable_region.is_structurally_equal(testcirc2.callable_region)
