from bloqade import qasm2
from bloqade.test_utils import assert_methods
from bloqade.qasm2.passes import Py2QASM, QASM2Py


def test_round_trip():
    @qasm2.main
    def test():
        q = qasm2.qreg(2)
        c = qasm2.creg(1)
        theta = qasm2.sin(1.0) / qasm2.sqrt(1.0)
        phi = qasm2.cos(1.0) * qasm2.exp(1.0)
        lam = -qasm2.tan(1.0) + qasm2.exp(1.0) - 2.0**4.0
        if c == c:
            qasm2.u(q[0], theta, phi, lam)

    test_original = test.similar()

    QASM2Py(test.dialects)(test)

    test = test.similar(qasm2.extended)

    Py2QASM(test.dialects)(test)

    test = test.similar(qasm2.main)
    assert_methods(test, test_original)
