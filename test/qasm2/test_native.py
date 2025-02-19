from bloqade import qasm2
from kirin.rewrite import Walk
from bloqade.qasm2.rewrite.native_gates import RydbergGateSetRewriteRule


def test_native():

    @qasm2.main
    def qasm2_cx():
        reg = qasm2.qreg(3)
        qasm2.cx(reg[0], reg[1])

    qasm2_cx.print()

    Walk(RydbergGateSetRewriteRule()).rewrite(qasm2_cx.code)

    qasm2_cx.print()


test_native()
