from bloqade import qasm2
from kirin.dialects import ilist


def test_global():

    @qasm2.extended
    def glob_u():
        qreg = qasm2.qreg(3)
        qreg1 = qasm2.qreg(3)
        qasm2.glob.u(theta=0.1, phi=0.2, lam=0.3, registers=[qreg, qreg1])

    glob_u.print()

    target = qasm2.emit.QASM2(
        main_target=qasm2.main.add(qasm2.dialects.glob).add(ilist),
        gate_target=qasm2.gate.add(qasm2.dialects.glob).add(ilist),
        custom_gate=True,
    )
    ast = target.emit(glob_u)
    qasm2.parse.pprint(ast)


test_global()


def test_para():

    @qasm2.extended
    def para_u():
        qreg = qasm2.qreg(3)
        qasm2.parallel.u(theta=0.1, phi=0.2, lam=0.3, qargs=[qreg[0], qreg[1]])

    para_u.print()

    target = qasm2.emit.QASM2(
        main_target=qasm2.main.add(qasm2.dialects.parallel).add(ilist),
        gate_target=qasm2.gate.add(qasm2.dialects.parallel).add(ilist),
        custom_gate=True,
    )
    ast = target.emit(para_u)
    qasm2.parse.pprint(ast)


test_para()
