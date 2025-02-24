from bloqade import qasm2
from kirin.dialects import ilist


def test_global():

    @qasm2.extended
    def glob_u():
        qreg = qasm2.qreg(3)
        qreg1 = qasm2.qreg(3)
        qasm2.glob.u(theta=0.1, phi=0.2, lam=0.3, registers=[qreg, qreg1])

    target = qasm2.emit.QASM2(
        main_target=qasm2.main.add(qasm2.dialects.glob).add(ilist),
        gate_target=qasm2.gate.add(qasm2.dialects.glob).add(ilist),
        custom_gate=True,
    )
    qasm2_str = target.emit_str(glob_u)
    assert (
        qasm2_str
        == """KIRIN {func,lowering.call,lowering.func,py.ilist,qasm2.core,qasm2.expr,qasm2.glob,qasm2.indexing,qasm2.uop,scf};
include "qelib1.inc";
qreg qreg[3];
qreg qreg1[3];
glob.U(0.1, 0.2, 0.3) {qreg, qreg1}
"""
    )


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
    qasm2_str = target.emit_str(para_u)
    assert (
        qasm2_str
        == """KIRIN {func,lowering.call,lowering.func,py.ilist,qasm2.core,qasm2.expr,qasm2.indexing,qasm2.parallel,qasm2.uop,scf};
include "qelib1.inc";
qreg qreg[3];
parallel.U(0.1, 0.2, 0.3) {
  qreg[0];
  qreg[1];
}
"""
    )
