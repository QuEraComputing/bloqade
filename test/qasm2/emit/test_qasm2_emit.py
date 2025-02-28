from bloqade import qasm2


def test_global_allow_global():

    @qasm2.extended
    def glob_u():
        qreg = qasm2.qreg(3)
        qreg1 = qasm2.qreg(3)
        qasm2.glob.u(theta=0.1, phi=0.2, lam=0.3, registers=[qreg, qreg1])

    target = qasm2.emit.QASM2(
        allow_global=True,
        allow_parallel=False,
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


def test_global_allow_global_allow_para():

    @qasm2.extended
    def glob_u():
        qreg = qasm2.qreg(3)
        qreg1 = qasm2.qreg(3)
        qasm2.glob.u(theta=0.1, phi=0.2, lam=0.3, registers=[qreg, qreg1])

    target = qasm2.emit.QASM2(
        allow_global=True,
        allow_parallel=True,
        custom_gate=True,
    )
    qasm2_str = target.emit_str(glob_u)
    print(qasm2_str)
    assert (
        qasm2_str
        == """KIRIN {func,lowering.call,lowering.func,py.ilist,qasm2.core,qasm2.expr,qasm2.glob,qasm2.indexing,qasm2.parallel,qasm2.uop,scf};
include "qelib1.inc";
qreg qreg[3];
qreg qreg1[3];
glob.U(0.1, 0.2, 0.3) {qreg, qreg1}
"""
    )


def test_global():

    @qasm2.extended
    def glob_u():
        qreg = qasm2.qreg(3)
        qreg1 = qasm2.qreg(3)
        qasm2.glob.u(theta=0.1, phi=0.2, lam=0.3, registers=[qreg, qreg1])

    target = qasm2.emit.QASM2(
        allow_global=False,
        allow_parallel=False,
        custom_gate=True,
    )
    qasm2_str = target.emit_str(glob_u)
    assert (
        qasm2_str
        == """OPENQASM 2.0;
include "qelib1.inc";
qreg qreg[3];
qreg qreg1[3];
U(0.1, 0.2, 0.3) qreg1[2];
U(0.1, 0.2, 0.3) qreg1[1];
U(0.1, 0.2, 0.3) qreg1[0];
U(0.1, 0.2, 0.3) qreg[2];
U(0.1, 0.2, 0.3) qreg[1];
U(0.1, 0.2, 0.3) qreg[0];
"""
    )


def test_global_allow_para():

    @qasm2.extended
    def glob_u():
        qreg = qasm2.qreg(3)
        qreg1 = qasm2.qreg(3)
        qasm2.glob.u(theta=0.1, phi=0.2, lam=0.3, registers=[qreg, qreg1])

    target = qasm2.emit.QASM2(
        allow_global=False,
        allow_parallel=True,
        custom_gate=True,
    )
    qasm2_str = target.emit_str(glob_u)

    assert (
        qasm2_str
        == """KIRIN {func,lowering.call,lowering.func,py.ilist,qasm2.core,qasm2.expr,qasm2.indexing,qasm2.parallel,qasm2.uop,scf};
include "qelib1.inc";
qreg qreg[3];
qreg qreg1[3];
parallel.U(0.1, 0.2, 0.3) {
  qreg[0];
  qreg[1];
  qreg[2];
  qreg1[0];
  qreg1[1];
  qreg1[2];
}
"""
    )


def test_para():

    @qasm2.extended
    def para_u():
        qreg = qasm2.qreg(3)
        qasm2.parallel.u(theta=0.1, phi=0.2, lam=0.3, qargs=[qreg[0], qreg[1]])

    para_u.print()

    target = qasm2.emit.QASM2(
        allow_parallel=False,
        allow_global=False,
        custom_gate=True,
    )
    qasm2_str = target.emit_str(para_u)
    assert (
        qasm2_str
        == """OPENQASM 2.0;
include "qelib1.inc";
qreg qreg[3];
U(0.1, 0.2, 0.3) qreg[1];
U(0.1, 0.2, 0.3) qreg[0];
"""
    )


def test_para_allow_para():

    @qasm2.extended
    def para_u():
        qreg = qasm2.qreg(3)
        qasm2.parallel.u(theta=0.1, phi=0.2, lam=0.3, qargs=[qreg[0], qreg[1]])

    para_u.print()

    target = qasm2.emit.QASM2(
        allow_parallel=True,
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


def test_para_allow_para_allow_global():

    @qasm2.extended
    def para_u():
        qreg = qasm2.qreg(3)
        qasm2.parallel.u(theta=0.1, phi=0.2, lam=0.3, qargs=[qreg[0], qreg[1]])

    para_u.print()

    target = qasm2.emit.QASM2(
        allow_parallel=True,
        allow_global=True,
        custom_gate=True,
    )
    qasm2_str = target.emit_str(para_u)
    assert (
        qasm2_str
        == """KIRIN {func,lowering.call,lowering.func,py.ilist,qasm2.core,qasm2.expr,qasm2.glob,qasm2.indexing,qasm2.parallel,qasm2.uop,scf};
include "qelib1.inc";
qreg qreg[3];
parallel.U(0.1, 0.2, 0.3) {
  qreg[0];
  qreg[1];
}
"""
    )


def test_para_allow_global():

    @qasm2.extended
    def para_u():
        qreg = qasm2.qreg(3)
        qasm2.parallel.u(theta=0.1, phi=0.2, lam=0.3, qargs=[qreg[0], qreg[1]])

    para_u.print()

    target = qasm2.emit.QASM2(
        allow_parallel=False,
        allow_global=True,
        custom_gate=True,
    )
    qasm2_str = target.emit_str(para_u)
    print(qasm2_str)
    assert (
        qasm2_str
        == """KIRIN {func,lowering.call,lowering.func,py.ilist,qasm2.core,qasm2.expr,qasm2.glob,qasm2.indexing,qasm2.uop,scf};
include "qelib1.inc";
qreg qreg[3];
U(0.1, 0.2, 0.3) qreg[1];
U(0.1, 0.2, 0.3) qreg[0];
"""
    )
