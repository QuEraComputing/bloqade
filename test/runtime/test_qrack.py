from unittest.mock import Mock, call

from bloqade import qasm2
from bloqade.runtime import qrack


@qasm2.main
def all_gates():
    q = qasm2.qreg(3)

    qasm2.H(q[0])
    qasm2.X(q[1])
    qasm2.Y(q[2])
    qasm2.Z(q[0])
    qasm2.S(q[1])
    qasm2.Sdag(q[2])
    qasm2.T(q[0])
    qasm2.Tdag(q[1])
    qasm2.RX(q[0], 0.5)
    qasm2.RY(q[1], 0.5)
    qasm2.RZ(q[2], 0.5)
    qasm2.U1(q[0], 0.5)
    qasm2.U2(q[1], 0.5, 0.5)
    qasm2.UGate(q[2], 0.5, 0.5, 0.5)
    qasm2.barrier((q[0], q[1], q[2]))
    qasm2.CCX(q[0], q[1], q[2])
    qasm2.CZ(q[0], q[1])
    qasm2.CY(q[1], q[2])
    qasm2.CX(q[0], q[2])
    qasm2.CRX(q[1], q[2], 0.5)
    qasm2.CU1(q[2], q[0], 0.5)
    qasm2.CU3(q[0], q[1], 0.5, 0.5, 0.5)
    qasm2.parallel.CZ((q[0],), (q[1],))
    qasm2.parallel.UGate((q[0], q[1], q[2]), 0.5, 0.5, 0.5)
    qasm2.parallel.RZ((q[0], q[1], q[2]), 0.5)


def test_all_gates():

    memory = qrack.Memory(3, 0, Mock())
    interp = qrack.PyQrackInterpreter(qasm2.main, memory=memory)
    interp.eval(all_gates)

    memory.sim_reg.calls = [
        call.h(0),
        call.x(1),
        call.y(2),
        call.z(0),
        call.s(1),
        call.sdag(2),
        call.t(0),
        call.tdag(1),
        call.rx(0, 0.5),
        call.ry(1, 0.5),
        call.rz(2, 0.5),
        call.u1(0, 0.5),
        call.u2(1, 0.5, 0.5),
        call.ugate(2, 0.5, 0.5, 0.5),
        call.mcx(0, 1, 2),
        call.mcz(0, 1),
        call.mcy(1, 2),
        call.mcx(0, 2),
        call.mrx(1, 2, 0.5),
        call.mcu(2, 0, 0.5, 0.0, 0.0),
        call.mcu(0, 1, 0.5, 0.5, 0.5),
        call.mcz(0, 1),
        call.u(0, 0.5, 0.5, 0.5),
        call.u(1, 0.5, 0.5, 0.5),
        call.u(2, 0.5, 0.5, 0.5),
        call.rz(0, 0.5),
        call.rz(1, 0.5),
        call.rz(2, 0.5),
    ]
