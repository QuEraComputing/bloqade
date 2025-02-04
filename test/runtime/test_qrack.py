from unittest.mock import Mock, call

from bloqade import qasm2
from bloqade.runtime import qrack


@qasm2.main
def all_gates():
    q = qasm2.qreg(3)

    qasm2.h(q[0])
    qasm2.x(q[1])
    qasm2.y(q[2])
    qasm2.z(q[0])
    qasm2.s(q[1])
    qasm2.sdg(q[2])
    qasm2.t(q[0])
    qasm2.tdg(q[1])
    qasm2.rx(q[0], 0.5)
    qasm2.ry(q[1], 0.5)
    qasm2.rz(q[2], 0.5)
    qasm2.u1(q[0], 0.5)
    qasm2.u2(q[1], 0.5, 0.5)
    qasm2.u(q[2], 0.5, 0.5, 0.5)
    qasm2.barrier((q[0], q[1], q[2]))
    qasm2.ccx(q[0], q[1], q[2])
    qasm2.cz(q[0], q[1])
    qasm2.cy(q[1], q[2])
    qasm2.cx(q[0], q[2])
    qasm2.crx(q[1], q[2], 0.5)
    qasm2.cu1(q[2], q[0], 0.5)
    qasm2.cu3(q[0], q[1], 0.5, 0.5, 0.5)
    qasm2.parallel.cz((q[0],), (q[1],))
    qasm2.parallel.u((q[0], q[1], q[2]), 0.5, 0.5, 0.5)
    qasm2.parallel.rz((q[0], q[1], q[2]), 0.5)


def test_all_gates():

    memory = qrack.Memory(3, 0, Mock())
    interp = qrack.PyQrackInterpreter(qasm2.main, memory=memory)
    interp.run(all_gates, ())

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
