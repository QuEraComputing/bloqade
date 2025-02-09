import math
from unittest.mock import Mock, call

from kirin import ir
from bloqade import qasm2
from bloqade.runtime import qrack


def run_mock(size: int, program: ir.Method) -> Mock:
    memory = qrack.Memory(size, 0, Mock())
    interp = qrack.PyQrackInterpreter(qasm2.main, memory=memory)
    interp.run(program, ())

    return memory.sim_reg


def test_basic_gates():
    @qasm2.main
    def program():
        q = qasm2.qreg(3)

        qasm2.h(q[0])
        qasm2.x(q[1])
        qasm2.y(q[2])
        qasm2.z(q[0])
        qasm2.s(q[1])
        qasm2.sdg(q[2])
        qasm2.t(q[0])
        qasm2.tdg(q[1])

    sim_reg = run_mock(3, program)
    sim_reg.assert_has_calls(
        [
            call.h(0),
            call.x(1),
            call.y(2),
            call.z(0),
            call.s(1),
            call.adjs(2),
            call.t(0),
            call.adjt(1),
        ]
    )


def test_rotation_gates():
    @qasm2.main
    def program():
        q = qasm2.qreg(3)

        qasm2.rx(q[0], 0.5)
        qasm2.ry(q[1], 0.5)
        qasm2.rz(q[2], 0.5)

    sim_reg = run_mock(3, program)

    sim_reg.assert_has_calls(
        [
            call.r(1, 0.5, 0),
            call.r(2, 0.5, 1),
            call.r(3, 0.5, 2),
        ]
    )


def test_u_gates():
    @qasm2.main
    def program():
        q = qasm2.qreg(3)

        qasm2.u(q[0], 0.5, 0.2, 0.1)
        qasm2.u2(q[1], 0.2, 0.1)
        qasm2.u1(q[2], 0.2)

    sim_reg = run_mock(3, program)
    sim_reg.assert_has_calls(
        [
            call.u(0, 0.5, 0.2, 0.1),
            call.u(1, math.pi / 2, 0.2, 0.1),
            call.u(2, 0, 0, 0.2),
        ]
    )


def test_basic_control_gates():
    @qasm2.main
    def program():
        q = qasm2.qreg(3)

        qasm2.cx(q[0], q[1])
        qasm2.cy(q[1], q[2])
        qasm2.cz(q[2], q[0])
        qasm2.ch(q[0], q[1])

    sim_reg = run_mock(3, program)
    sim_reg.assert_has_calls(
        [
            call.mcx([0], 1),
            call.mcy([1], 2),
            call.mcz([2], 0),
            call.mch([0], 1),
        ]
    )


def test_special_control():
    @qasm2.main
    def program():
        q = qasm2.qreg(3)

        qasm2.crx(q[0], q[1], 0.5)
        qasm2.cu1(q[1], q[2], 0.5)
        qasm2.cu3(q[2], q[0], 0.5, 0.2, 0.1)
        qasm2.ccx(q[0], q[1], q[2])

    sim_reg = run_mock(3, program)
    sim_reg.assert_has_calls(
        [
            call.mcr(1, 0.5, [0], 1),
            call.mcu([1], 2, 0, 0, 0.5),
            call.mcu([2], 0, 0.5, 0.2, 0.1),
            call.mcx([0, 1], 2),
        ]
    )
