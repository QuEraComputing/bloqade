import math
from typing import TYPE_CHECKING

from kirin import interp
from bloqade.qasm2.dialects import uop
from bloqade.runtime.qrack.reg import SimQubitRef

if TYPE_CHECKING:
    from pyqrack import QrackSimulator


@uop.dialect.register(key="pyqrack")
class PyQrackMethods(interp.MethodTable):
    GATE_TO_METHOD = {
        "x": "x",
        "y": "y",
        "z": "z",
        "h": "h",
        "s": "s",
        "t": "t",
        "cx": "mcx",
        "CX": "mcx",
        "cz": "mcz",
        "cy": "mcy",
        "ch": "mch",
        "sdag": "adjs",
        "tdag": "adjt",
    }

    AXIS_MAP = {"rx": 1, "ry": 2, "rz": 3}

    @interp.impl(uop.Barrier)
    def barrier(
        self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.Barrier
    ):
        return ()

    @interp.impl(uop.X)
    @interp.impl(uop.Y)
    @interp.impl(uop.Z)
    @interp.impl(uop.H)
    @interp.impl(uop.S)
    @interp.impl(uop.Sdag)
    @interp.impl(uop.T)
    @interp.impl(uop.Tdag)
    def single_qubit_gate(
        self,
        interp: interp.Interpreter,
        frame: interp.Frame,
        stmt: uop.SingleQubitGate,
    ):
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active():
            getattr(qarg.sim_reg, self.GATE_TO_METHOD[stmt.name])(qarg.addr)
        return ()

    @interp.impl(uop.UGate)
    def ugate(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.UGate):
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active():
            qarg.sim_reg.u(
                qarg.addr,
                frame.get(stmt.theta),
                frame.get(stmt.phi),
                frame.get(stmt.lam),
            )
        return ()

    @interp.impl(uop.CX)
    @interp.impl(uop.CZ)
    @interp.impl(uop.CY)
    @interp.impl(uop.CH)
    def control_gate(
        self,
        interp: interp.Interpreter,
        frame: interp.Frame,
        stmt: uop.CX | uop.CZ | uop.CY,
    ):
        ctrl: SimQubitRef["QrackSimulator"] = frame.get(stmt.ctrl)
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if ctrl.is_active() and qarg.is_active():
            getattr(qarg.sim_reg, self.GATE_TO_METHOD[stmt.name])(
                [ctrl.addr], qarg.addr
            )
        return ()

    @interp.impl(uop.CCX)
    def ccx(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.CCX):
        ctrl1: SimQubitRef["QrackSimulator"] = frame.get(stmt.ctrl1)
        ctrl2: SimQubitRef["QrackSimulator"] = frame.get(stmt.ctrl2)
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if ctrl1.is_active() and ctrl2.is_active() and qarg.is_active():
            qarg.sim_reg.mcx([ctrl1.addr, ctrl2.addr], qarg.addr)
        return ()

    @interp.impl(uop.RX)
    @interp.impl(uop.RY)
    @interp.impl(uop.RZ)
    def rotation(
        self,
        interp: interp.Interpreter,
        frame: interp.Frame,
        stmt: uop.RX | uop.RY | uop.RZ,
    ):
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active():
            qarg.sim_reg.r(self.AXIS_MAP[stmt.name], frame.get(stmt.theta), qarg.addr)
        return ()

    @interp.impl(uop.U1)
    def u1(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.U1):
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active():
            qarg.sim_reg.u(qarg.addr, 0, 0, frame.get(stmt.lam))
        return ()

    @interp.impl(uop.U2)
    def u2(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.U2):
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active():
            qarg.sim_reg.u(
                qarg.addr, math.pi / 2, frame.get(stmt.phi), frame.get(stmt.lam)
            )
        return ()

    @interp.impl(uop.CRX)
    def crx(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.CRX):
        ctrl: SimQubitRef["QrackSimulator"] = frame.get(stmt.ctrl)
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active() and ctrl.is_active():
            qarg.sim_reg.mcr(1, frame.get(stmt.theta), [ctrl.addr], qarg.addr)
        return ()

    @interp.impl(uop.CU1)
    def cu1(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.CU1):
        ctrl: SimQubitRef["QrackSimulator"] = frame.get(stmt.ctrl)
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active() and ctrl.is_active():
            qarg.sim_reg.mcu([ctrl.addr], qarg.addr, 0, 0, frame.get(stmt.lam))
        return ()

    @interp.impl(uop.CU3)
    def cu3(self, interp: interp.Interpreter, frame: interp.Frame, stmt: uop.CU3):
        ctrl: SimQubitRef["QrackSimulator"] = frame.get(stmt.ctrl)
        qarg: SimQubitRef["QrackSimulator"] = frame.get(stmt.qarg)
        if qarg.is_active() and ctrl.is_active():
            qarg.sim_reg.mcu(
                [ctrl.addr],
                qarg.addr,
                frame.get(stmt.theta),
                frame.get(stmt.phi),
                frame.get(stmt.lam),
            )
        return ()
