from typing import TYPE_CHECKING

from kirin import interp
from bloqade.qasm2.dialects import parallel
from bloqade.qasm2.runtime.qrack.reg import SimQubitRef
from bloqade.qasm2.runtime.qrack.base import PyQrackInterpreter

if TYPE_CHECKING:
    from pyqrack import QrackSimulator


@parallel.dialect.register(key="pyqrack")
class PyQrackMethods(interp.MethodTable):

    @interp.impl(parallel.CZ)
    def cz(self, interp: PyQrackInterpreter, frame: interp.Frame, stmt: parallel.CZ):
        qargs: tuple[SimQubitRef["QrackSimulator"], ...] = frame.get_values(stmt.qargs)
        ctrls: tuple[SimQubitRef["QrackSimulator"], ...] = frame.get_values(stmt.ctrls)
        for qarg, ctrl in zip(qargs, ctrls):
            interp.memory.sim_reg.mcz(qarg, ctrl)
        return ()

    @interp.impl(parallel.UGate)
    def ugate(
        self, interp: PyQrackInterpreter, frame: interp.Frame, stmt: parallel.UGate
    ):
        qargs: tuple[SimQubitRef["QrackSimulator"], ...] = frame.get_values(stmt.qargs)
        theta, phi, lam = (
            frame.get(stmt.theta),
            frame.get(stmt.phi),
            frame.get(stmt.lam),
        )
        for qarg in qargs:
            interp.memory.sim_reg.u(qarg, theta, phi, lam)
        return ()

    @interp.impl(parallel.RZ)
    def rz(self, interp: PyQrackInterpreter, frame: interp.Frame, stmt: parallel.RZ):
        qargs: tuple[SimQubitRef["QrackSimulator"], ...] = frame.get_values(stmt.qargs)
        phi = frame.get(stmt.theta)
        for qarg in qargs:
            interp.memory.sim_reg.r(3, phi, qarg)
        return ()
