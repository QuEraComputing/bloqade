from kirin import interp
from bloqade.qasm2.dialects import core
from bloqade.qasm2.runtime.qrack.reg import SimQubitRef, SimQRegister
from bloqade.qasm2.runtime.qrack.base import PyQrackInterpreter


@core.dialect.register(key="pyqrack")
class PyQrackMethods(interp.MethodTable):

    @interp.impl(core.QRegNew)
    def qreg_new(
        self, interp: PyQrackInterpreter, frame: interp.Frame, stmt: core.QRegNew
    ):
        n_qubits: int = frame.get(stmt.n_qubits)
        curr_allocated = interp.memory.allocated
        interp.memory.allocated += n_qubits

        if interp.memory.allocated > interp.memory.total:
            raise ValueError("qubit allocation exceeds memory")

        return (
            SimQRegister(
                size=n_qubits,
                sim_reg=interp.memory.sim_reg,
                addrs=tuple(range(curr_allocated, curr_allocated + n_qubits)),
            ),
        )

    @interp.impl(core.QRegGet)
    def qreg_get(
        self, interp: PyQrackInterpreter, frame: interp.Frame, stmt: core.QRegGet
    ):
        return (SimQubitRef(ref=frame.get(stmt.reg), pos=frame.get(stmt.idx)),)
