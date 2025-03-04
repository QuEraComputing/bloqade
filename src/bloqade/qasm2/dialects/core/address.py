from kirin import interp
from bloqade.analysis.address import (
    Address,
    NotQubit,
    AddressReg,
    AddressQubit,
    AddressAnalysis,
)

from .stmts import QRegGet, QRegNew
from ._dialect import dialect


@dialect.register(key="qubit.address")
class AddressMethodTable(interp.MethodTable):

    @interp.impl(QRegNew)
    def new(
        self,
        interp: AddressAnalysis,
        frame: interp.Frame[Address],
        stmt: QRegNew,
    ):
        n_qubits = interp.get_const_value(int, stmt.n_qubits)
        interp.next_address += n_qubits
        return (AddressReg(range(interp.next_address, interp.next_address + n_qubits)),)

    @interp.impl(QRegGet)
    def get(self, interp: AddressAnalysis, frame: interp.Frame[Address], stmt: QRegGet):
        addr = frame.get(stmt.reg)
        pos = interp.get_const_value(int, stmt.idx)
        if isinstance(addr, AddressReg):
            global_idx = addr.data[pos]
            return (AddressQubit(global_idx),)
        else:  # this is not reachable
            return (NotQubit(),)
