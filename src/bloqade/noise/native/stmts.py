from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from bloqade.qasm2.types import QubitType

from ._dialect import dialect


@statement(dialect=dialect)
class PauliChannel(ir.Statement):
    name = "native.pauli_channel"

    traits = frozenset({ir.FromPythonCall()})

    px: float = info.attribute(types.Float)
    py: float = info.attribute(types.Float)
    pz: float = info.attribute(types.Float)
    qargs: ir.SSAValue = info.argument(ilist.IListType[QubitType])


NumQubits = types.TypeVar("NumQubits")


@statement(dialect=dialect)
class CZPauliChannel(ir.Statement):
    name = "native.pauli_channel.cz_pauli_channel"

    traits = frozenset({ir.FromPythonCall()})

    paired: bool = info.attribute(types.Bool)
    px_ctrl: float = info.attribute(types.Float)
    py_ctrl: float = info.attribute(types.Float)
    pz_ctrl: float = info.attribute(types.Float)
    px_qarg: float = info.attribute(types.Float)
    py_qarg: float = info.attribute(types.Float)
    pz_qarg: float = info.attribute(types.Float)
    ctrls: ir.SSAValue = info.argument(ilist.IListType[QubitType, NumQubits])
    qargs: ir.SSAValue = info.argument(ilist.IListType[QubitType, NumQubits])


@statement(dialect=dialect)
class AtomLossChannel(ir.Statement):
    name = "native.atom_loss_channel"

    traits = frozenset({ir.FromPythonCall()})

    prob: float = info.attribute(types.Float)
    qargs: ir.SSAValue = info.argument(ilist.IListType[QubitType])
