from kirin import ir, types
from kirin.decl import info, statement
from bloqade.qasm2.types import QubitType

from ._dialect import dialect


@statement(dialect=dialect)
class PauliChannel(ir.Statement):
    name = "native.pauli_channel"

    traits = frozenset({ir.FromPythonCall()})

    px: ir.SSAValue = info.argument(type=types.Float)
    py: ir.SSAValue = info.argument(type=types.Float)
    pz: ir.SSAValue = info.argument(type=types.Float)
    qarg: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class CZPauliChannel(ir.Statement):
    name = "native.pauli_channel.cz_pauli_channel"

    traits = frozenset({ir.FromPythonCall()})

    paired: bool = info.attribute(property=True)
    px_1: ir.SSAValue = info.argument(type=types.Float)
    py_1: ir.SSAValue = info.argument(type=types.Float)
    pz_1: ir.SSAValue = info.argument(type=types.Float)
    px_2: ir.SSAValue = info.argument(type=types.Float)
    py_2: ir.SSAValue = info.argument(type=types.Float)
    pz_2: ir.SSAValue = info.argument(type=types.Float)
    qarg1: ir.SSAValue = info.argument(type=QubitType)
    qarg2: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class AtomLossChannel(ir.Statement):
    name = "native.atom_loss_channel"

    traits = frozenset({ir.FromPythonCall()})

    prob: ir.SSAValue = info.argument(type=types.Float)
    qarg: ir.SSAValue = info.argument(type=QubitType)
