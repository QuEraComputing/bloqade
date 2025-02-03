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
class CZUnpairedPauliChannel(ir.Statement):
    name = "native.pauli_channel.cz_unpaired"

    traits = frozenset({ir.FromPythonCall()})

    px: ir.SSAValue = info.argument(type=types.Float)
    py: ir.SSAValue = info.argument(type=types.Float)
    pz: ir.SSAValue = info.argument(type=types.Float)
    qarg1: ir.SSAValue = info.argument(type=QubitType)
    qarg2: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class CZPairedPauliChannel(ir.Statement):
    name = "native.pauli_channel.cz_unpaired"

    traits = frozenset({ir.FromPythonCall()})
    px: ir.SSAValue = info.argument(type=types.Float)
    py: ir.SSAValue = info.argument(type=types.Float)
    pz: ir.SSAValue = info.argument(type=types.Float)
    qarg1: ir.SSAValue = info.argument(type=QubitType)
    qarg2: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class AtomLossChannel(ir.Statement):
    name = "native.atom_loss_channel"

    traits = frozenset({ir.FromPythonCall()})

    prob: ir.SSAValue = info.argument(type=types.Float)
    qarg: ir.SSAValue = info.argument(type=QubitType)
