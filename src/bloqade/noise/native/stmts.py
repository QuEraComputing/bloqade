from kirin import ir, types
from kirin.decl import info, statement
from bloqade.qasm2.types import QubitType

from ._dialect import dialect


@statement(dialect=dialect)
class SingleQubitErrorChannel(ir.Statement):
    px: ir.SSAValue = info.argument(type=types.Float)
    py: ir.SSAValue = info.argument(type=types.Float)
    pz: ir.SSAValue = info.argument(type=types.Float)
    qarg: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class CZPauliUnpaired(ir.Statement):
    px: ir.SSAValue = info.argument(type=types.Float)
    py: ir.SSAValue = info.argument(type=types.Float)
    pz: ir.SSAValue = info.argument(type=types.Float)
    qarg1: ir.SSAValue = info.argument(type=QubitType)
    qarg2: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class CZErrorPaired(ir.Statement):
    px: ir.SSAValue = info.argument(type=types.Float)
    py: ir.SSAValue = info.argument(type=types.Float)
    pz: ir.SSAValue = info.argument(type=types.Float)
    qarg1: ir.SSAValue = info.argument(type=QubitType)
    qarg2: ir.SSAValue = info.argument(type=QubitType)


@statement(dialect=dialect)
class AtomLossChannel(ir.Statement):
    prob: ir.SSAValue = info.argument(type=types.Float)
    qarg: ir.SSAValue = info.argument(type=QubitType)
