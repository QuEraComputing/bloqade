from kirin import ir
from kirin.ir import types
from kirin.decl import info, statement

from .._dialect import dialect


@statement
class Reset(ir.Statement):
    name = "reset"
    traits = frozenset({ir.FromPythonCall()})
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class RZ(Reset):
    name = "RZ"


@statement(dialect=dialect)
class RY(Reset):
    name = "RY"


@statement(dialect=dialect)
class RX(Reset):
    name = "RX"
