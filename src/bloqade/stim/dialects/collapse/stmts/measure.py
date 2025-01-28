from kirin import ir
from kirin.ir import types
from kirin.decl import info, statement

from .._dialect import dialect


@statement
class Measurement(ir.Statement):
    name = "measurement"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(types.Float)
    """probability of noise introduced by measurement. For example 0.01 means 1% the measurement will be flipped"""
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


# 1Q measurements
@statement(dialect=dialect)
class MZ(Measurement):
    name = "MZ"


@statement(dialect=dialect)
class MY(Measurement):
    name = "MY"


@statement(dialect=dialect)
class MX(Measurement):
    name = "MX"


# Pair measurements
@statement(dialect=dialect)
class MZZ(Measurement):
    name = "MZZ"


@statement(dialect=dialect)
class MYY(Measurement):
    name = "MYY"


@statement(dialect=dialect)
class MXX(Measurement):
    name = "MXX"
