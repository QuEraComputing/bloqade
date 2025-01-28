from kirin import ir
from kirin.ir import types
from kirin.decl import info, statement

from .._dialect import dialect
from ...aux.types import PauliStringType


@statement(dialect=dialect)
class PPMeasurement(ir.Statement):
    name = "MPP"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(types.Float)
    """probability of noise introduced by measurement. For example 0.01 means 1% the measurement will be flipped"""
    targets: tuple[ir.SSAValue, ...] = info.argument(PauliStringType)
