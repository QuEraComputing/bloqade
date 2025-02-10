from kirin import ir
from kirin.ir import types
from kirin.decl import info, statement

from ..types import RecordType, PauliStringType
from .._dialect import dialect

PyNum = types.Union(types.Int, types.Float)


@statement(dialect=dialect)
class GetRecord(ir.Statement):
    name = "get_rec"
    traits = frozenset({ir.FromPythonCall()})
    id: ir.SSAValue = info.argument(type=types.Int)
    result: ir.ResultValue = info.result(type=RecordType)


@statement(dialect=dialect)
class Detector(ir.Statement):
    name = "detector"
    traits = frozenset({ir.FromPythonCall()})
    coord: tuple[ir.SSAValue, ...] = info.argument(PyNum)
    targets: tuple[ir.SSAValue, ...] = info.argument(RecordType)


@statement(dialect=dialect)
class ObservableInclude(ir.Statement):
    name = "obs.include"
    traits = frozenset({ir.FromPythonCall()})
    idx: ir.SSAValue = info.argument(type=types.Int)
    targets: tuple[ir.SSAValue, ...] = info.argument(RecordType)


@statement(dialect=dialect)
class Tick(ir.Statement):
    name = "tick"
    traits = frozenset({ir.FromPythonCall()})


@statement(dialect=dialect)
class NewPauliString(ir.Statement):
    name = "new_pauli_string"
    traits = frozenset({ir.FromPythonCall()})
    string: tuple[ir.SSAValue, ...] = info.argument(types.String)
    flipped: tuple[ir.SSAValue, ...] = info.argument(types.Bool)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)
    result: ir.ResultValue = info.result(type=PauliStringType)
