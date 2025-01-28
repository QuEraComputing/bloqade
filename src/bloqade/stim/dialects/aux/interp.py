from kirin import interp

from . import stmts
from ._dialect import dialect
from .types import RecordResult


@dialect.register
class StimAuxMethods(interp.MethodTable):

    @interp.impl(stmts.ConstFloat)
    @interp.impl(stmts.ConstInt)
    def const(
        self,
        interpreter: interp.Interpreter,
        frame: interp.Frame,
        stmt: stmts.ConstFloat | stmts.ConstInt,
    ):
        return (stmt.value,)

    @interp.impl(stmts.Neg)
    def neg(
        self,
        interpreter: interp.Interpreter,
        frame: interp.Frame,
        stmt: stmts.Neg,
    ):
        return (-frame.get(stmt.operand),)

    @interp.impl(stmts.GetRecord)
    def get_rec(
        self,
        interpreter: interp.Interpreter,
        frame: interp.Frame,
        stmt: stmts.GetRecord,
    ):
        return (RecordResult(value=frame.get(stmt.id)),)
