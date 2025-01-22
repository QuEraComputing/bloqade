from typing import Union

from kirin.interp import Interpreter, Frame, MethodTable, impl

from . import stmts
from ._dialect import dialect


@dialect.register
class Qasm2UopInterpreter(MethodTable):
    name = "qasm2.uop"
    dialect = dialect

    @impl(stmts.ConstFloat)
    @impl(stmts.ConstInt)
    def new_const(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: Union[stmts.ConstFloat, stmts.ConstInt],
    ):
        return (stmt.value,)

    @impl(stmts.ConstPI)
    def new_const_pi(self, interp: Interpreter, frame: Frame, stmt: stmts.ConstPI):
        return (3.141592653589793,)
