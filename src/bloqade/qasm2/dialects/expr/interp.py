import math
from typing import Union

from kirin.interp import Frame, Interpreter, MethodTable, impl

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

    @impl(stmts.Add)
    def add(self, interp: Interpreter, frame: Frame, stmt: stmts.Add):
        return (frame.get(stmt.lhs) + frame.get(stmt.rhs),)

    @impl(stmts.Sub)
    def sub(self, interp: Interpreter, frame: Frame, stmt: stmts.Sub):
        return (frame.get(stmt.lhs) - frame.get(stmt.rhs),)

    @impl(stmts.Mul)
    def mul(self, interp: Interpreter, frame: Frame, stmt: stmts.Mul):
        return (frame.get(stmt.lhs) * frame.get(stmt.rhs),)

    @impl(stmts.Div)
    def div(self, interp: Interpreter, frame: Frame, stmt: stmts.Div):
        return (frame.get(stmt.lhs) / frame.get(stmt.rhs),)

    @impl(stmts.Pow)
    def pow(self, interp: Interpreter, frame: Frame, stmt: stmts.Pow):
        return (frame.get(stmt.lhs) ** frame.get(stmt.rhs),)

    @impl(stmts.Neg)
    def neg(self, interp: Interpreter, frame: Frame, stmt: stmts.Neg):
        return (-frame.get(stmt.value),)

    @impl(stmts.Sqrt)
    def sqrt(self, interp: Interpreter, frame: Frame, stmt: stmts.Sqrt):
        return (math.sqrt(frame.get(stmt.value)),)

    @impl(stmts.Sin)
    def sin(self, interp: Interpreter, frame: Frame, stmt: stmts.Sin):
        return (math.sin(frame.get(stmt.value)),)

    @impl(stmts.Cos)
    def cos(self, interp: Interpreter, frame: Frame, stmt: stmts.Cos):
        return (math.cos(frame.get(stmt.value)),)

    @impl(stmts.Tan)
    def tan(self, interp: Interpreter, frame: Frame, stmt: stmts.Tan):
        return (math.tan(frame.get(stmt.value)),)

    @impl(stmts.Log)
    def log(self, interp: Interpreter, frame: Frame, stmt: stmts.Log):
        return (math.log(frame.get(stmt.value)),)

    @impl(stmts.Exp)
    def exp(self, interp: Interpreter, frame: Frame, stmt: stmts.Exp):
        return (math.exp(frame.get(stmt.value)),)
