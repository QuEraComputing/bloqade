from typing import Literal

from kirin import interp

from bloqade.qasm2.parse import ast
from bloqade.qasm2.emit.gate import EmitQASM2Frame, EmitQASM2Gate

from . import stmts
from ._dialect import dialect


@dialect.register(key="emit.qasm2.gate")
class EmitExpr(interp.MethodTable):

    @interp.impl(stmts.ConstInt)
    @interp.impl(stmts.ConstFloat)
    def emit_const_int(
        self,
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt: stmts.ConstInt | stmts.ConstFloat,
    ):
        return (ast.Number(stmt.value),)

    @interp.impl(stmts.ConstPI)
    def emit_const_pi(
        self,
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt: stmts.ConstPI,
    ):
        return (ast.Pi(),)

    @interp.impl(stmts.Neg)
    def emit_neg(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.Neg):
        arg = emit.assert_node(ast.Expr, frame.get(stmt.value))
        return (ast.UnaryOp("-", arg),)

    @interp.impl(stmts.Sin)
    @interp.impl(stmts.Cos)
    @interp.impl(stmts.Tan)
    @interp.impl(stmts.Exp)
    @interp.impl(stmts.Log)
    @interp.impl(stmts.Sqrt)
    def emit_sin(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt):
        arg = emit.assert_node(ast.Expr, frame.get(stmt.value))
        return (ast.Call(stmt.name, [arg]),)

    def emit_binop(
        self,
        sym: Literal["+", "-", "*", "/", "^"],
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt,
    ):
        lhs = emit.assert_node(ast.Expr, frame.get(stmt.lhs))
        rhs = emit.assert_node(ast.Expr, frame.get(stmt.rhs))
        return (ast.BinOp(sym, lhs, rhs),)

    @interp.impl(stmts.Add)
    def emit_add(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.Add):
        return self.emit_binop("+", emit, frame, stmt)

    @interp.impl(stmts.Sub)
    def emit_sub(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.Add):
        return self.emit_binop("-", emit, frame, stmt)

    @interp.impl(stmts.Mul)
    def emit_mul(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.Add):
        return self.emit_binop("*", emit, frame, stmt)

    @interp.impl(stmts.Div)
    def emit_div(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.Add):
        return self.emit_binop("/", emit, frame, stmt)

    @interp.impl(stmts.Pow)
    def emit_pow(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.Add):
        return self.emit_binop("^", emit, frame, stmt)
