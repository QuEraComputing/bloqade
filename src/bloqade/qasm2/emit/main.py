from dataclasses import field, dataclass

from kirin import ir, interp
from kirin.dialects import cf, func
from kirin.ir.dialect import Dialect as Dialect
from bloqade.qasm2.parse import ast

from .base import EmitQASM2Base, EmitQASM2Frame


def _default_dialect_group():
    from bloqade.qasm2.groups import main

    return main


@dataclass
class EmitQASM2Main(EmitQASM2Base[ast.Statement, ast.MainProgram]):
    keys = ["emit.qasm2.main", "emit.qasm2.gate"]
    dialects: ir.DialectGroup = field(default_factory=_default_dialect_group)


@func.dialect.register(key="emit.qasm2.main")
class Func(interp.MethodTable):

    @interp.impl(func.Function)
    def emit_func(
        self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: func.Function
    ):
        from bloqade.qasm2.dialects import uop, core, expr

        emit.run_ssacfg_region(frame, stmt.body)
        # TODO: support difference in DialectGroup
        if not emit.dialects.data.difference({core.dialect, uop.dialect, expr.dialect}):
            header = ast.OPENQASM(ast.Version(2, 0))
        else:
            header = ast.Kirin([dialect.name for dialect in emit.dialects])
        emit.output = ast.MainProgram(header=header, statements=frame.body)
        return ()


@cf.dialect.register(key="emit.qasm2.main")
class Cf(interp.MethodTable):

    @interp.impl(cf.Branch)
    def emit_branch(self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: cf.Branch):
        frame.worklist.append(
            interp.Successor(stmt.successor, frame.get_values(stmt.arguments))
        )
        return ()

    @interp.impl(cf.ConditionalBranch)
    def emit_conditional_branch(
        self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: cf.ConditionalBranch
    ):
        cond = emit.assert_node(ast.Cmp, frame.get(stmt.cond))
        body_frame = emit.new_frame(stmt)
        body_frame.entries.update(frame.entries)
        body_frame.set_values(
            stmt.then_successor.args, frame.get_values(stmt.then_arguments)
        )
        emit.emit_block(body_frame, stmt.then_successor)
        frame.body.append(
            ast.IfStmt(
                cond,
                body=body_frame.body,  # type: ignore
            )
        )
        frame.worklist.append(
            interp.Successor(stmt.else_successor, frame.get_values(stmt.else_arguments))
        )
        return ()
