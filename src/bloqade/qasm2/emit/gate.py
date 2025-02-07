from dataclasses import field, dataclass

from kirin import ir, interp
from bloqade.types import QubitType
from kirin.dialects import func, ilist
from kirin.ir.dialect import Dialect as Dialect
from bloqade.qasm2.parse import ast

from .base import EmitError, EmitQASM2Base, EmitQASM2Frame


def _default_dialect_group():
    from bloqade.qasm2.groups import gate

    return gate


@dataclass
class EmitQASM2Gate(EmitQASM2Base[ast.UOp | ast.Barrier, ast.Gate]):
    keys = ["emit.qasm2.gate"]
    dialects: ir.DialectGroup = field(default_factory=_default_dialect_group)


@ilist.dialect.register(key="emit.qasm2.gate")
class Ilist(interp.MethodTable):

    @interp.impl(ilist.New)
    def emit_ilist(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: ilist.New):
        return (ilist.IList(data=[frame.get(value) for value in stmt.values]),)


@func.dialect.register(key="emit.qasm2.gate")
class Func(interp.MethodTable):

    @interp.impl(func.Function)
    def emit_func(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Function
    ):
        cparams, qparams = [], []
        for arg in stmt.body.blocks[0].args[1:]:
            name = frame.get(arg)
            if not isinstance(name, ast.Name):
                raise EmitError("expected ast.Name")
            if arg.type.is_subseteq(QubitType):
                qparams.append(name.id)
            else:
                cparams.append(name.id)
        emit.run_ssacfg_region(frame, stmt.body)
        emit.output = ast.Gate(
            name=stmt.sym_name,
            cparams=cparams,
            qparams=qparams,
            body=frame.body,
        )
        return ()

    @interp.impl(func.Call)
    def emit_call(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Call):
        raise EmitError("cannot emit dynamic call")

    @interp.impl(func.Invoke)
    def emit_invoke(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Invoke
    ):
        cparams, qparams = [], []
        for arg in stmt.inputs:
            if arg.type.is_subseteq(QubitType):
                qparams.append(frame.get(arg))
            else:
                cparams.append(frame.get(arg))

        frame.body.append(
            ast.Instruction(
                name=ast.Name(stmt.callee.sym_name),
                params=cparams,
                qargs=qparams,
            )
        )
        return ()

    @interp.impl(func.Lambda)
    @interp.impl(func.GetField)
    def emit_err(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt):
        raise EmitError(f"illegal statement {stmt.name} for QASM2 gate routine")

    @interp.impl(func.Return)
    @interp.impl(func.ConstantNone)
    def ignore(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt):
        return ()
