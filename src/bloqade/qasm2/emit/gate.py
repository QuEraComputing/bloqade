from dataclasses import field, dataclass

from kirin import ir, types, interp
from bloqade.types import QubitType
from kirin.dialects import py, func, ilist
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
        return (ilist.IList(data=frame.get_values(stmt.values)),)


@py.constant.dialect.register(key="emit.qasm2.gate")
class Constant(interp.MethodTable):

    @interp.impl(py.Constant)
    def emit_constant(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: py.Constant
    ):
        return (stmt.value,)


@func.dialect.register(key="emit.qasm2.gate")
class Func(interp.MethodTable):

    @interp.impl(func.Call)
    def emit_call(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Call):
        raise EmitError("cannot emit dynamic call")

    @interp.impl(func.Invoke)
    def emit_invoke(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Invoke
    ):
        ret = ()
        if len(stmt.results) == 1 and stmt.results[0].type.is_subseteq(types.NoneType):
            ret = (None,)
        elif len(stmt.results) > 0:
            raise EmitError(
                "cannot emit invoke with results, this "
                "is not compatible QASM2 gate routine"
                " (consider pass qreg/creg by argument)"
            )

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
        return ret

    @interp.impl(func.Lambda)
    @interp.impl(func.GetField)
    def emit_err(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt):
        raise EmitError(f"illegal statement {stmt.name} for QASM2 gate routine")

    @interp.impl(func.Return)
    @interp.impl(func.ConstantNone)
    def ignore(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt):
        return ()

    @interp.impl(func.Function)
    def emit_func(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Function
    ):
        emit.run_ssacfg_region(frame, stmt.body)
        cparams, qparams = [], []
        for arg in stmt.args:
            if arg.type.is_subseteq(QubitType):
                qparams.append(frame.get(arg))
            else:
                cparams.append(frame.get(arg))
        emit.output = ast.Gate(stmt.sym_name, cparams, qparams, frame.body)
        return ()
