from kirin import interp
from bloqade.types import QubitType
from kirin.dialects import func
from kirin.exceptions import CodeGenError
from kirin.ir.dialect import Dialect as Dialect
from bloqade.qasm2.parse import ast

from .base import EmitQASM2Base, EmitQASM2Frame


class EmitQASM2Gate(EmitQASM2Base[ast.UOp | ast.Barrier, ast.Gate]):
    keys = ["emit.qasm2.gate"]

    def __init__(
        self,
        *,
        fuel: int | None = None,
        max_depth: int = 128,
        max_python_recursion_depth: int = 8192,
        prefix: str = "",
        prefix_if_none: str = "var_",
    ):
        from bloqade.qasm2.groups import gate

        super().__init__(
            gate,
            fuel=fuel,
            max_depth=max_depth,
            max_python_recursion_depth=max_python_recursion_depth,
            prefix=prefix,
            prefix_if_none=prefix_if_none,
        )


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
                raise CodeGenError("expected ast.Name")
            if arg.type.is_subseteq(QubitType):
                qparams.append(name.id)
            else:
                cparams.append(name.id)
        result = emit.run_ssacfg_region(frame, stmt.body)
        if isinstance(result, interp.Err):
            return result
        emit.output = ast.Gate(
            name=stmt.sym_name,
            cparams=cparams,
            qparams=qparams,
            body=frame.body,
        )
        return ()

    @interp.impl(func.Call)
    def emit_call(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: func.Call):
        return interp.Err(CodeGenError("cannot emit dynamic call"), emit.state.frames)

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
        return interp.Err(
            CodeGenError(f"illegal statement {stmt.name} for QASM2 gate routine"),
            emit.state.frames,
        )

    @interp.impl(func.Return)
    @interp.impl(func.ConstantNone)
    def ignore(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt):
        return ()
