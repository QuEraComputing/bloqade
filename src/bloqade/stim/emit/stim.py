from dataclasses import dataclass, field

from kirin import interp, ir
from kirin.dialects import func
from kirin.emit import EmitABC, EmitFrame

@dataclass
class EmitStimFrame(EmitFrame[str]):
    body: list[str] = field(default_factory=list)

def _default_dialect_group() -> ir.DialectGroup:
    from ..prelude import main
    return main

@dataclass
class EmitStimMain(EmitABC[EmitStimFrame, str|None]):
    void = ""
    keys = ["emit.stim"]
    output: str = field(default="") 
    dialects: ir.DialectGroup = field(default_factory=_default_dialect_group)

    def initialize(self):
        super().initialize()
        self.output=""
        return self

    def eval_stmt_fallback(
        self, frame: EmitStimFrame, stmt: ir.Statement
    ) -> tuple[str, ...]:
        return (stmt.name,)

    def new_frame(self, code: ir.Statement) -> EmitStimFrame:
        return EmitStimFrame.from_func_like(code)

    def run_method(
        self, method: ir.Method, args: tuple[str, ...]
    ) -> str | None:

        return self.run_callable(method.code, (method.sym_name,) + args)

    def emit_block(
        self, frame: EmitStimFrame, block: ir.Block
    ) -> str | None:
        for stmt in block.stmts:
            result = self.eval_stmt(frame, stmt)
            if isinstance(result, tuple):
                frame.set_values(stmt.results, result)
        return None


@func.dialect.register(key="emit.stim")
class FuncEmit(interp.MethodTable):

    @interp.impl(func.Function)
    def emit_func(self, emit: EmitStimMain, frame: EmitStimFrame, stmt: func.Function):
        result = emit.run_ssacfg_region(frame, stmt.body)
        emit.output = "\n".join(frame.body)
        return ()
