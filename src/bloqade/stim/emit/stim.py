from io import StringIO
from typing import IO, TypeVar
from dataclasses import field, dataclass

from kirin import ir, interp
from kirin.emit import EmitStr, EmitStrFrame
from kirin.dialects import func

# @dataclass
# class EmitStimFrame(EmitFrame[str]):
#     body: list[str] = field(default_factory=list)

IO_t = TypeVar("IO_t", bound=IO)


def _default_dialect_group() -> ir.DialectGroup:
    from ..groups import main

    return main


@dataclass
class EmitStimMain(EmitStr[IO_t]):
    keys = ["emit.stim"]
    dialects: ir.DialectGroup = field(default_factory=_default_dialect_group)
    file: IO_t = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.output = ""
        self.file = StringIO(self.output)

    def initialize(self):
        super().initialize()
        self.file.truncate(0)
        self.file.seek(0)
        return self

    def eval_stmt_fallback(
        self, frame: EmitStrFrame, stmt: ir.Statement
    ) -> tuple[str, ...]:
        return (stmt.name,)

    def emit_block(self, frame: EmitStrFrame, block: ir.Block) -> str | None:
        for stmt in block.stmts:
            result = self.eval_stmt(frame, stmt)
            if isinstance(result, tuple):
                frame.set_values(stmt.results, result)
        return None

    def get_output(self) -> str:
        self.file.seek(0)
        return self.file.read()


@func.dialect.register(key="emit.stim")
class FuncEmit(interp.MethodTable):

    @interp.impl(func.Function)
    def emit_func(self, emit: EmitStimMain, frame: EmitStrFrame, stmt: func.Function):
        _ = emit.run_ssacfg_region(frame, stmt.body)
        # emit.output = "\n".join(frame.body)
        return ()
