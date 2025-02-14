from abc import ABC
from typing import Generic, TypeVar, overload
from dataclasses import field, dataclass

from kirin import ir, idtable
from kirin.emit import EmitABC, EmitError, EmitFrame
from typing_extensions import Self
from bloqade.qasm2.parse import ast

StmtType = TypeVar("StmtType", bound=ast.Node)
EmitNode = TypeVar("EmitNode", bound=ast.Node)


@dataclass
class EmitQASM2Frame(EmitFrame[ast.Node | None], Generic[StmtType]):
    body: list[StmtType] = field(default_factory=list)


@dataclass
class EmitQASM2Base(
    EmitABC[EmitQASM2Frame[StmtType], ast.Node | None], ABC, Generic[StmtType, EmitNode]
):
    void = None
    prefix: str = field(default="", kw_only=True)
    prefix_if_none: str = field(default="var_", kw_only=True)

    output: EmitNode | None = field(init=False)
    ssa_id: idtable.IdTable[ir.SSAValue] = field(init=False)

    def initialize(self) -> Self:
        super().initialize()
        self.output: EmitNode | None = None
        self.ssa_id = idtable.IdTable[ir.SSAValue](
            prefix=self.prefix, prefix_if_none=self.prefix_if_none
        )
        return self

    def new_frame(self, code: ir.Statement) -> EmitQASM2Frame:
        return EmitQASM2Frame.from_func_like(code)

    def run_method(
        self, method: ir.Method, args: tuple[ast.Node | None, ...]
    ) -> tuple[EmitQASM2Frame[StmtType], ast.Node | None]:
        return self.run_callable(method.code, (ast.Name(method.sym_name),) + args)

    def emit_block(self, frame: EmitQASM2Frame, block: ir.Block) -> ast.Node | None:
        for stmt in block.stmts:
            result = self.eval_stmt(frame, stmt)
            if isinstance(result, tuple):
                frame.set_values(stmt.results, result)
        return None

    A = TypeVar("A")
    B = TypeVar("B")

    @overload
    def assert_node(self, typ: type[A], node: ast.Node | None) -> A: ...

    @overload
    def assert_node(
        self, typ: tuple[type[A], type[B]], node: ast.Node | None
    ) -> A | B: ...

    def assert_node(
        self,
        typ: type[A] | tuple[type[A], type[B]],
        node: ast.Node | None,
    ) -> A | B:
        if not isinstance(node, typ):
            raise EmitError(f"expected {typ}, got {type(node)}")
        return node
