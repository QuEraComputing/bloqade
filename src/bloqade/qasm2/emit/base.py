from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar, overload

from kirin import idtable, interp, ir
from kirin.emit import EmitABC, EmitFrame
from kirin.exceptions import CodeGenError, InterpreterError

from bloqade.qasm2.parse import ast

StmtType = TypeVar("StmtType", bound=ast.Node)
EmitNode = TypeVar("EmitNode", bound=ast.Node)


@dataclass
class EmitQASM2Frame(EmitFrame[ast.Node | None], Generic[StmtType]):
    body: list[StmtType] = field(default_factory=list)


class EmitQASM2Base(
    EmitABC[EmitQASM2Frame[StmtType], ast.Node | None], Generic[StmtType, EmitNode]
):

    def __init__(
        self,
        dialects: ir.DialectGroup | Iterable[ir.Dialect],
        *,
        fuel: int | None = None,
        max_depth: int = 128,
        max_python_recursion_depth: int = 8192,
        prefix: str = "",
        prefix_if_none: str = "var_",
    ):
        super().__init__(
            dialects,
            bottom=None,
            fuel=fuel,
            max_depth=max_depth,
            max_python_recursion_depth=max_python_recursion_depth,
        )
        self.output: EmitNode | None = None
        self.ssa_id = idtable.IdTable[ir.SSAValue](
            prefix=prefix, prefix_if_none=prefix_if_none
        )

    def new_frame(self, code: ir.Statement) -> EmitQASM2Frame:
        return EmitQASM2Frame.from_func_like(code)

    def run_method(
        self, method: ir.Method, args: tuple[ast.Node, ...]
    ) -> interp.MethodResult[ast.Node | None]:
        if len(self.state.frames) >= self.max_depth:
            raise InterpreterError("maximum recursion depth exceeded")
        return self.run_callable(method.code, (ast.Name(method.sym_name),) + args)

    def emit_block(
        self, frame: EmitQASM2Frame, block: ir.Block
    ) -> interp.MethodResult[ast.Node | None]:
        for stmt in block.stmts:
            result = self.run_stmt(frame, stmt)
            if isinstance(result, interp.Err):
                return result
            elif isinstance(result, tuple):
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
            raise CodeGenError(f"expected {typ}, got {type(node)}")
        return node
