"""Inline QASM dialect.

This dialect allows users to use QASM string as part of a `@qasm2.main` kernel.
"""

import ast
import textwrap
from dataclasses import dataclass

from kirin import ir, types, lowering
from kirin.decl import info, statement
from kirin.print import Printer
from kirin.exceptions import DialectLoweringError

dialect = ir.Dialect("qasm2.inline")


@dataclass(frozen=True)
class InlineQASMLowering(ir.FromPythonCall):

    def lower(
        self, stmt: type, state: lowering.LoweringState, node: ast.Call
    ) -> lowering.Result:
        from bloqade.qasm2.parse import loads
        from bloqade.qasm2.parse.lowering import LoweringQASM

        if len(node.args) != 1 or node.keywords:
            raise DialectLoweringError("InlineQASM takes 1 positional argument")
        text = node.args[0]
        # 1. string literal
        if isinstance(text, ast.Constant) and isinstance(text.value, str):
            value = text.value
        elif isinstance(text, ast.Name) and isinstance(text.ctx, ast.Load):
            value = state.get_global(text.id).expect(str)
        else:
            raise DialectLoweringError(
                "InlineQASM takes a string literal or global string"
            )

        raw = textwrap.dedent(value)
        qasm_lowering = LoweringQASM(state)
        qasm_lowering.visit(loads(raw))
        return lowering.Result()


# NOTE: this is a dummy statement that won't appear in IR.
# TODO: maybe we should save the string in IR then rewrite?
#       what would be the use case?
@statement(dialect=dialect)
class InlineQASM(ir.Statement):
    name = "text"
    traits = frozenset({InlineQASMLowering()})
    text: str = info.attribute(types.String)

    def __init__(self, text: str) -> None:
        super().__init__(properties={"text": ir.PyAttr(text)})

    def print_impl(self, printer: Printer) -> None:
        printer.print_name(self)
        printer.plain_print('"""')
        for line in self.text.splitlines():
            printer.plain_print(line)
            printer.print_newline()
        printer.plain_print('"""')
