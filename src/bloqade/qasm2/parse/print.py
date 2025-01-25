from typing import IO, Any, Callable
from contextlib import contextmanager
from dataclasses import field, dataclass

from rich.console import Console
from bloqade.qasm2.parse.ast import Version

from .ast import (
    Pi,
    Bit,
    Cmp,
    Call,
    CReg,
    Gate,
    Name,
    QReg,
    BinOp,
    Reset,
    UGate,
    CXGate,
    IfStmt,
    Number,
    Opaque,
    Barrier,
    Comment,
    Include,
    Measure,
    UnaryOp,
    ParaCZGate,
    ParaRZGate,
    ParaU3Gate,
    Instruction,
    MainProgram,
    ParallelQArgs,
)
from .visitor import Visitor


@dataclass
class ColorScheme:
    comment: str = "bright_black"
    keyword: str = "red"
    symbol: str = "cyan"
    string: str = "yellow"
    number: str = "green"
    irrational: str = "magenta"


@dataclass
class PrintState:
    indent: int = 0
    result_width: int = 0
    rich_style: str | None = None
    rich_highlight: bool | None = False
    indent_marks: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class Printer(Visitor[None]):

    def __init__(
        self,
        file: IO | None = None,
        show_indent_mark: bool = True,
        color: ColorScheme | None = None,
    ) -> None:
        self.console = Console(file=file)
        self.state = PrintState()
        self.show_indent_mark = show_indent_mark
        self.color = color or ColorScheme()

    def plain_print(self, *objects, sep="", end="", style=None, highlight=None):
        self.console.out(
            *objects,
            sep=sep,
            end=end,
            style=style or self.state.rich_style,
            highlight=highlight or self.state.rich_highlight,
        )

    def print_indent(self):
        indent_str = ""
        if self.show_indent_mark and self.state.indent_marks:
            indent_str = "".join(
                "│" if i in self.state.indent_marks else " "
                for i in range(self.state.indent)
            )
            with self.rich(style=self.color.comment):
                self.plain_print(indent_str)
        else:
            indent_str = " " * self.state.indent
            self.plain_print(indent_str)

    def print_newline(self):
        self.plain_print("\n")

        if self.state.messages:
            for message in self.state.messages:
                self.plain_print(message)
                self.plain_print("\n")
            self.state.messages.clear()
        self.print_indent()

    def print_sequence(
        self,
        seq,
        sep: str = ", ",
        start: str = "",
        end: str = "",
        print: Callable[[Any], None] | None = None,
    ) -> None:
        if print is None:
            print = self.visit
        self.plain_print(start)
        for i, item in enumerate(seq):
            if i > 0:
                self.plain_print(sep)
            print(item)
        self.plain_print(end)

    @contextmanager
    def indent(self, increase: int = 2, mark: bool | None = None):
        mark = mark if mark is not None else self.show_indent_mark
        self.state.indent += increase
        if mark:
            self.state.indent_marks.append(self.state.indent)
        try:
            yield self.state
        finally:
            self.state.indent -= increase
            if mark:
                self.state.indent_marks.pop()

    @contextmanager
    def rich(self, style: str | None = None, highlight: bool = False):
        old_style = self.state.rich_style
        old_highlight = self.state.rich_highlight
        self.state.rich_style = style
        self.state.rich_highlight = highlight
        try:
            yield self.state
        finally:
            self.state.rich_style = old_style
            self.state.rich_highlight = old_highlight

    def visit_MainProgram(self, node: MainProgram) -> None:
        self.print_indent()
        self.visit_Version(node.version)
        self.print_newline()
        for stmt in node.statements:
            self.visit(stmt)
            self.print_newline()

    def visit_Version(self, node: Version) -> None:
        self.plain_print(
            f"OPENQASM {node.major}.{node.minor}", style=self.color.comment
        )
        if node.ext:
            self.plain_print("-", node.ext, style=self.color.comment)
        self.plain_print(";")

    def visit_Include(self, node: Include) -> None:
        self.plain_print("include", style=self.color.keyword)
        self.plain_print(" ")
        self.plain_print('"', node.filename, '"', style=self.color.string)
        self.plain_print(";")

    def visit_Barrier(self, node: Barrier) -> None:
        self.print_indent()
        self.plain_print("barrier", style=self.color.keyword)
        self.plain_print(" ")
        self.print_sequence(node.qargs)
        self.plain_print(";")

    def visit_Instruction(self, node: Instruction) -> None:
        self.visit_Name(node.name)
        self.plain_print(" ")
        if node.params:
            self.print_sequence(node.params, sep=", ", start="(", end=") ")
        self.print_sequence(node.qargs)
        self.plain_print(";")

    def visit_Comment(self, node: Comment) -> None:
        self.plain_print("// ", node.text, style=self.color.comment)

    def visit_CReg(self, node: CReg) -> None:
        self.plain_print("creg", style=self.color.keyword)
        self.plain_print(f" {node.name}[{node.size}]")
        self.plain_print(";")

    def visit_QReg(self, node: QReg) -> None:
        self.plain_print("qreg", style=self.color.keyword)
        self.plain_print(f" {node.name}[{node.size}]")
        self.plain_print(";")

    def visit_CXGate(self, node: CXGate) -> None:
        self.plain_print("CX", style=self.color.keyword)
        self.plain_print(" ")
        self.visit(node.ctrl)
        self.plain_print(", ")
        self.visit(node.qarg)
        self.plain_print(";")

    def visit_UGate(self, node: UGate) -> None:
        self.plain_print("U", style=self.color.keyword)
        self.plain_print("(")
        self.visit(node.theta)
        self.plain_print(", ")
        self.visit(node.phi)
        self.plain_print(", ")
        self.visit(node.lam)
        self.plain_print(") ")
        self.visit(node.qarg)
        self.plain_print(";")

    def visit_Measure(self, node: Measure) -> None:
        self.plain_print("measure", style=self.color.keyword)
        self.plain_print(" ")
        self.visit(node.qarg)
        self.plain_print(" -> ")
        self.visit(node.carg)
        self.plain_print(";")

    def visit_Reset(self, node: Reset) -> None:
        self.plain_print("reset ")
        self.visit(node.qarg)
        self.plain_print(";")

    def visit_Opaque(self, node: Opaque) -> None:
        self.plain_print("opaque ", style=self.color.keyword)
        if node.cparams:
            self.print_sequence(node.cparams, sep=", ", start="(", end=")")

        if node.qparams:
            self.plain_print(" ")
            self.print_sequence(node.qparams, sep=", ")
        self.plain_print(";")

    def visit_Gate(self, node: Gate) -> None:
        self.plain_print("gate ", style=self.color.keyword)
        self.plain_print(node.name, style=self.color.symbol)
        if node.cparams:
            self.print_sequence(
                node.cparams, sep=", ", start="(", end=")", print=self.plain_print
            )

        if node.qparams:
            self.plain_print(" ")
            self.print_sequence(node.qparams, sep=", ", print=self.plain_print)

        self.plain_print(" {")
        with self.indent():
            self.print_newline()
            for idx, stmt in enumerate(node.body):
                self.visit(stmt)
                if idx < len(node.body) - 1:
                    self.print_newline()
        self.print_newline()
        self.plain_print("}")

    def visit_IfStmt(self, node: IfStmt) -> None:
        self.plain_print("if", style=self.color.keyword)
        self.visit(node.cond)
        if len(node.body) == 1:  # inline if
            self.visit(node.body[0])
        else:
            self.plain_print("{")
            with self.indent():
                self.print_newline()
                for idx, stmt in enumerate(node.body):
                    self.visit(stmt)
                    if idx < len(node.body) - 1:
                        self.print_newline()
            self.print_newline()
            self.plain_print("}")

    def visit_Cmp(self, node: Cmp) -> None:
        self.plain_print(" (")
        self.visit(node.lhs)
        self.plain_print(" == ", style=self.color.keyword)
        self.visit(node.rhs)
        self.plain_print(") ")

    def visit_Call(self, node: Call) -> None:
        self.plain_print(node.name)
        self.print_sequence(node.args, sep=", ", start="(", end=")")

    def visit_BinOp(self, node: BinOp) -> None:
        self.plain_print("(")
        self.visit(node.lhs)
        self.plain_print(f" {node.op} ", style=self.color.keyword)
        self.visit(node.rhs)
        self.plain_print(")")

    def visit_UnaryOp(self, node: UnaryOp) -> None:
        self.plain_print(f"{node.op}", style=self.color.keyword)
        self.visit(node.operand)

    def visit_Bit(self, node: Bit) -> None:
        self.visit_Name(node.name)
        if node.addr is not None:
            self.plain_print("[")
            self.plain_print(node.addr, style=self.color.number)
            self.plain_print("]")

    def visit_Number(self, node: Number) -> None:
        self.plain_print(node.value)

    def visit_Pi(self, node: Pi) -> None:
        self.plain_print("pi", style=self.color.irrational)

    def visit_Name(self, node: Name) -> None:
        return self.plain_print(node.id, style=self.color.symbol)

    def visit_ParallelQArgs(self, node: ParallelQArgs) -> None:
        self.plain_print("{")
        with self.indent():
            for idx, qargs in enumerate(node.qargs):
                self.print_newline()
                self.print_sequence(qargs)
                self.plain_print(";")
        self.print_newline()
        self.plain_print("}")

    def visit_ParaU3Gate(self, node: ParaU3Gate) -> None:
        self.plain_print("parallel.U", style=self.color.keyword)
        self.plain_print("(")
        self.visit(node.theta)
        self.plain_print(", ")
        self.visit(node.phi)
        self.plain_print(", ")
        self.visit(node.lam)
        self.plain_print(") ")
        self.visit_ParallelQArgs(node.qargs)

    def visit_ParaCZGate(self, node: ParaCZGate) -> None:
        self.plain_print("parallel.CZ ", style=self.color.keyword)
        self.visit_ParallelQArgs(node.qargs)

    def visit_ParaRZGate(self, node: ParaRZGate) -> None:
        self.plain_print("parallel.RZ", style=self.color.keyword)
        self.plain_print("(")
        self.visit(node.theta)
        self.plain_print(") ")
        self.visit_ParallelQArgs(node.qargs)
