from dataclasses import field, dataclass

from kirin import print

from .ast import (
    OPENQASM,
    Pi,
    Bit,
    Cmp,
    Call,
    CReg,
    Gate,
    Name,
    QReg,
    BinOp,
    Kirin,
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
    GlobUGate,
    ParaCZGate,
    ParaRZGate,
    ParaU3Gate,
    Instruction,
    MainProgram,
    NoisePAULI1,
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


class Printer(print.Printer, Visitor[None]):

    def visit_MainProgram(self, node: MainProgram) -> None:
        self.print_indent()
        self.visit(node.header)
        self.print_newline()
        for stmt in node.statements:
            self.visit(stmt)
            self.print_newline()

    def visit_OPENQASM(self, node: OPENQASM) -> None:
        self.plain_print(
            f"OPENQASM {node.version.major}.{node.version.minor}",
            style="comment",
        )
        self.plain_print(";")

    def visit_Kirin(self, node: Kirin) -> None:
        self.plain_print(
            "KIRIN " + "{" + ",".join(sorted(node.dialects)) + "}", style="comment"
        )
        self.plain_print(";")

    def visit_Include(self, node: Include) -> None:
        self.plain_print("include", style="keyword")
        self.plain_print(" ")
        self.plain_print('"', node.filename, '"', style="string")
        self.plain_print(";")

    def visit_Barrier(self, node: Barrier) -> None:
        self.print_indent()
        self.plain_print("barrier", style="keyword")
        self.plain_print(" ")
        self.print_seq(node.qargs, emit=self.visit)
        self.plain_print(";")

    def visit_Instruction(self, node: Instruction) -> None:
        self.visit_Name(node.name)
        self.plain_print(" ")
        if node.params:
            self.print_seq(
                node.params, delim=", ", prefix="(", suffix=") ", emit=self.visit
            )
        self.print_seq(node.qargs, emit=self.visit)
        self.plain_print(";")

    def visit_Comment(self, node: Comment) -> None:
        self.plain_print("// ", node.text, style="comment")

    def visit_CReg(self, node: CReg) -> None:
        self.plain_print("creg", style="keyword")
        self.plain_print(f" {node.name}[{node.size}]")
        self.plain_print(";")

    def visit_QReg(self, node: QReg) -> None:
        self.plain_print("qreg", style="keyword")
        self.plain_print(f" {node.name}[{node.size}]")
        self.plain_print(";")

    def visit_CXGate(self, node: CXGate) -> None:
        self.plain_print("CX", style="keyword")
        self.plain_print(" ")
        self.visit(node.ctrl)
        self.plain_print(", ")
        self.visit(node.qarg)
        self.plain_print(";")

    def visit_UGate(self, node: UGate) -> None:
        self.plain_print("U", style="keyword")
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
        self.plain_print("measure", style="keyword")
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
        self.plain_print("opaque ", style="keyword")
        if node.cparams:
            self.print_seq(
                node.cparams, delim=", ", prefix="(", suffix=")", emit=self.visit
            )

        if node.qparams:
            self.plain_print(" ")
            self.print_seq(node.qparams, delim=", ", emit=self.visit)
        self.plain_print(";")

    def visit_Gate(self, node: Gate) -> None:
        self.plain_print("gate ", style="keyword")
        self.plain_print(node.name, style="symbol")
        if node.cparams:
            self.print_seq(
                node.cparams, delim=", ", prefix="(", suffix=")", emit=self.plain_print
            )

        if node.qparams:
            self.plain_print(" ")
            self.print_seq(node.qparams, delim=", ", emit=self.plain_print)

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
        self.plain_print("if", style="keyword")
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
        self.plain_print(" == ", style="keyword")
        self.visit(node.rhs)
        self.plain_print(") ")

    def visit_Call(self, node: Call) -> None:
        self.plain_print(node.name)
        self.print_seq(node.args, delim=", ", prefix="(", suffix=")", emit=self.visit)

    def visit_BinOp(self, node: BinOp) -> None:
        self.plain_print("(")
        self.visit(node.lhs)
        self.plain_print(f" {node.op} ", style="keyword")
        self.visit(node.rhs)
        self.plain_print(")")

    def visit_UnaryOp(self, node: UnaryOp) -> None:
        self.plain_print(f"{node.op}", style="keyword")
        self.visit(node.operand)

    def visit_Bit(self, node: Bit) -> None:
        self.visit_Name(node.name)
        if node.addr is not None:
            self.plain_print("[")
            self.plain_print(node.addr, style="number")
            self.plain_print("]")

    def visit_Number(self, node: Number) -> None:
        self.plain_print(node.value)

    def visit_Pi(self, node: Pi) -> None:
        self.plain_print("pi", style="number")

    def visit_Name(self, node: Name) -> None:
        return self.plain_print(node.id, style="symbol")

    def visit_ParallelQArgs(self, node: ParallelQArgs) -> None:
        self.plain_print("{")
        with self.indent():
            for idx, qargs in enumerate(node.qargs):
                self.print_newline()
                self.print_seq(qargs, emit=self.visit)
                self.plain_print(";")
        self.print_newline()
        self.plain_print("}")

    def visit_ParaU3Gate(self, node: ParaU3Gate) -> None:
        self.plain_print("parallel.U", style="keyword")
        self.plain_print("(")
        self.visit(node.theta)
        self.plain_print(", ")
        self.visit(node.phi)
        self.plain_print(", ")
        self.visit(node.lam)
        self.plain_print(") ")
        self.visit_ParallelQArgs(node.qargs)

    def visit_ParaCZGate(self, node: ParaCZGate) -> None:
        self.plain_print("parallel.CZ ", style="keyword")
        self.visit_ParallelQArgs(node.qargs)

    def visit_ParaRZGate(self, node: ParaRZGate) -> None:
        self.plain_print("parallel.RZ", style="keyword")
        self.plain_print("(")
        self.visit(node.theta)
        self.plain_print(") ")
        self.visit_ParallelQArgs(node.qargs)

    def visit_GlobUGate(self, node: GlobUGate) -> None:
        self.plain_print("glob.U", style="keyword")
        self.plain_print("(")
        self.visit(node.theta)
        self.plain_print(", ")
        self.visit(node.phi)
        self.plain_print(", ")
        self.visit(node.lam)
        self.plain_print(") ")
        self.print_seq(node.registers, prefix="{", suffix="}", emit=self.visit)

    def visit_NoisePAULI1(self, node: NoisePAULI1) -> None:
        self.plain_print("noise.PAULI1", style="keyword")
        self.plain_print("(")
        self.visit(node.px)
        self.plain_print(", ")
        self.visit(node.py)
        self.plain_print(", ")
        self.visit(node.pz)
        self.plain_print(") ")
        self.visit(node.qarg)
        self.plain_print(";")
