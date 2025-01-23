from __future__ import annotations

from typing import Union, Literal
from dataclasses import dataclass


@dataclass
class Node:
    pass


@dataclass
class MainProgram(Node):
    version: str
    statements: list[Statement]


@dataclass
class Statement(Node):
    pass


@dataclass
class QReg(Statement):
    name: str
    size: int


@dataclass
class CReg(Statement):
    name: str
    size: int


@dataclass
class Gate(Statement):
    name: str
    cparams: list[str]
    qparams: list[str]
    body: list[UOp | Barrier]


@dataclass
class Opaque(Statement):
    name: str
    cparams: list[Expr]
    qparams: list[Bit]


@dataclass
class QOp(Statement):
    pass


@dataclass
class IfStmt(Statement):
    cond: Cmp
    body: list[QOp]


@dataclass
class Barrier(Statement):
    qargs: list[Bit | Name]


@dataclass
class Include(Statement):
    filename: str


@dataclass
class Measure(QOp):
    qarg: Bit | Name
    carg: Bit | Name


@dataclass
class Reset(QOp):
    qarg: Bit | Name


@dataclass
class UOp(QOp):
    pass


@dataclass
class Instruction(UOp):
    name: Name
    params: list[Expr]
    qargs: list[Bit | Name]


@dataclass
class UGate(UOp):
    theta: Expr
    phi: Expr
    lam: Expr
    qarg: Bit | Name


@dataclass
class CXGate(UOp):
    ctrl: Bit | Name
    qarg: Bit | Name


@dataclass
class Extension(UOp):
    pass


@dataclass
class ParallelQArgs(Extension):
    qargs: list[tuple[Bit | Name, ...]]


@dataclass
class ParaU3Gate(Extension):
    theta: Expr
    phi: Expr
    lam: Expr
    qargs: ParallelQArgs


@dataclass
class ParaRzGate(Extension):
    theta: Expr
    qargs: ParallelQArgs


@dataclass
class ParaCZGate(Extension):
    qargs: ParallelQArgs


@dataclass
class Expr(Node):
    pass


@dataclass
class BinOp(Expr):
    op: Literal["+", "-", "*", "/", "^"]
    lhs: Expr
    rhs: Expr


@dataclass
class Cmp(Statement):
    lhs: Expr
    rhs: Expr


@dataclass
class UnaryOp(Expr):
    op: Literal["-", "+"]
    operand: Expr


@dataclass
class Call(Expr):
    name: Literal["sin", "cos", "tan", "exp", "ln", "sqrt"]
    args: list[Expr]


@dataclass
class Constant(Expr):
    pass


@dataclass
class Number(Constant):
    value: Union[int, float]


@dataclass
class Pi(Constant):
    pass


@dataclass
class Name(Expr):
    id: str


@dataclass
class Bit(Expr):
    name: Name
    addr: int
