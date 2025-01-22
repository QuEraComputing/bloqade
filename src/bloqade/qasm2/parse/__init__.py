from typing import IO

from . import ast as ast
from .build import Build
from .parser import qasm2_parser as lark_parser
from .print import Printer as Printer
from .visitor import Visitor as Visitor


def loads(txt: str):
    raw = lark_parser.parse(txt)
    return Build().build(raw)


def pprint(node: ast.Node, file: IO | None = None):
    Printer(file=file).visit(node)
