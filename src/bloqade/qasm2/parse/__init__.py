import pathlib
from typing import IO

from . import ast as ast
from .build import Build
from .print import Printer as Printer
from .parser import qasm2_parser as lark_parser
from .visitor import Visitor as Visitor


def loads(txt: str):
    raw = lark_parser.parse(txt)
    return Build().build_mainprogram(raw)


def loadfile(file: str | pathlib.Path):
    with open(file) as f:
        return loads(f.read())


def pprint(node: ast.Node, file: IO | None = None):
    Printer(file=file).visit(node)
