from kirin import ir
from kirin.ir import types
from kirin.decl import info, statement
from kirin.print import Printer

from .._dialect import dialect as dialect


@statement(dialect=dialect)
class ConstInt(ir.Statement):
    """IR Statement representing a constant integer value."""

    name = "constant.int"
    traits = frozenset({ir.Pure(), ir.ConstantLike(), ir.FromPythonCall()})
    value: int = info.attribute(types.Int, property=True)
    """value (int): The constant integer value."""
    result: ir.ResultValue = info.result(types.Int)
    """result (Int): The result value."""

    def print_impl(self, printer: Printer) -> None:
        printer.print_name(self)
        printer.plain_print(" ")
        printer.plain_print(repr(self.value))
        with printer.rich(style=printer.color.comment):
            printer.plain_print(" : ")
            printer.print(self.result.type)


@statement(dialect=dialect)
class ConstFloat(ir.Statement):
    """IR Statement representing a constant float value."""

    name = "constant.float"
    traits = frozenset({ir.Pure(), ir.ConstantLike(), ir.FromPythonCall()})
    value: float = info.attribute(types.Float, property=True)
    """value (float): The constant float value."""
    result: ir.ResultValue = info.result(types.Float)
    """result (Float): The result value."""

    def print_impl(self, printer: Printer) -> None:
        printer.print_name(self)
        printer.plain_print(" ")
        printer.plain_print(repr(self.value))
        with printer.rich(style=printer.color.comment):
            printer.plain_print(" : ")
            printer.print(self.result.type)


@statement(dialect=dialect)
class ConstBool(ir.Statement):
    """IR Statement representing a constant float value."""

    name = "constant.bool"
    traits = frozenset({ir.Pure(), ir.ConstantLike(), ir.FromPythonCall()})
    value: bool = info.attribute(types.Bool, property=True)
    """value (float): The constant float value."""
    result: ir.ResultValue = info.result(types.Bool)
    """result (Float): The result value."""

    def print_impl(self, printer: Printer) -> None:
        printer.print_name(self)
        printer.plain_print(" ")
        printer.plain_print(repr(self.value))
        with printer.rich(style=printer.color.comment):
            printer.plain_print(" : ")
            printer.print(self.result.type)

@statement(dialect=dialect)
class ConstStr(ir.Statement):
    """IR Statement representing a constant str value."""

    name = "constant.str"
    traits = frozenset({ir.Pure(), ir.ConstantLike(), ir.FromPythonCall()})
    value: str = info.attribute(types.String, property=True)
    """value (str): The constant str value."""
    result: ir.ResultValue = info.result(types.String)
    """result (str): The result value."""

    def print_impl(self, printer: Printer) -> None:
        printer.print_name(self)
        printer.plain_print(" ")
        printer.plain_print(repr(self.value))
        with printer.rich(style=printer.color.comment):
            printer.plain_print(" : ")
            printer.print(self.result.type)

@statement(dialect=dialect)
class Neg(ir.Statement):
    """IR Statement representing a negation operation."""

    name = "neg"
    traits = frozenset({ir.FromPythonCall()})
    operand: ir.SSAValue = info.argument(types.Int)
    result: ir.ResultValue = info.result(types.Int)
