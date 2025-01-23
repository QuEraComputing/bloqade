from kirin import ir, types
from kirin.decl import info, statement
from kirin.print.printer import Printer

from ._dialect import dialect


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
class ConstPI(ir.Statement):
    """The constant value of PI."""

    # this is marked as constant but not pure.
    name = "constant.pi"
    traits = frozenset({ir.ConstantLike(), ir.FromPythonCall()})
    result: ir.ResultValue = info.result(types.Float)
    """result (ConstPI): The result value."""

    def print_impl(self, printer: Printer) -> None:
        printer.print_name(self)
        printer.plain_print(" ")
        printer.plain_print("PI")
        with printer.rich(style=printer.color.comment):
            printer.plain_print(" : ")
            printer.print(self.result.type)


# QASM 2.0 arithmetic operations
PyNum = types.Union(types.Int, types.Float)


@statement(dialect=dialect)
class Neg(ir.Statement):
    """Negate a number."""

    name = "neg"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to negate."""
    result: ir.ResultValue = info.result(PyNum)
    """result (Union[int, float]): The negated number."""


@statement(dialect=dialect)
class Sin(ir.Statement):
    """Take the sine of a number."""

    name = "sin"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to take the sine of."""
    result: ir.ResultValue = info.result(PyNum)
    """result (float): The sine of the number."""


@statement(dialect=dialect)
class Cos(ir.Statement):
    """Take the cosine of a number."""

    name = "cos"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to take the cosine of."""
    result: ir.ResultValue = info.result(PyNum)
    """result (float): The cosine of the number."""


@statement(dialect=dialect)
class Tan(ir.Statement):
    """Take the tangent of a number."""

    name = "tan"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to take the tangent of."""
    result: ir.ResultValue = info.result(PyNum)
    """result (float): The tangent of the number."""


@statement(dialect=dialect)
class Exp(ir.Statement):
    """Take the exponential of a number."""

    name = "exp"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to take the exponential of."""
    result: ir.ResultValue = info.result(PyNum)
    """result (float): The exponential of the number."""


@statement(dialect=dialect)
class Log(ir.Statement):
    """Take the natural log of a number."""

    name = "ln"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to take the natural log of."""
    result: ir.ResultValue = info.result(PyNum)
    """result (float): The natural log of the number."""


@statement(dialect=dialect)
class Sqrt(ir.Statement):
    """Take the square root of a number."""

    name = "sqrt"
    traits = frozenset({ir.FromPythonCall()})
    value: ir.SSAValue = info.argument(PyNum)
    """value (Union[int, float]): The number to take the square root of."""
    result: ir.ResultValue = info.result(PyNum)
    """result (float): The square root of the number."""


@statement(dialect=dialect)
class Add(ir.Statement):
    """Add two numbers."""

    name = "add"
    traits = frozenset({ir.FromPythonCall()})
    lhs: ir.SSAValue = info.argument(PyNum)
    """lhs (Union[int, float]): The left-hand side of the addition."""
    rhs: ir.SSAValue = info.argument(PyNum)
    """rhs (Union[int, float]): The right-hand side of the addition."""
    result: ir.ResultValue = info.result(PyNum)
    """result (Union[int, float]): The result of the addition."""


@statement(dialect=dialect)
class Sub(ir.Statement):
    """Subtract two numbers."""

    name = "sub"
    traits = frozenset({ir.FromPythonCall()})
    lhs: ir.SSAValue = info.argument(PyNum)
    """lhs (Union[int, float]): The left-hand side of the subtraction."""
    rhs: ir.SSAValue = info.argument(PyNum)
    """rhs (Union[int, float]): The right-hand side of the subtraction."""
    result: ir.ResultValue = info.result(PyNum)
    """result (Union[int, float]): The result of the subtraction."""


@statement(dialect=dialect)
class Mul(ir.Statement):
    """Multiply two numbers."""

    name = "mul"
    traits = frozenset({ir.FromPythonCall()})
    lhs: ir.SSAValue = info.argument(PyNum)
    """lhs (Union[int, float]): The left-hand side of the multiplication."""
    rhs: ir.SSAValue = info.argument(PyNum)
    """rhs (Union[int, float]): The right-hand side of the multiplication."""
    result: ir.ResultValue = info.result(PyNum)
    """result (Union[int, float]): The result of the multiplication."""


@statement(dialect=dialect)
class Pow(ir.Statement):
    """Take the power of a number."""

    name = "pow"
    traits = frozenset({ir.FromPythonCall()})
    lhs: ir.SSAValue = info.argument(PyNum)
    """lhs (Union[int, float]): The base."""
    rhs: ir.SSAValue = info.argument(PyNum)
    """rhs (Union[int, float]): The exponent."""
    result: ir.ResultValue = info.result(PyNum)
    """result (Union[int, float]): The result of the power operation."""


@statement(dialect=dialect)
class Div(ir.Statement):
    """Divide two numbers."""

    name = "div"
    traits = frozenset({ir.FromPythonCall()})
    lhs: ir.SSAValue = info.argument(PyNum)
    """lhs (Union[int, float]): The numerator."""
    rhs: ir.SSAValue = info.argument(PyNum)
    """rhs (Union[int, float]): The denominator."""
    result: ir.ResultValue = info.result(PyNum)
    """result (Union[int, float]): The result of the division."""
