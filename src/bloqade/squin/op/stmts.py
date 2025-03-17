from kirin import ir, types
from kirin.decl import info, statement

from .types import OpType
from .traits import Sized, HasSize, Unitary, MaybeUnitary
from ._dialect import dialect


@statement
class Operator(ir.Statement):
    pass


@statement
class PrimitiveOp(Operator):
    pass


@statement
class CompositeOp(Operator):
    pass


@statement
class BinaryOp(CompositeOp):
    lhs: ir.SSAValue = info.argument(OpType)
    rhs: ir.SSAValue = info.argument(OpType)
    result: ir.ResultValue = info.result(OpType)


@statement(dialect=dialect)
class Kron(BinaryOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), MaybeUnitary()})
    is_unitary: bool = info.attribute(default=False)


@statement(dialect=dialect)
class Mult(BinaryOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), MaybeUnitary()})
    is_unitary: bool = info.attribute(default=False)


@statement(dialect=dialect)
class Adjoint(CompositeOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), MaybeUnitary()})
    is_unitary: bool = info.attribute(default=False)
    op: ir.SSAValue = info.argument(OpType)
    result: ir.ResultValue = info.result(OpType)


@statement(dialect=dialect)
class Scale(CompositeOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), MaybeUnitary()})
    is_unitary: bool = info.attribute(default=False)
    op: ir.SSAValue = info.argument(OpType)
    factor: ir.SSAValue = info.argument(types.Complex)
    result: ir.ResultValue = info.result(OpType)


@statement(dialect=dialect)
class Control(CompositeOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), MaybeUnitary()})
    is_unitary: bool = info.attribute(default=False)
    op: ir.SSAValue = info.argument(OpType)
    n_controls: int = info.attribute()
    result: ir.ResultValue = info.result(OpType)


@statement(dialect=dialect)
class Rot(CompositeOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), Unitary()})
    axis: ir.SSAValue = info.argument(OpType)
    angle: ir.SSAValue = info.argument(types.Float)
    result: ir.ResultValue = info.result(OpType)


@statement(dialect=dialect)
class Identity(CompositeOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), Unitary(), HasSize()})
    size: int = info.attribute()
    result: ir.ResultValue = info.result(OpType)


@statement
class ConstantOp(PrimitiveOp):
    traits = frozenset({ir.Pure(), ir.FromPythonCall(), ir.ConstantLike(), Sized(1)})
    result: ir.ResultValue = info.result(OpType)


@statement
class ConstantUnitary(ConstantOp):
    traits = frozenset(
        {ir.Pure(), ir.FromPythonCall(), ir.ConstantLike(), Unitary(), Sized(1)}
    )


@statement(dialect=dialect)
class PhaseOp(PrimitiveOp):
    """
    A phase operator.

    $$
    PhaseOp(theta) = e^{i \theta} I
    $$
    """

    traits = frozenset({ir.Pure(), ir.FromPythonCall(), Unitary(), Sized(1)})
    theta: ir.SSAValue = info.argument(types.Float)
    result: ir.ResultValue = info.result(OpType)


@statement(dialect=dialect)
class ShiftOp(PrimitiveOp):
    """
    A phase shift operator.

    $$
    Shift(theta) = \\begin{bmatrix} 1 & 0 \\\\ 0 & e^{i \\theta} \\end{bmatrix}
    $$
    """

    traits = frozenset({ir.Pure(), ir.FromPythonCall(), Unitary(), Sized(1)})
    theta: ir.SSAValue = info.argument(types.Float)
    result: ir.ResultValue = info.result(OpType)


@statement
class PauliOp(ConstantUnitary):
    pass


@statement(dialect=dialect)
class X(PauliOp):
    pass


@statement(dialect=dialect)
class Y(PauliOp):
    pass


@statement(dialect=dialect)
class Z(PauliOp):
    pass


@statement(dialect=dialect)
class H(ConstantUnitary):
    pass


@statement(dialect=dialect)
class S(ConstantUnitary):
    pass


@statement(dialect=dialect)
class T(ConstantUnitary):
    pass


@statement(dialect=dialect)
class P0(ConstantOp):
    """
    The $P_0$ projection operator.

    $$
    P0 = \\begin{bmatrix} 1 & 0 \\\\ 0 & 0 \\end{bmatrix}
    $$
    """

    pass


@statement(dialect=dialect)
class P1(ConstantOp):
    """
    The $P_1$ projection operator.

    $$
    P1 = \\begin{bmatrix} 0 & 0 \\\\ 0 & 1 \\end{bmatrix}
    $$
    """

    pass


@statement(dialect=dialect)
class Sn(ConstantOp):
    """
    $S_{-}$ operator.

    $$
    Sn = \\frac{1}{2} (S_x - i S_y) = \\frac{1}{2} \\begin{bmatrix} 0 & 0 \\\\ 1 & 0 \\end{bmatrix}
    $$
    """

    pass


@statement(dialect=dialect)
class Sp(ConstantOp):
    """
    $S_{+}$ operator.

    $$
    Sp = \\frac{1}{2} (S_x + i S_y) = \\frac{1}{2}\\begin{bmatrix} 0 & 1 \\\\ 0 & 0 \\end{bmatrix}
    $$
    """

    pass
