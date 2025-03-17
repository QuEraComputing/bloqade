"""qubit dialect for squin language.

This dialect defines the operations that can be performed on qubits.

Depends on:
- `bloqade.squin.op`: provides the `OpType` type and semantics for operators applied to qubits.
- `kirin.dialects.ilist`: provides the `ilist.IListType` type for lists of qubits.
"""

from typing import Any

from kirin import ir, types
from kirin.decl import info, statement
from bloqade.types import Qubit, QubitType
from kirin.dialects import ilist
from kirin.lowering import wraps
from bloqade.squin.op.types import Op, OpType

dialect = ir.Dialect("squin.qubit")


@statement(dialect=dialect)
class New(ir.Statement):
    traits = frozenset({ir.FromPythonCall()})
    n_qubits: ir.SSAValue = info.argument(types.Int)
    result: ir.ResultValue = info.result(ilist.IListType[QubitType, types.Any])

    def __init__(self, n_qubits: int):
        super().__init__(
            attributes={"n_qubits": ir.PyAttr(n_qubits)},
            # NOTE: we don't have dependent types in Python, this might be removed
            #     in the future, if we end up supporting dependent types in Kirin.
            result_types=(ilist.IListType[QubitType, types.Literal(n_qubits)],),
        )


@statement(dialect=dialect)
class Apply(ir.Statement):
    operator: ir.SSAValue = info.argument(OpType)
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType])


@statement(dialect=dialect)
class Measure(ir.Statement):
    traits = frozenset({ir.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType])
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class MeasureAndReset(ir.Statement):
    traits = frozenset({ir.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType])
    result: ir.ResultValue = info.result(types.Int)


@statement(dialect=dialect)
class Reset(ir.Statement):
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType])


# NOTE: no dependent types in Python, so we have to mark it Any...
@wraps(New)
def new(n_qubits: int) -> ilist.IList[Qubit, Any]:
    """Create a new list of qubits.

    Args:
        n_qubits(int): The number of qubits to create.

    Returns:
        (ilist.IList[Qubit, n_qubits]) A list of qubits.
    """
    ...


@wraps(Apply)
def apply(operator: Op, qubits: ilist.IList[Qubit, Any]) -> None:
    """Apply an operator to a list of qubits.

    Args:
        operator: The operator to apply.
        qubits: The list of qubits to apply the operator to. The size of the list
            must be inferable and match the number of qubits expected by the operator.

    Returns:
        None
    """
    ...


@wraps(Measure)
def measure(qubits: ilist.IList[Qubit, Any]) -> int:
    """Measure the qubits in the list."

    Args:
        qubits: The list of qubits to measure.

    Returns:
        int: The result of the measurement.
    """
    ...


@wraps(MeasureAndReset)
def measure_and_reset(qubits: ilist.IList[Qubit, Any]) -> int:
    """Measure the qubits in the list and reset them."

    Args:
        qubits: The list of qubits to measure and reset.

    Returns:
        int: The result of the measurement.
    """
    ...


@wraps(Reset)
def reset(qubits: ilist.IList[Qubit, Any]) -> None:
    """Reset the qubits in the list."

    Args:
        qubits: The list of qubits to reset.
    """
    ...
