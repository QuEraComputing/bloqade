from kirin import ir, types
from kirin.decl import info, statement
from bloqade.qasm2.types import BitType, CRegType, QRegType, QubitType

from ._dialect import dialect


@statement(dialect=dialect)
class QRegNew(ir.Statement):
    """Create a new quantum register."""

    name = "qreg.new"
    traits = frozenset({ir.FromPythonCall()})
    n_qubits: ir.SSAValue = info.argument(types.Int)
    """n_qubits: The number of qubits in the register."""
    result: ir.ResultValue = info.result(QRegType)
    """A new quantum register with n_qubits set to |0>."""


@statement(dialect=dialect)
class CRegNew(ir.Statement):
    """Create a new classical register."""

    name = "creg.new"
    traits = frozenset({ir.FromPythonCall()})
    n_bits: ir.SSAValue = info.argument(types.Int)
    """n_bits (Int): The number of bits in the register."""
    result: ir.ResultValue = info.result(CRegType)
    """result (CReg): The new classical register with all bits set to 0."""


@statement(dialect=dialect)
class Reset(ir.Statement):
    """Reset a qubit to the |0> state."""

    name = "reset"
    traits = frozenset({ir.FromPythonCall()})
    qarg: ir.SSAValue = info.argument(QubitType)
    """qarg (Qubit): The qubit to reset."""


@statement(dialect=dialect)
class CRegEq(ir.Statement):
    """Check if two classical registers are equal."""

    name = "eq"
    traits = frozenset({ir.Pure(), ir.FromPythonCall()})
    lhs: ir.SSAValue = info.argument(types.Int | CRegType | BitType)
    """lhs (CReg): The first register."""
    rhs: ir.SSAValue = info.argument(types.Int | CRegType | BitType)
    """rhs (CReg): The second register."""
    result: ir.ResultValue = info.result(types.Bool)
    """result (bool): True if the registers are equal, False otherwise."""


@statement(dialect=dialect)
class QRegGet(ir.Statement):
    """Get a qubit from a quantum register."""

    name = "qreg.get"
    traits = frozenset({ir.FromPythonCall()})
    reg: ir.SSAValue = info.argument(QRegType)
    """reg (QReg): The quantum register."""
    idx: ir.SSAValue = info.argument(types.Int)
    """idx (Int): The index of the qubit in the register."""
    result: ir.ResultValue = info.result(QubitType)
    """result (Qubit): The qubit at position `idx`."""


@statement(dialect=dialect)
class CRegGet(ir.Statement):
    """Get a bit from a classical register."""

    name = "creg.get"
    traits = frozenset({ir.FromPythonCall()})
    reg: ir.SSAValue = info.argument(CRegType)
    """reg (CReg): The classical register."""
    idx: ir.SSAValue = info.argument(types.Int)
    """idx (Int): The index of the bit in the register."""
    result: ir.ResultValue = info.result(BitType)
    """result (Bit): The bit at position `idx`."""
