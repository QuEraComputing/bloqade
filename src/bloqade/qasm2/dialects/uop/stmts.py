from kirin import ir
from kirin.decl import info, statement
from bloqade.qasm2.types import QubitType
from bloqade.qasm2.dialects.expr.stmts import PyNum

from ._dialect import dialect


# trait
@statement
class SingleQubitGate(ir.Statement):
    """Base class for single qubit gates."""

    traits = frozenset({ir.FromPythonCall()})
    qarg: ir.SSAValue = info.argument(QubitType)
    """qarg (Qubit): The qubit argument."""


@statement
class TwoQubitCtrlGate(ir.Statement):
    traits = frozenset({ir.FromPythonCall()})
    ctrl: ir.SSAValue = info.argument(QubitType)
    """ctrl (Qubit): The control qubit."""
    qarg: ir.SSAValue = info.argument(QubitType)
    """qarg (Qubit): The target qubit."""


@statement(dialect=dialect)
class CX(TwoQubitCtrlGate):
    """Alias for the CNOT or CH gate operations."""

    name = "CX"  # Note this is capitalized


@statement(dialect=dialect)
class UGate(SingleQubitGate):
    """Apply A general single qubit unitary gate."""

    name = "U"
    theta: ir.SSAValue = info.argument(PyNum)
    """theta (float): The theta parameter."""
    phi: ir.SSAValue = info.argument(PyNum)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class Barrier(ir.Statement):
    """Apply the Barrier statement."""

    name = "barrier"
    traits = frozenset({ir.FromPythonCall()})
    qargs: tuple[ir.SSAValue, ...] = info.argument(QubitType)
    """qargs: tuple of qubits to apply the barrier to."""


# qelib1.inc as statements


@statement(dialect=dialect)
class Id(SingleQubitGate):
    """Apply the Identity gate."""

    name = "id"


@statement(dialect=dialect)
class H(SingleQubitGate):
    """Apply the Hadamard gate."""

    name = "h"


@statement(dialect=dialect)
class X(SingleQubitGate):
    """Apply the X gate."""

    name = "x"


@statement(dialect=dialect)
class Y(SingleQubitGate):
    """Apply the Y gate."""

    name = "y"


@statement(dialect=dialect)
class Z(SingleQubitGate):
    """Apply the Z gate."""

    name = "z"


@statement(dialect=dialect)
class S(SingleQubitGate):
    """Apply the S gate."""

    name = "s"


@statement(dialect=dialect)
class Sdag(SingleQubitGate):
    """Apply the hermitian conj of S gate."""

    name = "sdg"


@statement(dialect=dialect)
class SX(SingleQubitGate):
    """Apply the quantum Sqrt(X) gate."""

    name = "sx"


@statement(dialect=dialect)
class SXdag(SingleQubitGate):
    """Apply the dagger of quantum Sqrt(X) gate."""

    name = "sxdg"


@statement(dialect=dialect)
class T(SingleQubitGate):
    """Apply the T gate."""

    name = "t"


@statement(dialect=dialect)
class Tdag(SingleQubitGate):
    """Apply the hermitian conj of T gate."""

    name = "tdg"


@statement(dialect=dialect)
class RX(SingleQubitGate):
    """Apply the RX gate."""

    name = "rx"
    theta: ir.SSAValue = info.argument(PyNum)
    """theta (float): The angle of rotation around x axis."""


@statement(dialect=dialect)
class RY(SingleQubitGate):
    """Apply the RY gate."""

    name = "ry"
    theta: ir.SSAValue = info.argument(PyNum)
    """theta (float): The angle of rotation around y axis."""


@statement(dialect=dialect)
class RZ(SingleQubitGate):
    """Apply the RZ gate."""

    name = "rz"
    theta: ir.SSAValue = info.argument(PyNum)
    """theta (float): the angle of rotation around Z axis."""


@statement(dialect=dialect)
class U1(SingleQubitGate):
    """Apply the U1 gate."""

    name = "u1"
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class U2(SingleQubitGate):
    """Apply the U2 gate."""

    name = "u2"
    phi: ir.SSAValue = info.argument(PyNum)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class CZ(TwoQubitCtrlGate):
    """Apply the Controlled-Z gate."""

    name = "cz"


@statement(dialect=dialect)
class CY(TwoQubitCtrlGate):
    """Apply the Controlled-Y gate."""

    name = "cy"


@statement(dialect=dialect)
class CSX(TwoQubitCtrlGate):
    """Apply the Controlled-Sqrt(X) gate."""

    name = "csx"


@statement(dialect=dialect)
class Swap(TwoQubitCtrlGate):
    """Apply the Swap gate."""

    name = "swap"


@statement(dialect=dialect)
class CH(TwoQubitCtrlGate):
    """Apply the Controlled-H gate."""

    name = "ch"


@statement(dialect=dialect)
class CCX(ir.Statement):
    """Apply the doubly controlled X gate."""

    name = "ccx"
    traits = frozenset({ir.FromPythonCall()})
    ctrl1: ir.SSAValue = info.argument(QubitType)
    """ctrl1 (Qubit): The first control qubit."""
    ctrl2: ir.SSAValue = info.argument(QubitType)
    """ctrl2 (Qubit): The second control qubit."""
    qarg: ir.SSAValue = info.argument(QubitType)
    """qarg (Qubit): The target qubit."""


@statement(dialect=dialect)
class CSwap(ir.Statement):
    """Apply the controlled swap gate."""

    name = "ccx"
    traits = frozenset({ir.FromPythonCall()})
    ctrl: ir.SSAValue = info.argument(QubitType)
    """ctrl (Qubit): The control qubit."""
    qarg1: ir.SSAValue = info.argument(QubitType)
    """qarg1 (Qubit): The first target qubit."""
    qarg2: ir.SSAValue = info.argument(QubitType)
    """qarg2 (Qubit): The second target qubit."""


@statement(dialect=dialect)
class CRX(TwoQubitCtrlGate):
    """Apply the Controlled-RX gate."""

    name = "crx"
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The angle to rotate around the X axis."""


@statement(dialect=dialect)
class CRY(TwoQubitCtrlGate):
    """Apply the Controlled-RY gate."""

    name = "cry"
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The angle to rotate around the Y axis."""


@statement(dialect=dialect)
class CRZ(TwoQubitCtrlGate):
    """Apply the Controlled-RZ gate."""

    name = "crz"
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The angle to rotate around the Z axis."""


@statement(dialect=dialect)
class CU1(TwoQubitCtrlGate):
    """Apply the Controlled-U1 gate."""

    name = "cu1"
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class CU3(TwoQubitCtrlGate):
    """Apply the Controlled-U3 gate."""

    name = "cu3"
    theta: ir.SSAValue = info.argument(PyNum)
    phi: ir.SSAValue = info.argument(PyNum)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class CU(TwoQubitCtrlGate):
    """Apply the Controlled-U gate."""

    name = "cu"
    theta: ir.SSAValue = info.argument(PyNum)
    phi: ir.SSAValue = info.argument(PyNum)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(PyNum)
    """lam (float): The lambda parameter."""
    gamma: ir.SSAValue = info.argument(PyNum)


@statement(dialect=dialect)
class RXX(TwoQubitCtrlGate):
    """Apply the XX rotation gate."""

    name = "rxx"
    theta: ir.SSAValue = info.argument(PyNum)
    """theta (float): The angle of rotation around the X axis."""


@statement(dialect=dialect)
class RZZ(TwoQubitCtrlGate):
    """Apply the ZZ rotation gate."""

    name = "rzz"
    theta: ir.SSAValue = info.argument(PyNum)
    """theta (float): The angle of rotation around the Z axis."""
