from kirin import ir, types
from kirin.decl import info, statement
from bloqade.types import QubitType

from ._dialect import dialect


# trait
@statement(dialect=dialect)
class SingleQubitGate(ir.Statement):
    """Base class for single qubit gates."""

    name = "1q"
    traits = frozenset({ir.FromPythonCall()})
    qarg: ir.SSAValue = info.argument(QubitType)
    """qarg (Qubit): The qubit argument."""


@statement(dialect=dialect)
class TwoQubitCtrlGate(ir.Statement):
    name = "2q"
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
    theta: ir.SSAValue = info.argument(types.Float)
    """theta (float): The theta parameter."""
    phi: ir.SSAValue = info.argument(types.Float)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(types.Float)
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

    name = "sdag"


@statement(dialect=dialect)
class T(SingleQubitGate):
    """Apply the T gate."""

    name = "t"


@statement(dialect=dialect)
class Tdag(SingleQubitGate):
    """Apply the hermitian conj of T gate."""

    name = "tdag"


@statement(dialect=dialect)
class RX(SingleQubitGate):
    """Apply the RX gate."""

    name = "rx"
    theta: ir.SSAValue = info.argument(types.Float)
    """theta (float): The angle of rotation around x axis."""


@statement(dialect=dialect)
class RY(SingleQubitGate):
    """Apply the RY gate."""

    name = "ry"
    theta: ir.SSAValue = info.argument(types.Float)
    """theta (float): The angle of rotation around y axis."""


@statement(dialect=dialect)
class RZ(SingleQubitGate):
    """Apply the RZ gate."""

    name = "rz"
    theta: ir.SSAValue = info.argument(types.Float)
    """theta (float): the angle of rotation around Z axis."""


@statement(dialect=dialect)
class U1(SingleQubitGate):
    """Apply the U1 gate."""

    name = "u1"
    lam: ir.SSAValue = info.argument(types.Float)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class U2(SingleQubitGate):
    """Apply the U2 gate."""

    name = "u2"
    phi: ir.SSAValue = info.argument(types.Float)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(types.Float)
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
class CRX(TwoQubitCtrlGate):
    """Apply the Controlled-RX gate."""

    name = "crx"
    theta: ir.SSAValue = info.argument(types.Float)
    """theta (float): The angle to rotate around the X axis."""


@statement(dialect=dialect)
class CU1(TwoQubitCtrlGate):
    """Apply the Controlled-U1 gate."""

    name = "cu1"
    lam: ir.SSAValue = info.argument(types.Float)
    """lam (float): The lambda parameter."""


@statement(dialect=dialect)
class CU3(TwoQubitCtrlGate):
    """Apply the Controlled-U3 gate."""

    name = "cu3"
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    """phi (float): The phi parameter."""
    lam: ir.SSAValue = info.argument(types.Float)
    """lam (float): The lambda parameter."""
