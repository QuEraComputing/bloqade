from kirin.decl import statement

from .base import SingleQubitGate
from .._dialect import dialect


# Pauli Gates
# -----------------------------------
@statement(dialect=dialect)
class Identity(SingleQubitGate):
    name = "I"


@statement(dialect=dialect)
class X(SingleQubitGate):
    name = "X"


@statement(dialect=dialect)
class Y(SingleQubitGate):
    name = "Y"


@statement(dialect=dialect)
class Z(SingleQubitGate):
    name = "Z"


# Single Qubit Clifford Gates
# ---------------------------------------
@statement(dialect=dialect)
class H(SingleQubitGate):
    name = "H"


@statement(dialect=dialect)
class S(SingleQubitGate):
    name = "S"


@statement(dialect=dialect)
class SqrtX(SingleQubitGate):
    name = "SQRT_X"


@statement(dialect=dialect)
class SqrtY(SingleQubitGate):
    name = "SQRT_Y"


@statement(dialect=dialect)
class SqrtZ(SingleQubitGate):
    name = "SQRT_Z"
