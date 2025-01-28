from kirin.decl import statement

from .._dialect import dialect
from .base import ControlledTwoQubitGate


# Two Qubit Clifford Gates
# ---------------------------------------
@statement(dialect=dialect)
class CX(ControlledTwoQubitGate):
    name = "CX"


@statement(dialect=dialect)
class CY(ControlledTwoQubitGate):
    name = "CY"


@statement(dialect=dialect)
class CZ(ControlledTwoQubitGate):
    name = "CZ"
