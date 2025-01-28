from kirin.decl import statement

from .base import ControlledTwoQubitGate
from .._dialect import dialect


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
