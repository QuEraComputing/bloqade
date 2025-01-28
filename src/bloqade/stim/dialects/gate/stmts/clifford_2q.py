from kirin.decl import statement

from .base import TwoQubitGate
from .._dialect import dialect


# Two Qubit Clifford Gates
# ---------------------------------------
@statement(dialect=dialect)
class Swap(TwoQubitGate):
    name = "SWAP"
