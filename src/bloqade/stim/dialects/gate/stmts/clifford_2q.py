from kirin.decl import statement

from .._dialect import dialect
from .base import TwoQubitGate


# Two Qubit Clifford Gates
# ---------------------------------------
@statement(dialect=dialect)
class Swap(TwoQubitGate):
    name = "SWAP"
