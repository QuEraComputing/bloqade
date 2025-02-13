from kirin import ir, types
from kirin.decl import info, statement

from .._dialect import dialect
from ...aux.types import PauliStringType


# Generalized Pauli-product gates
# ---------------------------------------
@statement(dialect=dialect)
class SPP(ir.Statement):
    name = "SPP"
    traits = frozenset({ir.FromPythonCall()})
    dagger: bool = info.attribute(types.Bool, default=False)
    targets: tuple[ir.SSAValue, ...] = info.argument(PauliStringType)
