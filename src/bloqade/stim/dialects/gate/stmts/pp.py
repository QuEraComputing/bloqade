from kirin import ir
from kirin.decl import statement, info
from .._dialect import dialect
from ...aux.types import PauliStringType

# Generalized Pauli-product gates
# ---------------------------------------
@statement(dialect=dialect)
class SPP(ir.Statement):
    name = "SPP"
    traits = frozenset({ir.FromPythonCall()})
    dagger: bool = info.attribute(ir.types.Bool, property=True)    
    targets: tuple[ir.SSAValue, ...] = info.argument(PauliStringType)