from kirin import ir
from kirin.decl import info, statement

dialect = ir.Dialect("qasm2.glob")


@statement(dialect=dialect)
class UGate(ir.Statement):
    name = "ugate"
    traits = frozenset({ir.FromPythonCall()})
    theta: ir.SSAValue = info.argument(ir.types.Float)
    phi: ir.SSAValue = info.argument(ir.types.Float)
    lam: ir.SSAValue = info.argument(ir.types.Float)
