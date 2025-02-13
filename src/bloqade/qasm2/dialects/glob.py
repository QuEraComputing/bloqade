from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from bloqade.qasm2.types import QRegType

dialect = ir.Dialect("qasm2.glob")


@statement(dialect=dialect)
class UGate(ir.Statement):
    name = "ugate"
    traits = frozenset({ir.FromPythonCall()})
    registers: ir.SSAValue = info.argument(ilist.IListType[QRegType])
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)
