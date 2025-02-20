from kirin import ir, types, interp
from kirin.decl import info, statement
from kirin.dialects import ilist
from bloqade.qasm2.types import QRegType
from bloqade.analysis.schedule import DagScheduleAnalysis

dialect = ir.Dialect("qasm2.glob")


@statement(dialect=dialect)
class UGate(ir.Statement):
    name = "ugate"
    traits = frozenset({ir.FromPythonCall()})
    registers: ir.SSAValue = info.argument(ilist.IListType[QRegType])
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)


@dialect.register(key="qasm2.schedule.dag")
class Glob(interp.MethodTable):
    @interp.impl(UGate)
    def ugate(self, interp: DagScheduleAnalysis, frame: interp.Frame, stmt: UGate):
        interp.update_dag(stmt, [stmt.registers])
        return ()
