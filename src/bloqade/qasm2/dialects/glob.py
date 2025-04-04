from kirin import ir, types, interp
from kirin.decl import info, statement
from kirin.dialects import ilist
from bloqade.qasm2.parse import ast
from bloqade.qasm2.types import QRegType
from bloqade.qasm2.emit.gate import EmitQASM2Gate, EmitQASM2Frame
from bloqade.squin.analysis.schedule import DagScheduleAnalysis

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


@dialect.register(key="emit.qasm2.gate")
class GlobEmit(interp.MethodTable):
    @interp.impl(UGate)
    def ugate(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: UGate):
        registers = [
            emit.assert_node(ast.Name, reg) for reg in frame.get(stmt.registers)
        ]
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        frame.body.append(
            ast.GlobUGate(theta=theta, phi=phi, lam=lam, registers=registers)
        )
        return ()
