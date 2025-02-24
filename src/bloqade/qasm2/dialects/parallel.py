from typing import Any

from kirin import ir, types, interp
from kirin.decl import info, statement
from kirin.analysis import ForwardFrame
from kirin.dialects import ilist
from bloqade.qasm2.parse import ast
from bloqade.qasm2.types import QubitType
from bloqade.qasm2.emit.gate import EmitQASM2Gate, EmitQASM2Frame
from bloqade.analysis.schedule import DagScheduleAnalysis

dialect = ir.Dialect("qasm2.parallel")

N = types.TypeVar("N")


@statement(dialect=dialect)
class CZ(ir.Statement):
    name = "cz"
    traits = frozenset({ir.FromPythonCall()})
    ctrls: ir.SSAValue = info.argument(ilist.IListType[QubitType, N])
    qargs: ir.SSAValue = info.argument(ilist.IListType[QubitType, N])


@statement(dialect=dialect)
class UGate(ir.Statement):
    name = "u"
    traits = frozenset({ir.FromPythonCall()})
    qargs: ir.SSAValue = info.argument(ilist.IListType[QubitType])
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)


@statement(dialect=dialect)
class RZ(ir.Statement):
    name = "rz"
    traits = frozenset({ir.FromPythonCall()})
    qargs: ir.SSAValue = info.argument(ilist.IListType[QubitType])
    theta: ir.SSAValue = info.argument(types.Float)


@dialect.register(key="emit.qasm2.gate")
class Parallel(interp.MethodTable):

    def _emit_parallel_qargs(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, qargs: ir.SSAValue
    ):
        qargs_: ilist.IList[ast.Node, Any] = frame.get(qargs)  # type: ignore
        return [(emit.assert_node((ast.Name, ast.Bit), qarg),) for qarg in qargs_]

    @interp.impl(UGate)
    def ugate(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: UGate):
        qargs = self._emit_parallel_qargs(emit, frame, stmt.qargs)
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        frame.body.append(
            ast.ParaU3Gate(
                theta=theta, phi=phi, lam=lam, qargs=ast.ParallelQArgs(qargs=qargs)
            )
        )
        return ()

    @interp.impl(RZ)
    def rz(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: RZ):
        qargs = self._emit_parallel_qargs(emit, frame, stmt.qargs)
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        frame.body.append(
            ast.ParaRZGate(theta=theta, qargs=ast.ParallelQArgs(qargs=qargs))
        )
        return ()

    @interp.impl(CZ)
    def cz(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: CZ):
        ctrls = self._emit_parallel_qargs(emit, frame, stmt.ctrls)
        qargs = self._emit_parallel_qargs(emit, frame, stmt.qargs)
        frame.body.append(
            ast.ParaCZGate(
                qargs=ast.ParallelQArgs(
                    qargs=[ctrl + qarg for ctrl, qarg in zip(ctrls, qargs)]
                )
            )
        )
        return ()


@dialect.register(key="qasm2.schedule.dag")
class ParallelDag(interp.MethodTable):

    @interp.impl(CZ)
    def parallel_cz(self, interp: DagScheduleAnalysis, frame: ForwardFrame, stmt: CZ):
        interp.update_dag(stmt, [stmt.qargs, stmt.ctrls])
        return ()

    @interp.impl(UGate)
    def parallel_ugate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: UGate,
    ):
        interp.update_dag(stmt, [stmt.qargs])
        return ()

    @interp.impl(RZ)
    def parallel_rz(self, interp: DagScheduleAnalysis, frame: ForwardFrame, stmt: RZ):
        interp.update_dag(stmt, [stmt.qargs])
        return ()
