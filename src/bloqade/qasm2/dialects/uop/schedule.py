from kirin import interp
from kirin.analysis import ForwardFrame
from bloqade.analysis.schedule import GateSchedule, DagScheduleAnalysis

from . import stmts
from ._dialect import dialect


@dialect.register(key="qasm2.schedule.dag")
class UOp(interp.MethodTable):

    @interp.impl(stmts.SingleQubitGate)
    def single_qubit_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame[GateSchedule],
        stmt: stmts.SingleQubitGate,
    ):
        interp.update_dag(stmt, [stmt.qarg])
        return ()

    @interp.impl(stmts.TwoQubitCtrlGate)
    def two_qubit_ctrl_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame[GateSchedule],
        stmt: stmts.TwoQubitCtrlGate,
    ):
        interp.update_dag(stmt, [stmt.ctrl, stmt.qarg])
        return ()

    @interp.impl(stmts.CCX)
    def ccx_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame[GateSchedule],
        stmt: stmts.CCX,
    ):
        interp.update_dag(stmt, [stmt.ctrl1, stmt.ctrl2, stmt.qarg])
        return ()

    @interp.impl(stmts.Barrier)
    def barrier(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame[GateSchedule],
        stmt: stmts.Barrier,
    ):
        interp.update_dag(stmt, stmt.qargs)
        return ()
