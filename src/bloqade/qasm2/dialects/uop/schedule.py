from kirin import interp
from kirin.analysis import ForwardFrame
from bloqade.squin.analysis.schedule import DagScheduleAnalysis

from . import stmts
from ._dialect import dialect


@dialect.register(key="qasm2.schedule.dag")
class UOp(interp.MethodTable):

    @interp.impl(stmts.Id)
    @interp.impl(stmts.SXdag)
    @interp.impl(stmts.SX)
    @interp.impl(stmts.X)
    @interp.impl(stmts.Y)
    @interp.impl(stmts.Z)
    @interp.impl(stmts.H)
    @interp.impl(stmts.S)
    @interp.impl(stmts.Sdag)
    @interp.impl(stmts.T)
    @interp.impl(stmts.Tdag)
    @interp.impl(stmts.RX)
    @interp.impl(stmts.RY)
    @interp.impl(stmts.RZ)
    @interp.impl(stmts.U1)
    @interp.impl(stmts.U2)
    @interp.impl(stmts.UGate)
    def single_qubit_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: stmts.SingleQubitGate,
    ):
        interp.update_dag(stmt, [stmt.qarg])
        return ()

    @interp.impl(stmts.RXX)
    @interp.impl(stmts.RZZ)
    @interp.impl(stmts.CX)
    @interp.impl(stmts.CY)
    @interp.impl(stmts.CZ)
    @interp.impl(stmts.CH)
    @interp.impl(stmts.CRZ)
    @interp.impl(stmts.CRY)
    @interp.impl(stmts.CRX)
    @interp.impl(stmts.CU1)
    @interp.impl(stmts.CU3)
    @interp.impl(stmts.CU)
    @interp.impl(stmts.CSX)
    def two_qubit_ctrl_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: stmts.TwoQubitCtrlGate,
    ):
        interp.update_dag(stmt, [stmt.ctrl, stmt.qarg])
        return ()

    @interp.impl(stmts.CCX)
    def ccx_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: stmts.CCX,
    ):
        interp.update_dag(stmt, [stmt.ctrl1, stmt.ctrl2, stmt.qarg])
        return ()

    @interp.impl(stmts.CSwap)
    def cswap_gate(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: stmts.CSwap,
    ):
        interp.update_dag(stmt, [stmt.ctrl, stmt.qarg1, stmt.qarg2])
        return ()

    @interp.impl(stmts.Barrier)
    def barrier(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: stmts.Barrier,
    ):
        interp.update_dag(stmt, stmt.qargs)
        return ()
