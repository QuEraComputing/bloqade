from kirin.emit import EmitStrFrame
from kirin.interp import MethodTable, impl
from bloqade.stim.emit.stim import EmitStimMain

from . import stmts
from ._dialect import dialect
from .stmts.base import SingleQubitGate, ControlledTwoQubitGate


@dialect.register(key="emit.stim")
class EmitStimGateMethods(MethodTable):

    gate_1q_map: dict[str, tuple[str, str]] = {
        stmts.X.name: ("X", "X"),
        stmts.Y.name: ("Y", "Y"),
        stmts.Z.name: ("Z", "Z"),
        stmts.H.name: ("H", "H"),
        stmts.S.name: ("S", "S_DAG"),
        stmts.SqrtX.name: ("SQRT_X", "SQRT_X_DAG"),
        stmts.SqrtY.name: ("SQRT_Y", "SQRT_Y_DAG"),
        stmts.SqrtZ.name: ("SQRT_Z", "SQRT_Z_DAG"),
    }

    @impl(stmts.X)
    @impl(stmts.Y)
    @impl(stmts.Z)
    @impl(stmts.S)
    @impl(stmts.H)
    @impl(stmts.SqrtX)
    @impl(stmts.SqrtY)
    @impl(stmts.SqrtZ)
    def single_qubit_gate(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: SingleQubitGate
    ):
        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        res = f"{self.gate_1q_map[stmt.name][int(stmt.dagger)]} " + " ".join(targets)
        emit.writeln(frame, res)

        return ()

    gate_2q_map: dict[str, tuple[str, str]] = {
        stmts.Swap.name: ("SWAP", "SWAP"),
    }


    @impl(stmts.Swap)
    def two_qubit_gate(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: ControlledTwoQubitGate
    ):
        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        res = f"{self.gate_ctrl_2q_map[stmt.name][int(stmt.dagger)]} " + " ".join(targets)
        emit.writeln(frame, res)

        return ()

    gate_ctrl_2q_map: dict[str, tuple[str, str]] = {
        stmts.CX.name: ("CX", "CX"),
        stmts.CY.name: ("CY", "CY"),
        stmts.CZ.name: ("CZ", "CZ"),
        stmts.Swap.name: ("SWAP", "SWAP"),
    }

    @impl(stmts.CX)
    @impl(stmts.CY)
    @impl(stmts.CZ)
    def ctrl_two_qubit_gate(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: ControlledTwoQubitGate
    ):
        controls: tuple[str, ...] = frame.get_values(stmt.controls)
        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        res = f"{self.gate_ctrl_2q_map[stmt.name][int(stmt.dagger)]} " + " ".join(
            f"{ctrl} {tgt}" for ctrl, tgt in zip(controls, targets)
        )
        emit.writeln(frame, res)

        return ()

    @impl(stmts.SPP)
    def spp(self, emit: EmitStimMain, frame: EmitStrFrame, stmt: stmts.SPP):

        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        res = "SPP " + " ".join(targets)
        emit.writeln(frame, res)

        return ()
