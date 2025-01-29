from kirin.emit import EmitStrFrame
from kirin.interp import MethodTable, impl
from bloqade.stim.emit.stim import EmitStimMain

from . import stmts
from ._dialect import dialect


@dialect.register(key="emit.stim")
class EmitStimNoiseMethods(MethodTable):

    @impl(stmts.Depolarize1)
    def depolarize1(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: stmts.Depolarize1
    ):

        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        p: str = frame.get(stmt.p)
        res = f"DEPOLARIZE1({p}) " + " ".join(targets)
        emit.writeln(frame, res)

        return ()

    @impl(stmts.Depolarize2)
    def depolarize2(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: stmts.Depolarize2
    ):

        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        p: str = frame.get(stmt.p)
        res = f"DEPOLARIZE2({p}) " + " ".join(targets)
        emit.writeln(frame, res)

        return ()

    @impl(stmts.PauliChannel1)
    def pauli_channel1(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: stmts.PauliChannel1
    ):

        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        px: str = frame.get(stmt.px)
        py: str = frame.get(stmt.py)
        pz: str = frame.get(stmt.pz)
        res = f"PAULI_CHANNEL_1({px},{py},{pz}) " + " ".join(targets)
        emit.writeln(frame, res)

        return ()

    @impl(stmts.PauliChannel2)
    def pauli_channel2(
        self, emit: EmitStimMain, frame: EmitStrFrame, stmt: stmts.PauliChannel2
    ):

        targets: tuple[str, ...] = frame.get_values(stmt.targets)
        prob: tuple[str, ...] = frame.get_values(stmt.args)[
            :15
        ]  # extract the first 15 argument, which is the probabilities
        prob_str: str = ", ".join(prob)

        res = f"PAULI_CHANNEL_2({prob_str}) " + " ".join(targets)
        emit.writeln(frame, res)

        return ()
