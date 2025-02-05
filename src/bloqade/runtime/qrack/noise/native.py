from kirin import interp
from bloqade.noise import native
from bloqade.runtime.qrack import PyQrackInterpreter, reg


@native.dialect.register(key="pyqrack")
class PyQrackMethods(interp.MethodTable):
    def apply_pauli_error(
        self,
        interp: PyQrackInterpreter,
        qarg: reg.SimQubitRef,
        px: float,
        py: float,
        pz: float,
    ):
        p = [1 - (px + py + pz), px, py, pz]

        assert all(0 <= x <= 1 for x in p), "Invalid Pauli error probabilities"

        which = interp.rng_state.choice(["i", "x", "y", "z"], p=p)

        if which == "i":
            return

        getattr(qarg.sim_reg, which)(qarg.addr)

    @interp.impl(native.PauliChannel)
    def single_qubit_error_channel(
        self,
        interp: PyQrackInterpreter,
        frame: interp.Frame,
        stmt: native.PauliChannel,
    ):
        px: float = frame.get(stmt.px)
        py: float = frame.get(stmt.py)
        pz: float = frame.get(stmt.pz)
        qarg: reg.SimQubitRef = frame.get(stmt.qarg)

        if qarg.is_active():
            self.apply_pauli_error(interp, qarg, px, py, pz)

        return ()

    @interp.impl(native.CZPauliChannel)
    def cz_pauli_unpaired(
        self,
        interp: PyQrackInterpreter,
        frame: interp.Frame,
        stmt: native.CZPauliChannel,
    ):
        px_1: float = frame.get(stmt.px_1)
        py_1: float = frame.get(stmt.py_1)
        pz_1: float = frame.get(stmt.pz_1)
        qarg1: reg.SimQubitRef = frame.get(stmt.qarg1)

        px_2: float = frame.get(stmt.px_2)
        py_2: float = frame.get(stmt.py_2)
        pz_2: float = frame.get(stmt.pz_2)
        qarg2: reg.SimQubitRef = frame.get(stmt.qarg2)

        is_active_1 = qarg1.is_active()
        is_active_2 = qarg2.is_active()
        is_both_active = is_active_1 and is_active_2

        if stmt.paired and is_both_active:
            self.apply_pauli_error(interp, qarg1, px_1, py_1, pz_1)
            self.apply_pauli_error(interp, qarg2, px_2, py_2, pz_2)
        elif not stmt.paired:
            if is_both_active:
                return ()
            elif is_active_1:
                self.apply_pauli_error(interp, qarg1, px_1, py_1, pz_1)
            elif is_active_2:
                self.apply_pauli_error(interp, qarg2, px_2, py_2, pz_2)

        return ()

    @interp.impl(native.AtomLossChannel)
    def atom_loss_channel(
        self,
        interp: PyQrackInterpreter,
        frame: interp.Frame,
        stmt: native.AtomLossChannel,
    ):
        prob: float = frame.get(stmt.prob)
        qarg: reg.SimQubitRef = frame.get(stmt.qarg)

        if qarg.is_active() and interp.rng_state.uniform() > prob:
            qarg.drop()

        return ()
