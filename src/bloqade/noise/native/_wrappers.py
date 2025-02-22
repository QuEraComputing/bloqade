from bloqade.noise import native
from kirin.dialects import ilist
from kirin.lowering import wraps
from bloqade.qasm2.types import Qubit


@wraps(native.AtomLossChannel)
def atom_loss_channel(prob: float, qargs: ilist.IList[Qubit] | list) -> None: ...


@wraps(native.PauliChannel)
def pauli_channel(
    px: float, py: float, pz: float, qargs: ilist.IList[Qubit] | list
) -> None: ...


@wraps(native.CZPauliChannel)
def cz_pauli_channel(
    ctrls: ilist.IList[Qubit] | list,
    qarg2: ilist.IList[Qubit] | list,
    *,
    px_1: float,
    py_1: float,
    pz_1: float,
    px_2: float,
    py_2: float,
    pz_2: float,
    paired: bool,
) -> None: ...
