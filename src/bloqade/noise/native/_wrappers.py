from bloqade.noise import native
from kirin.lowering import wraps
from bloqade.qasm2.types import Qubit


@wraps(native.AtomLossChannel)
def atom_loss_channel(prob: float, qarg: Qubit) -> None: ...


@wraps(native.PauliChannel)
def pauli_channel(px: float, py: float, pz: float, qarg: Qubit) -> None: ...


@wraps(native.CZPauliChannel)
def cz_pauli_channel(
    px_1: float,
    py_1: float,
    pz_1: float,
    px_2: float,
    py_2: float,
    pz_2: float,
    qarg1: Qubit,
    qarg2: Qubit,
    *,
    paired: bool,
) -> None: ...
