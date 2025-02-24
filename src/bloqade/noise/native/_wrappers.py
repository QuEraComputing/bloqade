from typing import Any

from bloqade.noise import native
from kirin.dialects import ilist
from kirin.lowering import wraps
from bloqade.qasm2.types import Qubit


@wraps(native.AtomLossChannel)
def atom_loss_channel(
    qargs: ilist.IList[Qubit, Any] | list, *, prob: float
) -> None: ...


@wraps(native.PauliChannel)
def pauli_channel(
    qargs: ilist.IList[Qubit, Any] | list, *, px: float, py: float, pz: float
) -> None: ...


@wraps(native.CZPauliChannel)
def cz_pauli_channel(
    ctrls: ilist.IList[Qubit, Any] | list,
    qarg2: ilist.IList[Qubit, Any] | list,
    *,
    px_ctrl: float,
    py_ctrl: float,
    pz_ctrl: float,
    px_qarg: float,
    py_qarg: float,
    pz_qarg: float,
    paired: bool,
) -> None: ...
