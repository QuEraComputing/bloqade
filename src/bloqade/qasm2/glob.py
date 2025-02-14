"""QASM2 extension for global gates."""

from typing import Any

from kirin.dialects import ilist
from kirin.lowering import wraps

from .types import QReg
from .dialects import glob


@wraps(glob.UGate)
def u(
    theta: float, phi: float, lam: float, registers: ilist.IList[QReg, Any] | list
) -> None:
    """Apply a U gate to all qubits in the input registers.

    Args:
        theta (float): The angle theta.
        phi (float): The angle phi.
        lam (float): The angle lam.
        registers (IList[QReg] | list[QReg]): The registers to apply the gate to.

    """
