"""QASM2 extension for global gates."""

from typing import Any

from kirin.dialects import ilist
from kirin.lowering import wraps

from .types import QReg
from .dialects import glob


@wraps(glob.UGate)
def u(
    theta: float, phi: float, lam: float, registers: ilist.IList[QReg, Any] | list
) -> None: ...
