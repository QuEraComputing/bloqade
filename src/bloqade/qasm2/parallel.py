"""QASM2 extension for parallel execution of gates."""

from typing import Any

from kirin.dialects import ilist
from kirin.lowering import wraps

from .types import Qubit
from .dialects import parallel


@wraps(parallel.CZ)
def cz(
    ctrls: ilist.IList[Qubit, Any] | list, qargs: ilist.IList[Qubit, Any] | list
) -> None: ...


@wraps(parallel.UGate)
def u(
    qargs: ilist.IList[Qubit, Any] | list, theta: float, phi: float, lam: float
) -> None: ...


@wraps(parallel.RZ)
def rz(qargs: ilist.IList[Qubit, Any] | list, theta: float) -> None: ...
