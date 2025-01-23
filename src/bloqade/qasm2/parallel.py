"""QASM2 extension for parallel execution of gates.
"""

from kirin.lowering import wraps

from .types import Qubit
from .dialects import parallel


@wraps(parallel.CZ)
def CZ(ctrls: tuple[Qubit, ...], qargs: tuple[Qubit, ...]) -> None: ...


@wraps(parallel.UGate)
def UGate(qargs: tuple[Qubit, ...], theta: float, phi: float, lam: float) -> None: ...


@wraps(parallel.RZ)
def RZ(qargs: tuple[Qubit, ...], theta: float) -> None: ...
