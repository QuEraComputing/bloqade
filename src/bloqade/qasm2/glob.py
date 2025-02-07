"""QASM2 extension for global gates."""

from kirin.lowering import wraps

from .dialects import glob


@wraps(glob.UGate)
def u(theta: float, phi: float, lam: float) -> None: ...
