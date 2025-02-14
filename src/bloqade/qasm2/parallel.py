"""QASM2 extension for parallel execution of gates."""

from typing import Any

from kirin.dialects import ilist
from kirin.lowering import wraps

from .types import Qubit
from .dialects import parallel


@wraps(parallel.CZ)
def cz(
    ctrls: ilist.IList[Qubit, Any] | list, qargs: ilist.IList[Qubit, Any] | list
) -> None:
    """Apply a controlled-Z gate to input qubits in parallel.

    Args:
        ctrls (IList[Qubit] | list[Qubit]): The control qubits.
        qargs (IList[Qubit] | list[Qubit]): The target qubits.

    """


@wraps(parallel.UGate)
def u(
    qargs: ilist.IList[Qubit, Any] | list, theta: float, phi: float, lam: float
) -> None:
    """Apply a U gate to input qubits in parallel.

    Args:
        qargs (IList[Qubit] | list[Qubit]): The target qubits.
        theta (float): The angle theta.
        phi (float): The angle phi.
        lam (float): The angle lam.

    """


@wraps(parallel.RZ)
def rz(qargs: ilist.IList[Qubit, Any] | list, theta: float) -> None:
    """Apply a RZ gate to input qubits in parallel.

    Args:
        qargs (IList[Qubit] | list[Qubit]): The target qubits.
        theta (float): The angle theta.

    """
