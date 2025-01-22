"""Kirin types for the QASM2 dialect."""

from kirin.ir import types


class Qubit:
    """Runtime representation of a qubit."""

    pass


class Bit:
    """Runtime representation of a bit."""

    pass


class QReg:
    """Runtime representation of a quantum register."""

    pass


class CReg:
    """Runtime representation of a classical register."""

    pass


QubitType = types.PyClass(Qubit)
"""Kirin type for a qubit."""

BitType = types.PyClass(Bit)
"""Kirin type for a classical bit."""

QRegType = types.PyClass(QReg)
"""Kirin type for a quantum register."""

CRegType = types.PyClass(CReg)
"""Kirin type for a classical register."""
