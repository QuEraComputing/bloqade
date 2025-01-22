"""Bloqade types.

This module defines the basic types used in Bloqade eDSLs.
"""

from kirin.ir import types


class Qubit:
    """Runtime representation of a qubit.

    Note:
        This is the base class of more specific qubit types, such as
        a reference to a piece of quantum register in some quantum register
        dialects.
    """

    pass


class Bit:
    """Runtime representation of a bit.

    Note:
        This is the base class of more specific bit types, such as
        a reference to a piece of classical register in some quantum register
        dialects.
    """

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
