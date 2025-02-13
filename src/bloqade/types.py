"""Bloqade types.

This module defines the basic types used in Bloqade eDSLs.
"""

from abc import ABC

from kirin import types


class Qubit(ABC):
    """Runtime representation of a qubit.

    Note:
        This is the base class of more specific qubit types, such as
        a reference to a piece of quantum register in some quantum register
        dialects.
    """

    pass


QubitType = types.PyClass(Qubit)
"""Kirin type for a qubit."""
