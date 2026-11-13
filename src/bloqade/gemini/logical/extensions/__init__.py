"""Experimental logical-level extensions.

These wrappers are usable from a `@gemini.logical.kernel`, but unlike the gates
in the main `bloqade.gemini.logical` namespace they are not logical operations:
they address the physical qubits underneath a logical qubit and take the state
out of the logical subspace. Using one is only meaningful alongside a
post-selection scheme (e.g. Steane error-correction checks) that discards the
shots where the state left the code space.
"""

from bloqade.gemini.logical.extensions import broadcast as broadcast
from bloqade.gemini.logical.extensions.star import star_rz as star_rz

__all__ = ["broadcast", "star_rz"]
