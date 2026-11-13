"""Qubit arrangement helpers for Gemini kernels.

Use ``bloqade.gemini.move_to`` for explicit destinations and
``bloqade.gemini.arrange.permute`` for permutations of qubit references.
Compiler statements and the dialect object belong to ``bloqade.lanes``.
"""

from bloqade.lanes.dialects.arrange import permute as permute

__all__ = ["permute"]
