from typing import Any

from kirin.dialects import ilist
from bloqade.lanes.dialects.steane713 import extensions as extensions
from bloqade.squin.stdlib.broadcast.gate import _radian_to_turn
from bloqade.lanes.dialects.steane713.star import DEFAULT_STEANE_STAR_SUPPORT

from bloqade import types
from bloqade.gemini.logical.group import kernel


@kernel(aggressive_unroll=True, verify=False)
def star_rz(
    theta: float,
    qubits: ilist.IList[types.Qubit, Any],
) -> None:
    extensions.star_rz(_radian_to_turn(theta), qubits, DEFAULT_STEANE_STAR_SUPPORT)
