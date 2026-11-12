from typing import TYPE_CHECKING, TypeVar, Optional, TypeAlias

from kirin import types
from bloqade.types import Qubit
from kirin.dialects import ilist
from bloqade.lanes.dialects.arch import new_at

from bloqade import qubit
from bloqade.gemini.logical import kernel

N = TypeVar("N")

if TYPE_CHECKING:
    PositionIndex: TypeAlias = Optional[int]  # noqa: UP045
else:
    # Spell the element type as a Kirin type, not a Python union hint. Kirin
    # releases up to 0.22.21 convert a union nested inside a generic hint
    # (``IList[int | None, N]``) to the wrong type, which drops the static
    # ``IList`` length through ``ilist.map``. ``typing.Optional`` avoided that
    # before Python 3.14, where it became the same object as ``int | None``.
    # Revert to ``Optional[int]`` once the minimum Kirin includes
    # QuEraComputing/kirin#818.
    PositionIndex = types.Union(types.Int, types.NoneType)


@kernel(aggressive_unroll=True, verify=False)
def qalloc_at(
    positions: ilist.IList[PositionIndex, N],
) -> ilist.IList[Qubit, N]:
    """Allocate logical qubits at optional linear logical positions.

    An integer ``position`` pins the qubit to zone 0 at word ``2 * position``;
    ``None`` allocates an unpinned qubit for the layout heuristic to place.
    The input must be statically known, and the calling kernel must keep
    ``aggressive_unroll=True`` (the default), so the map can lower into
    individual allocations.
    """

    def position_to_new_at(
        position: PositionIndex,
    ) -> Qubit:
        if position is None:
            q = qubit.new()
        else:
            q = new_at(0, position * 2, 0)
        return q

    return ilist.map(position_to_new_at, positions)
