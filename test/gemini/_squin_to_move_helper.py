"""Test-only replacement for the removed bloqade.lanes.upstream.squin_to_move,
composing the canonical transform stage classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kirin import ir, passes
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform import (
    PlaceToMove,
    NativeToPlace,
    LogicalNativeToPlace,
)

if TYPE_CHECKING:
    from bloqade.lanes.analysis import layout, placement
    from bloqade.lanes.arch.spec import ArchSpec


def squin_to_move(
    mt: ir.Method,
    *,
    layout_heuristic: layout.LayoutHeuristicABC,
    placement_strategy: placement.PlacementStrategyABC,
    no_raise: bool = True,
    logical_initialize: bool = True,
    place_opt_type: type[passes.Pass] = SequentialPlacePass,
) -> ir.Method:
    """Compose the canonical transform stage classes into a squin→move pipeline.

    Replacement for the removed ``bloqade.lanes.upstream.squin_to_move`` for
    use in tests and demos.  Behavior:

    * ``logical_initialize=True``  → ``LogicalNativeToPlace`` (default; logical path)
    * ``logical_initialize=False`` → generic ``NativeToPlace`` (physical/neutral path)

    The ``arch_spec`` is extracted from ``layout_heuristic`` (if present) and
    forwarded to the native stage to enable post-unroll address/duplicate
    validation, matching the legacy ``upstream.squin_to_move`` behavior.

    ``SequentialPlacePass`` (or ``place_opt_type`` when overridden) runs between
    the native→place stage and ``PlaceToMove``.
    """
    arch_spec: ArchSpec | None = getattr(layout_heuristic, "arch_spec", None)
    if logical_initialize:
        native_stage: LogicalNativeToPlace | NativeToPlace = LogicalNativeToPlace(
            arch_spec=arch_spec
        )
    else:
        native_stage = NativeToPlace(arch_spec=arch_spec)
    place = native_stage.emit(mt, no_raise=no_raise)
    place_opt_type(place.dialects, no_raise=no_raise)(place)
    return PlaceToMove(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_initialize=logical_initialize,
    ).emit(place, no_raise=no_raise)
