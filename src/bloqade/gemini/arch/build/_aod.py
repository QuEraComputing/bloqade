"""AOD realizability rules for architecture buses.

An AOD is a crossed pair of deflectors: its traps sit at the Cartesian
product of the x-tones and y-tones, and each tone is swept independently.
A transport is realizable exactly when it is **separable and
order-preserving**:

* every atom's x-displacement depends only on which column it is in, and
  its y-displacement only on its row — an atom cannot leave its own tone;
* tones keep their relative order, since two frequencies cannot pass
  through each other.

This is weaker than a uniform translation: compressing or expanding a
rectangle gives each column a different displacement and is still one AOD
operation.  The path search in :mod:`.._geometry` is the stricter party —
it derives one reference path and applies its deltas to every lane — so a
legal non-uniform bus is accepted here and simply lands unrouted.
"""

from __future__ import annotations

from collections.abc import Sequence, Collection

from bloqade.gemini.arch.build._geometry import NM_PER_UM

Position = tuple[int, int]
Pair = tuple[Position, Position]


def _um(p: Position) -> tuple[float, float]:
    """Render an nm position in µm, for diagnostics."""
    return (p[0] / NM_PER_UM, p[1] / NM_PER_UM)


def check_aod_transport(
    pairs: Sequence[Pair],
    *,
    occupied: Collection[Position],
    label: str,
) -> None:
    """Reject a bus no AOD operation can perform.

    Args:
        pairs: ``(src_position, dst_position)`` in nm, one entry per atom
            the bus carries.  Must be non-empty.
        occupied: Every atom position in the zone, in nm.  Used to detect
            a non-participating atom sitting on one of the bus's tone
            intersections, which the AOD would carry along.
        label: Prefix identifying the bus, for error messages.

    Raises:
        ValueError: If the transport is not separable, not
            order-preserving, changes its tone count, or would pick up an
            atom that does not belong to it.
    """
    if not pairs:
        raise ValueError(f"{label}: a bus must carry at least one atom")

    src_x = sorted({p[0][0] for p in pairs})
    src_y = sorted({p[0][1] for p in pairs})
    dst_x = sorted({p[1][0] for p in pairs})
    dst_y = sorted({p[1][1] for p in pairs})

    if len(src_x) != len(dst_x) or len(src_y) != len(dst_y):
        raise ValueError(
            f"{label}: the bus spans a {len(src_x)}x{len(src_y)} grid of tones "
            f"at its source but {len(dst_x)}x{len(dst_y)} at its destination. "
            "An AOD cannot add or drop a tone mid-transport."
        )

    # Separability and order preservation: an atom's column decides its
    # x-displacement and its row decides its y-displacement, and sorted
    # tones map to sorted tones so none of them cross.
    column = {v: i for i, v in enumerate(src_x)}
    row = {v: i for i, v in enumerate(src_y)}
    for (sx, sy), (dx, dy) in pairs:
        want = (dst_x[column[sx]], dst_y[row[sy]])
        if (dx, dy) != want:
            raise ValueError(
                f"{label}: the atom at {_um((sx, sy))} µm is sent to "
                f"{_um((dx, dy))} µm, but its column and row land at "
                f"{_um(want)} µm. An AOD sweeps whole x- and y-tones, which "
                "cannot cross or trade places, so every atom must stay on "
                "its own."
            )

    # Stowaways: the AOD traps at every intersection of its tones, so an
    # atom sitting at an intersection is carried whether or not it belongs
    # to this bus.  An intersection that no atom occupies is harmless —
    # the trap simply arrives empty — so the test is occupancy, not
    # completeness.
    carried = {p[0] for p in pairs}
    intruders = sorted(
        (x, y)
        for x in src_x
        for y in src_y
        if (x, y) in occupied and (x, y) not in carried
    )
    if intruders:
        listed = [_um(p) for p in intruders]
        raise ValueError(
            f"{label}: {listed} µm "
            f"{'is a site' if len(listed) == 1 else 'are sites'} that this "
            "bus does not carry, but the AOD traps at every intersection of "
            "its tones, so whatever sits there would be carried along too. "
            "Exclude the atom from the bus's tone grid, or add it to the bus."
        )
