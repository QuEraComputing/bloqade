"""AOD realizability rules for architecture buses.

An AOD is a crossed pair of deflectors: its traps sit at the Cartesian
product of the x-tones and y-tones, and each tone is swept independently.
A transport must cover every source tone crossing and be separable and
order-preserving:

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

from collections.abc import Sequence

from bloqade.gemini.arch.build._geometry import NM_PER_UM

Position = tuple[int, int]
Pair = tuple[Position, Position]


def _um(p: Position) -> tuple[float, float]:
    """Render an nm position in µm, for diagnostics."""
    return (p[0] / NM_PER_UM, p[1] / NM_PER_UM)


def check_aod_transport(
    pairs: Sequence[Pair],
    *,
    label: str,
) -> None:
    """Reject a bus no AOD operation can perform.

    Args:
        pairs: ``(src_position, dst_position)`` in nm, one entry per atom
            the bus carries.  Must be non-empty.
        label: Prefix identifying the bus, for error messages.

    Raises:
        ValueError: If the transport is not separable, not
            order-preserving, changes its tone count, or skips a source crossing.
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

    carried = {src for src, _ in pairs}
    if len(carried) != len(src_x) * len(src_y):
        raise ValueError(f"{label}: bus sources must form a complete grid of tones")
