"""Geometry for architecture construction, including transport paths and blockade pairs.

All internal geometry is stored as nm-integer values so that set
membership, tuple equality, and candidate hashing are exact.  User-facing
APIs accept and return µm floats; conversion happens at the boundary.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left, bisect_right
from itertools import pairwise
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import rustworkx as rx
from bloqade.lanes.bytecode import Zone
from bloqade.lanes.bytecode.encoding import MoveType, Direction, LaneAddress

# Internal length unit: 1 nm.  All user-facing lengths are in µm;
# internally we convert to nm integers so that path search (hashing,
# set membership, equality) and blockade-radius distance comparisons
# are exact without floating-point hazards.
NM_PER_UM = 1000


def to_nm(value_um: float, name: str) -> int:
    """Convert a µm length to an integer nm count, validating precision.

    Raises ``ValueError`` if *value_um* is NaN, infinite, or has sub-nm
    resolution.
    """
    if not math.isfinite(value_um):
        raise ValueError(f"{name} must be finite, got {value_um}")
    scaled = value_um * NM_PER_UM
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-6:
        raise ValueError(f"{name} {value_um} µm is not representable at 1 nm precision")
    return int(rounded)


def path_nm_to_um(
    path_nm: tuple[tuple[int, int], ...],
) -> tuple[tuple[float, float], ...]:
    """Convert an nm-integer waypoint sequence to µm floats."""
    return tuple((w[0] / NM_PER_UM, w[1] / NM_PER_UM) for w in path_nm)


@dataclass(frozen=True)
class ZoneGeometry:
    """Everything the geometry routines need to know about one zone.

    A zone's grid supplies the coordinates; the word template supplies the
    ``(x_idx, y_idx)`` pairs that index into it.  Under the spec's model the
    template is shared across zones, so a ``ZoneGeometry`` is that shared
    template resolved against this zone's own coordinates.

    Attributes:
        name: Zone name, used only in diagnostics.
        grid_x_nm: Grid x-coordinates in nm, ascending.
        grid_y_nm: Grid y-coordinates in nm, ascending.
        x_clearance_nm: Minimum x-axis distance a waypoint must keep from
            every grid line.
        y_clearance_nm: Same, for the y-axis.
        words: ``words[word_id][site_id]`` is the ``(x_idx, y_idx)`` grid
            index pair for that site.
    """

    name: str
    grid_x_nm: tuple[int, ...]
    grid_y_nm: tuple[int, ...]
    x_clearance_nm: int
    y_clearance_nm: int
    words: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def num_words(self) -> int:
        """Number of words in the template."""
        return len(self.words)

    @property
    def sites_per_word(self) -> int:
        """Sites in each word; 0 when the template is empty."""
        return len(self.words[0]) if self.words else 0

    def site_nm(self, word_id: int, site_id: int) -> tuple[int, int]:
        """Physical (x, y) position of a site, in nm integers."""
        x_idx, y_idx = self.words[word_id][site_id]
        return (self.grid_x_nm[x_idx], self.grid_y_nm[y_idx])

    def word_origin(self, word_id: int) -> tuple[int, int]:
        """The ``(min_x_idx, min_y_idx)`` origin of a word, in grid indices."""
        positions = self.words[word_id]
        return (min(p[0] for p in positions), min(p[1] for p in positions))


# ── Path search ──


def enumerate_safe_positions(
    grid_positions: Sequence[int],
    source_positions: Sequence[int],
    min_cl_nm: int,
) -> list[int]:
    """Enumerate reference positions on one axis that keep the bus clear.

    Returns sorted integer positions ``p`` (nm) such that, when the
    reference atom is at ``p`` and every other atom follows by the same
    shift (AOD invariant), every atom is at least ``min_cl_nm`` from every
    grid line on this axis.

    Candidate positions are chosen to maximize clearance rather than
    sitting exactly on the ``min_cl_nm`` threshold:

    * **Midpoints** between consecutive grid lines.  When a shifted atom
      lands at a midpoint, its distance to the two neighboring grid lines
      is half the grid gap — which is typically larger than ``min_cl_nm``
      on non-uniform grids, giving extra breathing room.
    * **Boundary edges** at ``min_grid - min_cl_nm`` and
      ``max_grid + min_cl_nm``, so the search can route around the outside
      of the grid when the interior is too crowded.

    All candidates are filtered so that the distance to every grid line,
    for every offset in the bus, is ``>= min_cl_nm``.
    """
    if not source_positions:
        return []

    ref_src = source_positions[0]
    offsets = sorted({s - ref_src for s in source_positions})
    sorted_grid = sorted(set(grid_positions))

    candidates: set[int] = set()

    # Midpoints between consecutive grid lines, shifted per offset.
    # ``p = mid - off`` places the atom at offset ``off`` exactly at the
    # midpoint.
    for i in range(len(sorted_grid) - 1):
        mid = (sorted_grid[i] + sorted_grid[i + 1]) // 2
        for off in offsets:
            candidates.add(mid - off)

    # Outer boundary edges for routing around the grid.
    if sorted_grid:
        for off in offsets:
            candidates.add(sorted_grid[0] - off - min_cl_nm)
            candidates.add(sorted_grid[-1] - off + min_cl_nm)

    def _valid(p: int) -> bool:
        for off in offsets:
            shifted = p + off
            for g in grid_positions:
                if abs(shifted - g) < min_cl_nm:
                    return False
        return True

    return sorted(c for c in candidates if _valid(c))


def merge_collinear(
    path: list[tuple[int, int]],
    segment_safe: Callable[[tuple[int, int], tuple[int, int]], bool],
) -> tuple[tuple[int, int], ...]:
    """Collapse consecutive same-axis waypoints into longer segments.

    Walks the path and, for each axis-aligned run, extends the anchor to
    the farthest reachable point whose direct segment passes
    ``segment_safe``.  Positions within an axis run are monotonic
    (shortest-path guarantee), so once a merge is blocked all subsequent
    points on the same axis are also blocked.
    """
    if len(path) <= 2:
        return tuple(path)

    merged: list[tuple[int, int]] = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = i + 1
        # Determine which coordinate must stay constant (the axis).
        is_horizontal = path[j][1] == merged[-1][1]
        coord = 1 if is_horizontal else 0
        anchor_val = merged[-1][coord]

        # Extend as far as possible on the current axis.
        best = j
        for k in range(j + 1, len(path)):
            if path[k][coord] != anchor_val:
                break
            if segment_safe(merged[-1], path[k]):
                best = k
            else:
                break

        merged.append(path[best])
        i = best

    return tuple(merged)


def apply_deltas(
    lane_src: tuple[int, int],
    ref_waypoints: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Build a lane's waypoint sequence by shifting the reference path."""
    ref_src = ref_waypoints[0]
    dx0 = lane_src[0] - ref_src[0]
    dy0 = lane_src[1] - ref_src[1]
    return tuple((w[0] + dx0, w[1] + dy0) for w in ref_waypoints)


def search_path(
    geom: ZoneGeometry,
    ref_src: tuple[int, int],
    ref_dst: tuple[int, int],
    bus_src_positions: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    """Graph-based shortest path for a bus's reference atom, in nm space.

    Builds a position graph where nodes are safe waypoint positions and
    edges are axis-aligned moves validated against bus-level grid
    crossings.  Dijkstra's algorithm finds the shortest-distance path, and
    a merge pass collapses consecutive same-axis segments.

    Returns a waypoint sequence ``[ref_src, ..., ref_dst]`` or ``None`` if
    no valid path exists.
    """
    src_xs = [p[0] for p in bus_src_positions]
    src_ys = [p[1] for p in bus_src_positions]
    safe_xs = set(enumerate_safe_positions(geom.grid_x_nm, src_xs, geom.x_clearance_nm))
    safe_ys = set(enumerate_safe_positions(geom.grid_y_nm, src_ys, geom.y_clearance_nm))

    x_candidates = sorted({ref_src[0], ref_dst[0], *safe_xs})
    y_candidates = sorted({ref_src[1], ref_dst[1], *safe_ys})

    offsets = [(p[0] - ref_src[0], p[1] - ref_src[1]) for p in bus_src_positions]
    grid_xs = geom.grid_x_nm
    grid_ys = geom.grid_y_nm
    grid_xs_set = set(grid_xs)
    grid_ys_set = set(grid_ys)

    # ── Build graph nodes ──
    # A node is valid if it is src/dst or a safe middle waypoint
    # (x in safe_xs OR y in safe_ys).
    pos_to_idx: dict[tuple[int, int], int] = {}
    idx_to_pos: list[tuple[int, int]] = []

    for x in x_candidates:
        x_safe = x in safe_xs
        for y in y_candidates:
            pos = (x, y)
            if pos == ref_src or pos == ref_dst or x_safe or y in safe_ys:
                pos_to_idx[pos] = len(idx_to_pos)
                idx_to_pos.append(pos)

    if ref_src not in pos_to_idx or ref_dst not in pos_to_idx:
        return None

    graph: rx.PyGraph = rx.PyGraph()
    graph.add_nodes_from(range(len(idx_to_pos)))

    # ── Build edges via blocking-position sweep ──
    # Group nodes by row (y) for horizontal edges, by column (x) for
    # vertical edges.
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for pos, idx in pos_to_idx.items():
        rows[pos[1]].append(idx)
        cols[pos[0]].append(idx)

    # Horizontal edges: for each row, compute blocking x-positions from bus
    # offsets, then connect adjacent candidates without a blocker strictly
    # between them.
    for y, node_indices in rows.items():
        # Blocking ref-x positions: grid x values shifted by -off_x for
        # each offset whose shifted y lands on a grid row.
        blockers: list[int] = []
        for off_x, off_y in offsets:
            if (y + off_y) in grid_ys_set:
                for g_x in grid_xs:
                    blockers.append(g_x - off_x)
        sorted_blockers = sorted(set(blockers))

        # Sort nodes on this row by x-coordinate.
        node_indices.sort(key=lambda i: idx_to_pos[i][0])

        for a, b in pairwise(node_indices):
            x_a = idx_to_pos[a][0]
            x_b = idx_to_pos[b][0]
            # Check if any blocker lies strictly between x_a and x_b.
            lo = bisect_right(sorted_blockers, x_a)
            hi = bisect_left(sorted_blockers, x_b)
            if lo >= hi:
                # No blocker in (x_a, x_b) → safe edge.
                graph.add_edge(a, b, x_b - x_a)

    # Vertical edges: same logic transposed.
    for x, node_indices in cols.items():
        blockers = []
        for off_x, off_y in offsets:
            if (x + off_x) in grid_xs_set:
                for g_y in grid_ys:
                    blockers.append(g_y - off_y)
        sorted_blockers = sorted(set(blockers))

        node_indices.sort(key=lambda i: idx_to_pos[i][1])

        for a, b in pairwise(node_indices):
            y_a = idx_to_pos[a][1]
            y_b = idx_to_pos[b][1]
            lo = bisect_right(sorted_blockers, y_a)
            hi = bisect_left(sorted_blockers, y_b)
            if lo >= hi:
                graph.add_edge(a, b, y_b - y_a)

    # ── Dijkstra's shortest path ──
    src_idx = pos_to_idx[ref_src]
    dst_idx = pos_to_idx[ref_dst]

    paths = rx.dijkstra_shortest_paths(graph, src_idx, target=dst_idx, weight_fn=float)
    if dst_idx not in paths:
        return None

    raw_path = [idx_to_pos[i] for i in paths[dst_idx]]

    # ── Merge consecutive same-axis segments ──
    # Dijkstra's path uses fine-grained adjacent-candidate steps.  Collapse
    # runs on the same axis where the direct segment does not cross any
    # grid atom for the bus.
    def _segment_safe(start: tuple[int, int], end: tuple[int, int]) -> bool:
        if start[1] == end[1]:
            for off_x, off_y in offsets:
                if (start[1] + off_y) not in grid_ys_set:
                    continue
                lo = min(start[0], end[0]) + off_x
                hi = max(start[0], end[0]) + off_x
                for g in grid_xs:
                    if lo < g < hi:
                        return False
        else:
            for off_x, off_y in offsets:
                if (start[0] + off_x) not in grid_xs_set:
                    continue
                lo = min(start[1], end[1]) + off_y
                hi = max(start[1], end[1]) + off_y
                for g in grid_ys:
                    if lo < g < hi:
                        return False
        return True

    return merge_collinear(raw_path, _segment_safe)


# ── Transport paths ──


def compute_transport_paths(
    geom: ZoneGeometry,
    *,
    zone: Zone,
    zone_id: int,
    warn_stacklevel: int = 2,
) -> dict[LaneAddress, tuple[tuple[float, float], ...]]:
    """Compute axis-aligned AOD waypoint paths for one zone's buses.

    For each bus the reference atom's path is searched, and every other
    lane derives its waypoints by applying the same per-segment deltas.
    Site bus paths are intra-word; word bus paths are intra-zone.  Zone
    buses are not included — inter-zone routing is separate.

    The reference atom, the obstacle set, and the emitted lanes are all
    drawn from ``zone.words_with_site_buses`` rather than from every word in the
    template.  Only those words appear in the zone's
    ``words_with_site_buses``, so only they are carried when a site bus
    fires; taking the reference from a non-participating word would apply
    that word's displacement to the words that do participate and store a
    path ending away from its destination.

    Args:
        geom: The zone's resolved geometry.
        zone: Native zone supplying the buses and their participating sites and words.
        zone_id: Zone index, for the emitted ``LaneAddress`` values.
        warn_stacklevel: ``stacklevel`` for the skip warnings, so they
            point at the caller's caller rather than at this module.

    Returns:
        Dict mapping ``LaneAddress`` to waypoint tuples (µm floats) for
        both directions.
    """
    paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
    movers = zone.words_with_site_buses

    # ── Site bus paths (intra-word) ──
    for bus_id, bus in enumerate(zone.site_buses):
        src_sites, dst_sites = bus.src, bus.dst
        if not movers or not src_sites:
            continue

        # The search derives one reference path and applies its deltas to
        # every lane, so it handles rigid translations only.  Every
        # participating word is compared, not just the first: sampling one
        # would apply its delta to a word of different internal pitch.
        displacements = {
            (
                geom.site_nm(w, ds)[0] - geom.site_nm(w, ss)[0],
                geom.site_nm(w, ds)[1] - geom.site_nm(w, ss)[1],
            )
            for w in movers
            for ss, ds in zip(src_sites, dst_sites)
        }
        if len(displacements) > 1:
            warnings.warn(
                f"Zone '{geom.name}' site bus {bus_id}: inconsistent "
                f"site displacements {sorted(displacements)} violate "
                f"the AOD single-shift invariant. "
                f"Skipping path generation for this bus.",
                stacklevel=warn_stacklevel,
            )
            continue

        bus_src_atoms = [geom.site_nm(w, s) for w in movers for s in src_sites]

        ref_word = movers[0]
        ref_src = geom.site_nm(ref_word, src_sites[0])
        ref_dst = geom.site_nm(ref_word, dst_sites[0])

        if ref_src == ref_dst:
            ref_waypoints: tuple[tuple[int, int], ...] = (ref_src, ref_dst)
        else:
            result = search_path(geom, ref_src, ref_dst, bus_src_atoms)
            if result is None:
                warnings.warn(
                    f"Zone '{geom.name}' site bus {bus_id}: no valid path "
                    f"found (x_clearance={geom.x_clearance_nm / NM_PER_UM}, "
                    f"y_clearance={geom.y_clearance_nm / NM_PER_UM}). "
                    f"Skipping path generation for this bus.",
                    stacklevel=warn_stacklevel,
                )
                continue
            ref_waypoints = result

        for local_word in movers:
            for src_s in src_sites:
                lane_src = geom.site_nm(local_word, src_s)
                lane_path = path_nm_to_um(apply_deltas(lane_src, ref_waypoints))
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    lane = LaneAddress(
                        MoveType.SITE,
                        local_word,
                        src_s,
                        bus_id,
                        direction,
                        zone_id,
                    )
                    paths[lane] = (
                        lane_path if direction == Direction.FORWARD else lane_path[::-1]
                    )

    # ── Word bus paths (intra-zone) ──
    for bus_id, word_bus in enumerate(zone.word_buses):
        src_words, dst_words = word_bus.src, word_bus.dst
        sites = zone.word_bus_site_pairs(bus_id)
        if not src_words or not sites:
            continue

        transports = [
            (geom.site_nm(sw, ss), geom.site_nm(dw, ds))
            for sw, dw in zip(src_words, dst_words)
            for ss, ds in sites
        ]
        displacements = {(dx - sx, dy - sy) for (sx, sy), (dx, dy) in transports}
        if len(displacements) > 1:
            warnings.warn(
                f"Zone '{geom.name}' word bus {bus_id}: inconsistent "
                f"word displacements {sorted(displacements)} violate "
                f"the AOD single-shift invariant. "
                f"Skipping path generation for this bus.",
                stacklevel=warn_stacklevel,
            )
            continue

        bus_src_atoms = [src for src, _ in transports]
        ref_src, ref_dst = transports[0]

        if ref_src == ref_dst:
            ref_waypoints = (ref_src, ref_dst)
        else:
            result = search_path(geom, ref_src, ref_dst, bus_src_atoms)
            if result is None:
                warnings.warn(
                    f"Zone '{geom.name}' word bus {bus_id}: no valid path "
                    f"found (x_clearance={geom.x_clearance_nm / NM_PER_UM}, "
                    f"y_clearance={geom.y_clearance_nm / NM_PER_UM}). "
                    f"Skipping path generation for this bus.",
                    stacklevel=warn_stacklevel,
                )
                continue
            ref_waypoints = result

        for src_w in src_words:
            for site_id, _ in sites:
                lane_src = geom.site_nm(src_w, site_id)
                lane_path = path_nm_to_um(apply_deltas(lane_src, ref_waypoints))
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    lane = LaneAddress(
                        MoveType.WORD,
                        src_w,
                        site_id,
                        bus_id,
                        direction,
                        zone_id,
                    )
                    paths[lane] = (
                        lane_path if direction == Direction.FORWARD else lane_path[::-1]
                    )

    return paths


# ── Blockade pairs ──


def scan_blockade_pairs(geom: ZoneGeometry, radius_nm: int) -> list[tuple[int, int]]:
    """Scan word pairs and classify under the matching-index CZ rule.

    Returns the list of valid ``(a, b)`` word pairs with ``a < b``.  Raises
    ``ValueError`` on partial blockade, crossed-index, or multi-partner
    cases.

    Uses ``scipy.spatial.KDTree.query_pairs`` to enumerate only the site
    pairs within ``radius_nm`` (O(n log n + k) over all sites in the zone,
    rather than O(n² · spw²) all-to-all).  Coordinates are fed in as nm
    integers so the ``<= radius_nm`` cutoff lands on exact boundaries
    without float drift.
    """
    n = geom.num_words
    spw = geom.sites_per_word
    if n * spw < 2:
        # Need at least two sites anywhere in the zone before a pair can
        # even exist; skip the KDTree build.
        return []

    from scipy.spatial import KDTree

    # Flatten every site into one KDTree, tracking (word, site_index) so we
    # can classify each returned pair.
    positions = [geom.site_nm(w, s) for w in range(n) for s in range(spw)]
    owners = [(w, s) for w in range(n) for s in range(spw)]

    tree = KDTree(positions)
    # query_pairs returns (i, j) with i < j and dist(p_i, p_j) <= radius_nm.
    raw_pairs = tree.query_pairs(radius_nm, output_type="set")

    # For each cross-word pair within the radius, record which matching
    # site-indices fell within (and bail immediately on crossed-index).
    matching_sites: dict[tuple[int, int], set[int]] = {}
    radius_um = radius_nm / NM_PER_UM
    for i, j in raw_pairs:
        w1, s1 = owners[i]
        w2, s2 = owners[j]
        if w1 == w2:
            # Two sites of the same word are within blockade — the layout
            # can't support a CZ at this radius (it would entangle atoms
            # inside a single word).
            raise ValueError(
                f"Zone '{geom.name}' blockade scan: word {w1} "
                f"has two intra-word sites ({s1} and {s2}) within "
                f"radius {radius_um} µm. Entanglement within a "
                f"single word is not allowed — tighten "
                f"blockade_radius or space the word's sites apart."
            )
        if s1 != s2:
            raise ValueError(
                f"Zone '{geom.name}' blockade scan: words "
                f"{min(w1, w2)} and {max(w1, w2)} have a "
                f"non-matching-index site pair (site {s1} ↔ "
                f"site {s2}) within radius {radius_um} µm "
                f"(crossed-index blockade). The layout cannot "
                f"be cleanly paired under the CZ matching-index "
                f"convention."
            )
        # Matching-index pair. Canonicalize only the word pair.
        matching_sites.setdefault((min(w1, w2), max(w1, w2)), set()).add(s1)

    # A clean CZ pair has all `spw` matching-index site-pairs within
    # radius; fewer is a partial blockade.
    valid_pairs: list[tuple[int, int]] = []
    for (a, b), sites in matching_sites.items():
        if len(sites) != spw:
            raise ValueError(
                f"Zone '{geom.name}' blockade scan: words "
                f"{a} and {b} have {len(sites)}/{spw} "
                f"matching-index site pairs within radius "
                f"{radius_um} µm (partial blockade). The layout "
                f"cannot be cleanly paired under the CZ "
                f"matching-index convention."
            )
        valid_pairs.append((a, b))

    # Deterministic order (`query_pairs` returns a set).
    valid_pairs.sort()

    # Each word must appear in at most one valid pair.
    partner: dict[int, int] = {}
    for a, b in valid_pairs:
        for x, y in ((a, b), (b, a)):
            if x in partner and partner[x] != y:
                raise ValueError(
                    f"Zone '{geom.name}' blockade scan: word {x} "
                    f"has multiple blockade partners: "
                    f"{partner[x]}, {y}. Tighten blockade_radius "
                    f"or adjust the word layout."
                )
            partner[x] = y

    return valid_pairs
