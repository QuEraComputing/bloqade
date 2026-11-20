"""Spec-shaped architecture builder.

``ArchBuilder`` owns everything the :class:`ArchSpec` owns at spec scope —
the word template and the grid index space it indexes — and treats a zone
as what the spec says it is: a name, a set of coordinates for that shared
index space, and the buses that run on it.

The API is phased, and the phase order is what makes every check
decidable at the call that raises it:

1. **Words.**  ``add_word`` defines the spec-wide template.  A word is a
   set of grid ``(row, column)`` index pairs, so the template is
   independent of any zone's coordinates.  Every index-space API here is
   ``(rows, columns)``, row-major, including the ``words[...]`` and
   ``sites[...]`` selections.
2. **Zones.**  ``add_zone`` supplies coordinates for that index space,
   plus which words and sites take part in transport there.  The template
   is frozen from the first zone on, because a later word would change the
   occupancy and the participant set of every zone already added.
3. **Buses.**  Always zone-qualified.  By this point the carried-atom set
   of every bus is fully determined, so realizability is checked here
   rather than deferred to build time.
4. **Spec scope.**  Blockade radius, modes, capabilities, then ``build``.

"""

from __future__ import annotations

import operator
from collections import Counter
from dataclasses import field, dataclass
from collections.abc import Sequence

from bloqade.lanes.bytecode import SiteBus, ArchSpecError
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.bytecode._native import (
    Grid as _RustGrid,
    Mode as _RustMode,
    Zone as _RustZone,
    WordBus as _RustWordBus,
    ZoneBus as _RustZoneBus,
    LocationAddress as _RustLocAddr,
)
from bloqade.lanes.bytecode.encoding import MoveType, LaneAddress

from bloqade.gemini.arch.build._aod import check_aod_transport
from bloqade.gemini.arch.build._geometry import (
    ZoneGeometry,
    to_nm,
    scan_blockade_pairs,
    compute_transport_paths,
)

Index = slice | int | Sequence[int]


# ── Index normalization ──


def _query_axis(idx: Index, size: int, axis: str, what: str) -> list[int]:
    """Normalize one axis of a grid query, rejecting bad indices.

    The selection is returned **in the order given** — a sequence
    verbatim, a slice as it expands (including a negative step).  A query
    as a whole then behaves like ``arr[np.ix_(y_values, x_values)].ravel()``:
    the order within each axis is the caller's, and "row-major" describes
    only the nesting, with y as the outer loop.

    Out-of-range, negative and repeated indices raise rather than
    selecting nothing, so a typo cannot silently shorten a selection into
    a smaller-but-still-valid bus.  Slices are clamped by ``slice.indices``
    and are always valid.
    """
    if isinstance(idx, slice):
        return list(range(*idx.indices(size)))
    values = [idx] if isinstance(idx, int) else list(idx)
    out_of_range = sorted({i for i in values if not 0 <= i < size})
    if out_of_range:
        raise IndexError(f"{axis} {what} {out_of_range} out of range [0, {size})")
    repeated = sorted(i for i, n in Counter(values).items() if n > 1)
    if repeated:
        raise ValueError(
            f"{axis} {what} {repeated} repeated in {values}; each position is "
            "selected once, so a repeat cannot select anything new"
        )
    return values


def _checked_subset(values: Sequence[int], size: int, what: str) -> tuple[int, ...]:
    """Validate an explicit participation subset: in range and unique."""
    out = list(values)
    bad = sorted({v for v in out if not 0 <= v < size})
    if bad:
        raise IndexError(f"{what} {bad} out of range [0, {size})")
    repeated = sorted(v for v, n in Counter(out).items() if n > 1)
    if repeated:
        raise ValueError(f"{what} {repeated} repeated in {out}")
    return tuple(out)


def _uniform_row_major_shape(
    words: Sequence[Sequence[tuple[int, int]]],
) -> tuple[int, int] | None:
    """The ``(rows, columns)`` shape every word shares, or ``None``.

    ``None`` means the template is something ``word_shape`` cannot describe:
    a word listed other than row-major, a word that is not a Cartesian
    product, or words of differing shapes.  ``ArchSpec`` permits all three —
    it stores a word as an ordered list of grid positions and only requires
    every word to hold the same *number* of sites.
    """
    shapes: set[tuple[int, int]] = set()
    for sites in words:
        rows = list(dict.fromkeys(y for _, y in sites))
        cols = list(dict.fromkeys(x for x, _ in sites))
        if [(c, r) for r in rows for c in cols] != list(sites):
            return None
        shapes.add((len(rows), len(cols)))
    return shapes.pop() if len(shapes) == 1 else None


# ── Template queries ──


class _SiteQuery:
    """Select site indices within the word shape, as ``[rows, columns]``.

    Traversal is row-major over the selected positions — rows the outer
    loop, columns the inner — with each axis visited in the order given.
    Site index is ``column + row * num_columns``.
    """

    def __init__(self, word_shape: tuple[int, int]):
        self._rows, self._cols = word_shape

    def __getitem__(self, key: tuple[Index, Index]) -> list[int]:
        row_idx, col_idx = key
        rows = _query_axis(row_idx, self._rows, "row", "site index")
        cols = _query_axis(col_idx, self._cols, "column", "site index")
        return [c + r * self._cols for r in rows for c in cols]


class _WordQuery:
    """Select word IDs by grid region, as ``[rows, columns]``.

    The selected positions are traversed row-major — rows the outer loop,
    columns the inner, each axis in the order given — and each word is
    taken at its first arrival, since a word spans several grid positions.

    Ordering by traversal rather than by word ID matters whenever word IDs
    are not monotone in the grid, which is the normal case for interleaved
    CZ layouts: scanning a row visits words 0, 1, 2, 3, 0, 1, 2, 3, ..., so
    a partial column selection reaches them out of ID order.  Because a bus's
    ``src`` and ``dst`` are related by a separable, order-preserving AOD
    transport, selecting both endpoints the same way pairs ``src[i]`` with
    ``dst[i]`` correctly.

    The query lives on the builder rather than on a zone because the
    template is spec-wide: the answer is the same for every zone.
    """

    def __init__(self, builder: ArchBuilder):
        self._builder = builder

    def __getitem__(self, key: tuple[Index, Index]) -> list[int]:
        row_idx, col_idx = key
        n_rows, n_cols = self._builder.grid_shape
        rows = _query_axis(row_idx, n_rows, "row", "grid index")
        cols = _query_axis(col_idx, n_cols, "column", "grid index")
        result: list[int] = []
        seen: set[int] = set()
        for r in rows:
            for c in cols:
                word_id = self._builder._position_to_word.get((c, r))
                if word_id is not None and word_id not in seen:
                    seen.add(word_id)
                    result.append(word_id)
        return result


# ── Zone ──


@dataclass
class _Zone:
    """One zone: coordinates for the shared index space, plus its buses.

    A plain value. It carries no validation logic, because every fact that
    decides whether its buses are legal is owned by the builder.
    """

    name: str
    grid: _RustGrid
    grid_x_nm: tuple[int, ...]
    grid_y_nm: tuple[int, ...]
    x_clearance_nm: int | None
    y_clearance_nm: int | None
    words_with_site_buses: tuple[int, ...]
    sites_with_word_buses: tuple[int, ...]
    site_buses: list[SiteBus] = field(default_factory=list)
    word_buses: list[_RustWordBus] = field(default_factory=list)
    entangling_pairs: list[tuple[int, int]] = field(default_factory=list)
    blockade_radius_nm: int | None = None


# ── ArchBuilder ──


class ArchBuilder:
    """Build an :class:`ArchSpec` from a shared word template and zones."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        word_shape: tuple[int, int],
    ):
        """Initialize the builder.

        Args:
            grid_shape: ``(num_rows, num_columns)`` size of the grid index
                space that word templates index into.  Every zone supplies
                exactly this many coordinates on each axis.
            word_shape: ``(num_rows, num_columns)`` of sites in every word;
                ``sites_per_word`` is their product.

        Raises:
            ValueError: If either shape is not a pair of positive integers.
        """

        def _shape(label: str, shape: tuple[int, int]) -> tuple[int, int]:
            # ``int(2.5)`` would silently accept a malformed index space, so
            # require values that are integers already.
            if len(shape) != 2:
                raise ValueError(f"{label} must be two positive ints, got {shape!r}")
            out = []
            for v in shape:
                if isinstance(v, bool) or not hasattr(v, "__index__"):
                    raise ValueError(
                        f"{label} must be two positive ints, got {shape!r}"
                    )
                i = operator.index(v)
                if i < 1:
                    raise ValueError(
                        f"{label} must be two positive ints, got {shape!r}"
                    )
                out.append(i)
            return (out[0], out[1])

        self._grid_shape = _shape("grid_shape", grid_shape)
        self._word_shape: tuple[int, int] | None = _shape("word_shape", word_shape)
        self._words: list[list[tuple[int, int]]] = []
        self._position_to_word: dict[tuple[int, int], int] = {}
        self._zones: list[_Zone] = []
        self._zone_ids: dict[str, int] = {}
        self._connections: list[tuple[tuple[int, list[int]], tuple[int, list[int]]]] = (
            []
        )
        self._modes: list[tuple[str, list[str], list | None]] = []
        self._blockade_radius: float | None = None
        self._feed_forward = False
        self._atom_reloading = False
        # Paths carried in from an existing spec, plus the bus structure
        # they were computed for.  Bundled architectures ship hardware-
        # derived lane geometry that this path search does not reproduce,
        # and lane durations (hence fidelity estimates) are read off those
        # segment lengths — so a round-trip must not silently replace them.
        self._inherited_paths: (
            dict[LaneAddress, tuple[tuple[float, float], ...]] | None
        ) = None
        # Per zone: (n_site_buses, n_word_buses, movers, carried) as of
        # from_spec, so lanes for buses that existed then stay inheritable
        # even after new buses are appended.
        self._inherited_shape: dict[
            str, tuple[int, int, tuple[int, ...], tuple[int, ...]]
        ] = {}

    # ── Shape ──

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(num_rows, num_columns)`` of the shared grid index space."""
        return self._grid_shape

    @property
    def word_shape(self) -> tuple[int, int] | None:
        """``(num_rows, num_columns)`` of sites in every word.

        ``None`` after :meth:`from_spec` restored a template that no single
        shape describes — see :func:`_uniform_row_major_shape`.
        """
        return self._word_shape

    @property
    def sites_per_word(self) -> int:
        """Total sites per word.

        Read off the template once words exist, since that holds whatever
        was restored; ``ArchSpec`` requires every word to have the same
        count even when their shapes differ.
        """
        if self._words:
            return len(self._words[0])
        assert self._word_shape is not None
        return self._word_shape[0] * self._word_shape[1]

    @property
    def num_words(self) -> int:
        """Words defined in the template so far."""
        return len(self._words)

    @property
    def num_zones(self) -> int:
        """Zones added so far."""
        return len(self._zones)

    @property
    def words(self) -> _WordQuery:
        """Select word IDs by ``[rows, columns]`` of the grid index space."""
        return _WordQuery(self)

    @property
    def sites(self) -> _SiteQuery:
        """Select site indices by ``[rows, columns]`` of the word shape.

        Raises:
            ValueError: If the template has no single row-major shape to
                query against, which :meth:`from_spec` can restore.
        """
        if self._word_shape is None:
            raise ValueError(
                "this template has no single row-major (rows x columns) "
                "shape — it was restored from a spec whose words are listed "
                "in another order, are not Cartesian products, or differ in "
                "shape. Address sites by index, or read a word's positions "
                "from word_sites()."
            )
        return _SiteQuery(self._word_shape)

    # ── Phase 1: the word template ──

    def word_sites(self, word_id: int) -> list[tuple[int, int]]:
        """The ``(row, column)`` grid position of each of a word's sites.

        Indexed by site ID, so this works for any template — including one
        restored from a spec that ``word_shape`` cannot describe.
        """
        return [(r, c) for c, r in self._words[word_id]]

    def add_word(self, rows: Index, columns: Index) -> int:
        """Add a word to the spec-wide template.

        Args:
            rows: Grid row indices for the word's sites; must number
                ``word_shape[0]``.
            columns: Grid column indices; must number ``word_shape[1]``.

        Returns:
            The new word ID.

        Raises:
            ValueError: If the index count does not match ``word_shape``,
                an index repeats, a grid position is already taken by
                another word, or a zone has already been added.
            IndexError: If an index falls outside ``grid_shape``.
        """
        if self._zones:
            raise ValueError(
                f"the word template is frozen: {len(self._zones)} zone(s) "
                "already resolve it against their own coordinates, so adding "
                "a word now would change their occupancy and their transport "
                "participants. Add every word before the first add_zone."
            )
        if self._word_shape is None:
            raise ValueError(
                "this builder restored a template with no single row-major "
                "shape, so add_word has no shape to validate against"
            )
        n_rows, n_cols = self._grid_shape
        rs = _query_axis(rows, n_rows, "row", "grid index")
        cs = _query_axis(columns, n_cols, "column", "grid index")
        if len(rs) != self._word_shape[0]:
            raise ValueError(
                f"rows has {len(rs)} indices but word_shape requires "
                f"{self._word_shape[0]}"
            )
        if len(cs) != self._word_shape[1]:
            raise ValueError(
                f"columns has {len(cs)} indices but word_shape requires "
                f"{self._word_shape[1]}"
            )

        # Stored in the spec's own ``(x_idx, y_idx)`` order, which is
        # ``(column, row)``; the row-major traversal is what fixes site IDs.
        positions = [(c, r) for r in rs for c in cs]
        for pos in positions:
            owner = self._position_to_word.get(pos)
            if owner is not None:
                raise ValueError(
                    f"grid position (row={pos[1]}, column={pos[0]}) already "
                    f"belongs to word {owner}"
                )

        word_id = len(self._words)
        self._words.append(positions)
        for pos in positions:
            self._position_to_word[pos] = word_id
        return word_id

    # ── Phase 2: zones ──

    def add_zone(
        self,
        name: str,
        rows: Sequence[float],
        columns: Sequence[float],
        *,
        x_clearance: float | None,
        y_clearance: float | None,
        words_with_site_buses: Sequence[int] | None = None,
        sites_with_word_buses: Sequence[int] | None = None,
    ) -> int:
        """Add a zone: coordinates for the shared index space.

        Args:
            name: Zone name; must be unique.
            rows: ``grid_shape[0]`` row positions — the y-coordinate of
                each grid row, in µm, each representable at 1 nm precision.
            columns: ``grid_shape[1]`` column positions — the x-coordinate
                of each grid column, in µm.
            x_clearance: Minimum x-axis distance (> 0, µm) that path
                waypoints keep from every grid line.  ``None`` only when
                paths are inherited from an existing spec via
                :meth:`from_spec`, since the search cannot run without it.
            y_clearance: Same, for the y-axis.
            words_with_site_buses: Words eligible for site-bus transport in
                this zone.  Defaults to every word.
            sites_with_word_buses: Sites carried by this zone's word buses.
                Defaults to every site.

        Returns:
            The new zone ID.

        Raises:
            ValueError: On a duplicate name, a coordinate count that does
                not match ``grid_shape``, a non-positive or non-nm-precise
                clearance, no words defined yet, or a repeated
                participation entry.
            IndexError: If a participation entry is out of range.
        """
        if not self._words:
            raise ValueError(
                "define the word template with add_word before adding a zone; "
                "a zone resolves that template against its own coordinates."
            )
        if name in self._zone_ids:
            raise ValueError(f"duplicate zone name: '{name}'")
        n_rows, n_cols = self._grid_shape
        if len(rows) != n_rows or len(columns) != n_cols:
            raise ValueError(
                f"zone '{name}' supplies {len(rows)} rows x {len(columns)} "
                f"columns but grid_shape is {n_rows}x{n_cols}; every zone "
                "indexes the same shared grid index space."
            )
        if x_clearance is not None and x_clearance <= 0:
            raise ValueError(f"x_clearance must be positive, got {x_clearance}")
        if y_clearance is not None and y_clearance <= 0:
            raise ValueError(f"y_clearance must be positive, got {y_clearance}")

        movers = (
            tuple(range(self.num_words))
            if words_with_site_buses is None
            else _checked_subset(
                words_with_site_buses, self.num_words, "words_with_site_buses"
            )
        )
        carried = (
            tuple(range(self.sites_per_word))
            if sites_with_word_buses is None
            else _checked_subset(
                sites_with_word_buses, self.sites_per_word, "sites_with_word_buses"
            )
        )

        zone = _Zone(
            name=name,
            grid=_RustGrid.from_positions(list(columns), list(rows)),
            grid_x_nm=tuple(to_nm(v, "column position") for v in columns),
            grid_y_nm=tuple(to_nm(v, "row position") for v in rows),
            x_clearance_nm=(
                None if x_clearance is None else to_nm(x_clearance, "x_clearance")
            ),
            y_clearance_nm=(
                None if y_clearance is None else to_nm(y_clearance, "y_clearance")
            ),
            words_with_site_buses=movers,
            sites_with_word_buses=carried,
        )
        zone_id = len(self._zones)
        self._zone_ids[name] = zone_id
        self._zones.append(zone)
        return zone_id

    # ── Phase 3: buses ──

    def add_site_bus(self, zone: str, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a site bus (intra-word movement) to one zone.

        The atoms it carries are every participating word's ``src`` sites,
        which is fixed once the template is frozen and the zone declares
        its participants — so realizability is decided here.

        Args:
            zone: Zone name.
            src: Source site indices within the word shape.
            dst: Destination site indices, parallel to ``src``.

        Raises:
            ValueError: On mismatched or empty sequences, an out-of-range
                site index, or a transport no AOD can perform.
        """
        z = self._zone(zone)
        self._check_bus_endpoints(src, dst, self.sites_per_word, "site")
        if not z.words_with_site_buses:
            raise ValueError(
                f"zone '{zone}' has no words_with_site_buses, so a site bus "
                "there would carry nothing. Declare the participating words "
                "on add_zone."
            )
        bus_id = len(z.site_buses)
        bus = SiteBus(src=list(src), dst=list(dst))
        self._reject_duplicate_bus(z.site_buses, bus, "site", zone)
        geom = self._geometry(z)
        check_aod_transport(
            [
                (geom.site_nm(w, s), geom.site_nm(w, d))
                for w in z.words_with_site_buses
                for s, d in zip(src, dst)
            ]
            or [],
            label=f"site bus {bus_id} on zone '{zone}'",
        )
        z.site_buses.append(bus)

    def _reject_duplicate_bus(
        self,
        existing: Sequence[SiteBus | _RustWordBus],
        bus: SiteBus | _RustWordBus,
        kind: str,
        zone: str,
    ) -> None:
        if bus in existing:
            raise ValueError(
                f"{kind} bus {len(existing)} on zone '{zone}': {kind} bus "
                f"{existing.index(bus)} already has this exact ordered mapping"
            )

    def add_word_bus(
        self,
        zone: str,
        src: Sequence[int],
        dst: Sequence[int],
        *,
        site_map: SiteBus | None = None,
    ) -> None:
        """Add a word bus (intra-zone movement) to one zone.

        Args:
            zone: Zone name.
            src: Source word IDs.
            dst: Destination word IDs, parallel to ``src``.
            site_map: Optional ``bloqade.lanes.bytecode.SiteBus`` mapping source
                sites to destination sites on every word pair. Both ends must
                belong to the zone's ``sites_with_word_buses``. Omitting the
                map carries all participating sites to the same site IDs.

        Raises:
            ValueError: On mismatched or empty sequences, an out-of-range
                word ID, or a transport no AOD can perform.
        """
        z = self._zone(zone)
        self._check_bus_endpoints(src, dst, self.num_words, "word")
        if not z.sites_with_word_buses:
            raise ValueError(
                f"zone '{zone}' has no sites_with_word_buses, so a word bus "
                "there would carry nothing. Declare the participating sites "
                "on add_zone."
            )
        bus = _RustWordBus(src=list(src), dst=list(dst), site_map=site_map)
        self._reject_duplicate_bus(z.word_buses, bus, "word", zone)

        # Validate just the proposed bus against the shared template. Rust
        # resolves its site map and checks its relation and AOD geometry. No
        # existing bus needs revalidation, and failure leaves the builder intact.
        candidate = _RustZone(
            name=zone,
            grid=z.grid,
            site_buses=[],
            word_buses=[bus],
            words_with_site_buses=[],
            sites_with_word_buses=list(z.sites_with_word_buses),
        )
        try:
            ArchSpec.from_components(
                words=tuple(Word(tuple(sites)) for sites in self._words),
                zones=(candidate,),
                modes=[],
            )
        except ArchSpecError as exc:
            raise ValueError(
                f"word bus {len(z.word_buses)} on zone '{zone}': {exc}"
            ) from exc
        z.word_buses.append(bus)

    def connect(
        self,
        src: tuple[str, Sequence[int]],
        dst: tuple[str, Sequence[int]],
    ) -> None:
        """Add an inter-zone word bus.

        Args:
            src: ``(zone_name, word_ids)`` for the source endpoint.
            dst: ``(zone_name, word_ids)`` for the destination endpoint.

        Raises:
            ValueError: On an unknown zone, mismatched or empty sequences,
                or a transport no AOD can perform.
            IndexError: If a word ID is out of range.
        """
        src_name, src_words = src
        dst_name, dst_words = dst
        src_zone = self._zone(src_name)
        dst_zone = self._zone(dst_name)
        if src_name == dst_name:
            raise ValueError(
                f"connect adds an inter-zone bus, but '{src_name}' is both "
                "endpoints. Use add_word_bus for movement inside one zone."
            )
        self._check_bus_endpoints(src_words, dst_words, self.num_words, "word")

        src_geom = self._geometry(src_zone)
        dst_geom = self._geometry(dst_zone)
        bus_id = len(self._connections)
        check_aod_transport(
            [
                (src_geom.site_nm(sw, s), dst_geom.site_nm(dw, s))
                for sw, dw in zip(src_words, dst_words)
                for s in range(self.sites_per_word)
            ],
            label=f"zone bus {bus_id} from '{src_name}' to '{dst_name}'",
        )
        self._connections.append(
            (
                (self._zone_ids[src_name], list(src_words)),
                (self._zone_ids[dst_name], list(dst_words)),
            )
        )

    # ── Phase 4: spec scope ──

    def add_entangling_pairs(
        self, zone: str, words_a: Sequence[int], words_b: Sequence[int]
    ) -> None:
        """Mark paired words as CZ pairs within one zone.

        ``words_a[i]`` is paired with ``words_b[i]``.  Any blockade radius
        previously recorded on this zone is cleared, since a manual append
        means the pair list is no longer purely radius-derived.
        """
        z = self._zone(zone)
        if len(words_a) != len(words_b):
            raise ValueError(
                f"words_a has {len(words_a)} entries but words_b has " f"{len(words_b)}"
            )
        for a, b in zip(words_a, words_b):
            for w in (a, b):
                if not 0 <= w < self.num_words:
                    raise IndexError(
                        f"word index {w} out of range [0, {self.num_words})"
                    )
            z.entangling_pairs.append((a, b))
        z.blockade_radius_nm = None

    def set_blockade_radius(self, radius: float) -> None:
        """Derive entangling word pairs in every zone from the radius.

        The radius is a spec-scope value, but the pairs it implies are
        per-zone: the same radius against different coordinates yields
        different pairs.  Every zone is scanned before any pair list is
        overwritten, so a layout error in a later zone cannot leave earlier
        zones partially updated.

        Raises:
            ValueError: If ``radius`` is not positive or nm-precise, or if
                any zone's layout is inconsistent with it.
        """
        if radius <= 0:
            raise ValueError(f"blockade_radius must be positive, got {radius}")
        radius_nm = to_nm(radius, "blockade_radius")
        scanned = [
            scan_blockade_pairs(self._geometry(z), radius_nm) for z in self._zones
        ]
        for z, pairs in zip(self._zones, scanned):
            z.entangling_pairs = pairs
            z.blockade_radius_nm = radius_nm
        self._blockade_radius = radius

    @property
    def blockade_radius(self) -> float | None:
        """Blockade radius (µm) applied to every zone, or ``None``."""
        return self._blockade_radius

    def add_mode(
        self,
        name: str,
        zones: Sequence[str],
        *,
        bitstring_order: Sequence[_RustLocAddr] | None = None,
    ) -> None:
        """Add an operational mode over a subset of zones.

        Args:
            name: Mode name.
            zones: Zone names active in this mode.
            bitstring_order: Explicit measurement bit ordering.  Defaults to
                every ``(zone, word, site)`` of the listed zones in order;
                an ``ArchSpec`` may carry any ordering, so :meth:`from_spec`
                passes the original through rather than regenerating it.
        """
        for z in zones:
            if z not in self._zone_ids:
                raise ValueError(f"unknown zone: '{z}'")
        self._modes.append(
            (
                name,
                list(zones),
                None if bitstring_order is None else list(bitstring_order),
            )
        )

    def build(
        self,
        *,
        feed_forward: bool | None = None,
        atom_reloading: bool | None = None,
        blockade_radius: float | None = None,
        recompute_paths: bool = False,
    ) -> ArchSpec:
        """Assemble the :class:`ArchSpec` and validate it via Rust.

        Args:
            feed_forward: Whether the device supports feed-forward.  Defaults
                to the value carried in by :meth:`from_spec`, else ``False``.
            atom_reloading: Whether the device supports atom reloading, with
                the same default.
            blockade_radius: Explicit radius (µm), overriding the value
                recorded by :meth:`set_blockade_radius`.
            recompute_paths: Search every lane afresh, discarding inherited
                paths even for buses that predate any edit.  Off by default,
                so neither a round-trip nor an extension silently replaces
                hardware-derived lane geometry.

        Raises:
            ValueError: If no words or zones were defined, or if Rust
                validation fails.
        """
        if not self._words:
            raise ValueError("no words defined; call add_word before build")
        if not self._zones:
            raise ValueError("no zones defined; call add_zone before build")

        words = tuple(Word(tuple(positions)) for positions in self._words)

        rust_zones = [
            _RustZone(
                name=z.name,
                grid=z.grid,
                site_buses=z.site_buses,
                word_buses=z.word_buses,
                words_with_site_buses=(
                    list(z.words_with_site_buses) if z.site_buses else []
                ),
                sites_with_word_buses=(
                    list(z.sites_with_word_buses) if z.word_buses else []
                ),
                entangling_pairs=list(z.entangling_pairs),
            )
            for z in self._zones
        ]

        zone_buses = [
            _RustZoneBus(
                src=[(src_zid, w) for w in src_words],
                dst=[(dst_zid, w) for w in dst_words],
            )
            for (src_zid, src_words), (dst_zid, dst_words) in self._connections
        ]

        modes: list[_RustMode] = []
        for mode_name, zone_names, order in self._modes:
            zone_ids = [self._zone_ids[z] for z in zone_names]
            modes.append(
                _RustMode(
                    name=mode_name,
                    zones=zone_ids,
                    bitstring_order=(
                        list(order)
                        if order is not None
                        else [
                            _RustLocAddr(zid, w, s)
                            for zid in zone_ids
                            for w in range(self.num_words)
                            for s in range(self.sites_per_word)
                        ]
                    ),
                )
            )

        # Reuse inherited paths when nothing about the buses has changed.
        # Bundled architectures ship hardware-derived lane geometry that
        # this search does not reproduce, and move durations are measured
        # off those segment lengths, so recomputing silently would shift
        # fidelity estimates and routing metrics.
        paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
        for zone_id, z in enumerate(self._zones):
            shape = self._inherited_shape.get(z.name)
            unchanged = shape is not None and shape == self._zone_shape(z)
            if not recompute_paths and unchanged:
                # Nothing about this zone's buses moved, so its inherited
                # lanes still describe it exactly; no search, and no
                # clearances needed.
                paths.update(self._reusable_inherited(zone_id, z))
                continue
            computed = compute_transport_paths(
                self._routing_geometry(z),
                zone=rust_zones[zone_id],
                zone_id=zone_id,
                warn_stacklevel=3,
            )
            if not recompute_paths:
                # Buses that existed at from_spec keep their inherited
                # geometry; only the appended ones take search results.
                computed.update(self._reusable_inherited(zone_id, z))
            paths.update(computed)

        return ArchSpec.from_components(
            words=words,
            zones=tuple(rust_zones),
            modes=modes,
            zone_buses=zone_buses,
            paths=paths or None,
            feed_forward=(self._feed_forward if feed_forward is None else feed_forward),
            atom_reloading=(
                self._atom_reloading if atom_reloading is None else atom_reloading
            ),
            blockade_radius=(
                blockade_radius
                if blockade_radius is not None
                else self._blockade_radius
            ),
        )

    # ── from_spec ──

    @classmethod
    def from_spec(
        cls,
        spec: ArchSpec,
        *,
        x_clearance: float | None = None,
        y_clearance: float | None = None,
    ) -> ArchBuilder:
        """Rebuild a builder from an existing :class:`ArchSpec`.

        Every zone is replayed in phase order, so a spec that violates the
        builder's rules — a bus no AOD can perform, say — is rejected here
        rather than restored silently.

        The spec's transport paths are carried over and reused verbatim by
        :meth:`build`, per bus: appending a bus searches paths for that bus
        alone and leaves every pre-existing lane exactly as it was.  That
        matters: the bundled architectures ship hardware-derived lane
        geometry that this path search does not reproduce (no clearance
        value regenerates it), and move durations — hence fidelity
        estimates and the committed benchmark baselines — are read off
        those segment lengths.

        Clearances are **not** part of an ``ArchSpec``: they are inputs to
        the path search, not properties of the architecture, so a
        round-trip cannot recover them.  Supply them if you intend to edit
        the buses, which makes the inherited paths stale and forces a
        search; without them, such an edit raises rather than silently
        re-routing.  Passing them does not by itself discard the inherited
        paths — use ``build(recompute_paths=True)`` for that.

        The word template is restored **verbatim**, not replayed through
        :meth:`add_word`.  An ``ArchSpec`` word is an ordered list of grid
        positions — it may be listed in any order, need not be a Cartesian
        product, and words may differ in shape, since the only cross-word
        rule is an equal site count.  A site's ID is its index in that list,
        so regenerating it would renumber sites and re-map every bus
        endpoint and inherited lane that addresses them.  When the restored
        template has no single row-major shape, :attr:`word_shape` is
        ``None`` and the ``sites[...]`` query is unavailable; use
        :meth:`word_sites` and plain site indices instead.

        Two things ``ArchSpec`` permits and this builder still refuses,
        because both put two atoms in one place: a word listing the same
        grid position twice, and two words sharing a position.

        **Specs this builder cannot represent are rejected, not
        reinterpreted.**  ``ArchSpec`` is a more permissive format than this
        builder's model, so some valid specs have no equivalent here.  Each
        raises ``ValueError`` naming the problem rather than being restored
        as something subtly different, because the alternative is a silent
        change to word numbering or routing:

        * **A zone bus whose endpoint spans several zones.**  :meth:`connect`
          addresses one zone per side, while a ``Bus[ZonedWordRef]`` carries a
          zone ID per element.  Rust only asks that each *pair* cross a zone
          boundary, so a bus whose source entries name two different zones is
          valid and unrepresentable here.  This is a genuine expressiveness
          gap, not an oversight — an architecture that needs that shape needs
          a richer ``connect`` first.
        * **A zone bus with an empty endpoint**, which Rust also accepts.
        * **Zones that disagree on grid dimensions**, since the word template
          is spec-wide and indexes one shared index space.
        * **A bus with no participants** — an empty ``words_with_site_buses``
          alongside site buses, or the same for word buses — which would carry
          no atoms.

        Args:
            spec: The spec to rebuild.
            x_clearance: Minimum x-axis waypoint clearance (µm), or
                ``None`` to keep the spec's own paths.
            y_clearance: Same, for the y-axis.

        Raises:
            ValueError: If the spec has no words or zones, if it uses any of
                the shapes listed above, or if any bus is not realizable
                under the builder's AOD rules.
        """
        inner = spec._inner
        if not inner.words:
            raise ValueError("spec has no words")
        if not inner.zones:
            raise ValueError("spec has no zones")

        # The template is restored verbatim rather than replayed through
        # add_word: ArchSpec stores a word as an ordered list of grid
        # positions, which is more general than (rows x columns), and a
        # site's ID is its index in that list — the integer every bus
        # endpoint and inherited lane address points at. Regenerating the
        # list from axis values would renumber any word not already listed
        # row-major and so silently re-map routing.
        words = [[(int(s[0]), int(s[1])) for s in w.sites] for w in inner.words]
        word_shape = _uniform_row_major_shape(words)

        shapes = {(z.grid.num_y, z.grid.num_x) for z in inner.zones}
        if len(shapes) > 1:
            raise ValueError(
                f"zones disagree on grid dimensions (rows, columns): "
                f"{sorted(shapes)}. The word "
                "template indexes one shared index space, so every zone's grid "
                "must have the same dimensions."
            )
        grid_shape = shapes.pop()

        # word_shape may be None here; the constructor needs *a* shape, and
        # the real template overwrites it immediately.
        builder = cls(
            grid_shape=grid_shape,
            word_shape=word_shape or (1, len(words[0])),
        )
        builder._word_shape = word_shape
        builder._words = [list(sites) for sites in words]

        # Rust validates site indices and equal site counts, but not these
        # two, so the builder keeps them: both describe an architecture with
        # two atoms in one place.
        for word_id, sites in enumerate(words):
            seen: set[tuple[int, int]] = set()
            for pos in sites:
                if pos in seen:
                    raise ValueError(
                        f"word {word_id} lists grid position "
                        f"(row={pos[1]}, column={pos[0]}) more than once"
                    )
                seen.add(pos)
                owner = builder._position_to_word.get(pos)
                if owner is not None:
                    raise ValueError(
                        f"words {owner} and {word_id} both occupy grid "
                        f"position (row={pos[1]}, column={pos[0]})"
                    )
                builder._position_to_word[pos] = word_id

        for zone in inner.zones:
            builder.add_zone(
                zone.name,
                rows=list(zone.grid.y_positions),
                columns=list(zone.grid.x_positions),
                x_clearance=x_clearance,
                y_clearance=y_clearance,
                # Participation is passed through verbatim: an empty list
                # alongside buses is a real value and must not be rebuilt as
                # "everything". With no buses of that kind, ``build`` emits
                # ``[]`` whatever the builder holds, so the spec's empty list
                # carries no information and defaulting to all keeps the zone
                # extensible.
                words_with_site_buses=(
                    list(zone.words_with_site_buses)
                    if zone.site_buses
                    else list(range(len(inner.words)))
                ),
                sites_with_word_buses=(
                    list(zone.sites_with_word_buses)
                    if zone.word_buses
                    else list(range(len(words[0])))
                ),
            )

        for zone in inner.zones:
            for bus in zone.site_buses:
                builder.add_site_bus(zone.name, list(bus.src), list(bus.dst))
            for bus in zone.word_buses:
                builder.add_word_bus(zone.name, bus.src, bus.dst, site_map=bus.site_map)
            if zone.entangling_pairs:
                pairs = [tuple(p) for p in zone.entangling_pairs]
                builder.add_entangling_pairs(
                    zone.name, [a for a, _ in pairs], [b for _, b in pairs]
                )

        names = [z.name for z in inner.zones]
        for bus_id, bus in enumerate(inner.zone_buses):
            endpoints = []
            for side, entries in (("src", bus.src), ("dst", bus.dst)):
                pairs = [tuple(e) for e in entries]
                if not pairs:
                    raise ValueError(
                        f"zone bus {bus_id} has an empty {side} endpoint; a bus "
                        "must move at least one word."
                    )
                zone_ids = {z for z, _ in pairs}
                if len(zone_ids) != 1:
                    raise ValueError(
                        f"zone bus {bus_id}'s {side} endpoint spans zones "
                        f"{sorted(zone_ids)}. connect addresses one zone per "
                        "side, so this builder cannot express it."
                    )
                endpoints.append((names[zone_ids.pop()], [w for _, w in pairs]))
            builder.connect(endpoints[0], endpoints[1])

        for mode in inner.modes:
            builder.add_mode(
                mode.name,
                [names[z] for z in mode.zones],
                bitstring_order=list(mode.bitstring_order),
            )

        builder._inherited_paths = dict(spec.paths) if spec.paths else None
        builder._inherited_shape = {
            z.name: builder._zone_shape(z) for z in builder._zones
        }
        builder._feed_forward = spec.feed_forward
        builder._atom_reloading = spec.atom_reloading
        builder._blockade_radius = spec.blockade_radius
        if spec.blockade_radius is not None:
            radius_nm = to_nm(spec.blockade_radius, "blockade_radius")
            for z in builder._zones:
                z.blockade_radius_nm = radius_nm
        return builder

    # ── Internals ──

    def _zone(self, name: str) -> _Zone:
        """Look up a zone by name."""
        if name not in self._zone_ids:
            raise ValueError(f"unknown zone: '{name}'")
        return self._zones[self._zone_ids[name]]

    def _geometry(self, zone: _Zone) -> ZoneGeometry:
        """Resolve the shared template against one zone's coordinates.

        Bus validation and the blockade scan read only positions, so a
        zone without clearances still resolves; :meth:`_routing_geometry`
        is the accessor that insists on them.
        """
        return ZoneGeometry(
            name=zone.name,
            grid_x_nm=zone.grid_x_nm,
            grid_y_nm=zone.grid_y_nm,
            x_clearance_nm=zone.x_clearance_nm or 0,
            y_clearance_nm=zone.y_clearance_nm or 0,
            words=tuple(tuple(w) for w in self._words),
        )

    def _routing_geometry(self, zone: _Zone) -> ZoneGeometry:
        """Like :meth:`_geometry`, but require clearances for path search."""
        if zone.x_clearance_nm is None or zone.y_clearance_nm is None:
            raise ValueError(
                f"zone '{zone.name}' has no clearances, so transport paths "
                "cannot be searched for it. It came from a spec whose paths "
                "were inherited; supply x_clearance/y_clearance to from_spec "
                "to recompute them."
            )
        return self._geometry(zone)

    def _zone_shape(self, zone: _Zone) -> tuple[int, int, tuple, tuple]:
        """Bus counts and participation, the things inherited lanes depend on."""
        return (
            len(zone.site_buses),
            len(zone.word_buses),
            zone.words_with_site_buses,
            zone.sites_with_word_buses,
        )

    def _reusable_inherited(
        self, zone_id: int, zone: _Zone
    ) -> dict[LaneAddress, tuple[tuple[float, float], ...]]:
        """Inherited lanes still valid for this zone.

        Buses are only ever appended, so a bus that existed at
        ``from_spec`` keeps its ID and its lanes stay meaningful.  What
        invalidates them is a change of participation: ``words_with_site_buses``
        decides which words get site lanes, ``sites_with_word_buses`` which
        sites get word lanes.
        """
        shape = self._inherited_shape.get(zone.name)
        if shape is None or self._inherited_paths is None:
            return {}
        n_site, n_word, movers, carried = shape
        site_ok = movers == zone.words_with_site_buses
        word_ok = carried == zone.sites_with_word_buses
        site_cap = min(n_site, len(zone.site_buses))
        word_cap = min(n_word, len(zone.word_buses))
        out: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
        for lane, path in self._inherited_paths.items():
            if lane.zone_id != zone_id:
                continue
            if lane.move_type == MoveType.SITE:
                if site_ok and lane.bus_id < site_cap:
                    out[lane] = path
            elif word_ok and lane.bus_id < word_cap:
                out[lane] = path
        return out

    def _check_bus_endpoints(
        self,
        src: Sequence[int],
        dst: Sequence[int],
        size: int,
        kind: str,
    ) -> None:
        """Shared endpoint checks: equal length, non-empty, in range."""
        if len(src) != len(dst):
            raise ValueError(
                f"{kind} bus src has {len(src)} entries but dst has {len(dst)}"
            )
        if not src:
            raise ValueError(
                f"a {kind} bus must move at least one {kind}; an empty bus "
                "would reach the spec as a bus that transports nothing"
            )
        for label, seq in (("src", src), ("dst", dst)):
            bad = sorted({i for i in seq if not 0 <= i < size})
            if bad:
                raise IndexError(
                    f"{kind} bus {label} {kind} index {bad} out of range "
                    f"[0, {size})"
                )


__all__ = ["ArchBuilder"]
