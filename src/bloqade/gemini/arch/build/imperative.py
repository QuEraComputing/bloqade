"""Zone-aware architecture builder.

Provides ``ZoneBuilder`` (single-zone construction with validation) and
``ArchBuilder`` (multi-zone composition) as the low-level building blocks.
The high-level ``build_arch()`` function uses these internally.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bloqade.lanes.bytecode._native import (
    Grid as _RustGrid,
    LocationAddress as _RustLocAddr,
    Mode as _RustMode,
    SiteBus as _RustSiteBus,
    WordBus as _RustWordBus,
    Zone as _RustZone,
    ZoneBus as _RustZoneBus,
)
from bloqade.gemini.arch.build._geometry import (
    NM_PER_UM as _NM_PER_UM,
    ZoneGeometry,
    compute_transport_paths,
    enumerate_safe_positions,
    scan_blockade_pairs,
    search_path,
    to_nm as _to_nm,
)
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress
from bloqade.lanes.bytecode.word import Word

# ── Helpers ──

# Unit conversion, AOD path search and blockade-pair derivation live in
# :mod:`bloqade.gemini.arch.build._geometry`, shared with the redesigned
# builder in :mod:`bloqade.gemini.arch.build.v2`.  This module keeps the
# imperative assembly API and delegates the geometry.


def _normalize_index(idx: slice | int | Sequence[int], size: int) -> list[int]:
    """Convert a slice, int, or list to a sorted list of indices."""
    if isinstance(idx, slice):
        return list(range(*idx.indices(size)))
    if isinstance(idx, int):
        return [idx]
    return sorted(idx)


def _query_axis(
    idx: slice | int | Sequence[int], size: int, axis: str, what: str
) -> list[int]:
    """Normalize one axis of a grid *query*, rejecting bad indices.

    The selection is returned **in the order given** — a sequence verbatim,
    a slice as it expands (including a negative step).  The query as a whole
    then behaves like ``arr[np.ix_(y_values, x_values)].ravel()``: the axis
    order is the caller's, and "row-major" describes only the nesting, with
    y as the outer loop.

    Indices must be unique.  Repeats cannot add anything — a word bus moves
    each position once — and since duplicates would otherwise be dropped by
    the first-unique rule, rejecting them keeps a typo from silently
    shortening the result.

    Out-of-range and negative indices also raise rather than selecting
    nothing, since queries resolve positions through a dict lookup: without
    this a ``src``/``dst`` pair shortened by the same amount would still
    satisfy ``add_word_bus``'s equal-length check.  ``add_word`` and
    ``add_word_bus`` already reject such indices; this makes the read side
    agree.  Slices are clamped by ``slice.indices`` and are always valid.
    """
    if isinstance(idx, slice):
        return list(range(*idx.indices(size)))
    values = [idx] if isinstance(idx, int) else list(idx)
    out_of_range = sorted({i for i in values if not 0 <= i < size})
    if out_of_range:
        raise IndexError(f"{axis} {what} {out_of_range} out of range [0, {size})")
    counts = Counter(values)
    repeated = sorted(i for i, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(
            f"{axis} {what} {repeated} repeated in {values}; each position is "
            "selected once, so a repeat cannot select anything new"
        )
    return values


def _reject_duplicate_endpoints(
    src: Sequence[int], dst: Sequence[int], kind: str, zone_name: str, unit: str
) -> None:
    """Reject a bus that names the same endpoint twice.

    The geometry checks downstream collapse endpoints to sets — a
    Cartesian product, a tone count, a rank map — so a repeated index
    survives all of them and the malformed bus is only caught by Rust at
    ``build()``.  It is malformed for two independent reasons: a repeated
    source shadows every later pair under the positional first-match
    resolution the bus relation uses, and a repeated destination asks for
    two simultaneous transports into one place.
    """
    for side, values in (("src", src), ("dst", dst)):
        repeated = sorted(i for i, n in Counter(values).items() if n > 1)
        if repeated:
            raise ValueError(
                f"{kind} bus on zone '{zone_name}': {side} {unit} "
                f"{repeated} repeated in {list(values)}; a bus moves each "
                f"{unit} once"
            )


def _validate_aod_rectangle(
    positions: list[tuple[int, int]],
    label: str,
) -> None:
    """Validate that positions form a Cartesian product on the grid.

    The set of (x, y) positions must equal {x_values} × {y_values}
    for some sets x_values and y_values. This ensures the bus defines
    a complete rectangular AOD grid.
    """
    if not positions:
        return
    pos_set = set(positions)
    xs = sorted({p[0] for p in positions})
    ys = sorted({p[1] for p in positions})
    expected = {(x, y) for x in xs for y in ys}
    if pos_set != expected:
        missing = expected - pos_set
        raise ValueError(
            f"{label} positions do not form a valid AOD Cartesian product. "
            f"Missing positions: {sorted(missing)}"
        )


def _require_complete_tone_grid(
    positions: list[tuple[int, int]], prefix: str, side: str
) -> None:
    """Require a bus's atoms to fill its tone grid exactly.

    ``_validate_aod_rectangle`` answers the same question for grid indices;
    this one works on physical nm positions and reports in µm, because the
    atoms a bus carries can form an incomplete rectangle even when the word
    origins or site-template indices behind them do not.

    Completeness is what keeps a foreign atom out: the AOD traps at every
    intersection of its tones, so an unoccupied intersection is one a
    non-participating atom could be sitting on, and it would be carried
    along with the bus.
    """
    if not positions:
        return
    xs = sorted({x for x, _ in positions})
    ys = sorted({y for _, y in positions})
    missing = {(x, y) for x in xs for y in ys} - set(positions)
    if missing:
        listed = sorted((x / _NM_PER_UM, y / _NM_PER_UM) for x, y in missing)
        raise ValueError(
            f"{prefix}: the {side} atoms span a {len(xs)}x{len(ys)} grid of "
            f"tones but do not fill it — {listed} µm "
            f"{'is' if len(listed) == 1 else 'are'} unoccupied by this bus. "
            "An AOD traps at every intersection of its tones, so any atom "
            "sitting there would be carried along too."
        )


# ── Grid query helpers ──


class _SiteGridQuery:
    """Query site indices within a word shape.

    Traversal is row-major over the selected positions — y is the outer
    loop, x the inner — with each axis visited in the order given, exactly
    like ``arr[np.ix_(y_values, x_values)].ravel()``.  Site index is
    ``x + y * num_x``.
    """

    def __init__(self, word_shape: tuple[int, int]):
        self._nx, self._ny = word_shape

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> list[int]:
        x_idx, y_idx = key
        xs = _query_axis(x_idx, self._nx, "x", "site index")
        ys = _query_axis(y_idx, self._ny, "y", "site index")
        return [x + y * self._nx for y in ys for x in xs]


class _WordGridQuery:
    """Query word indices by grid region on a ZoneBuilder.

    The selected positions are traversed row-major — y the outer loop, x
    the inner, each axis in the order given::

        for y in y_values:
            for x in x_values:
                ...  # append the word here, if not already appended

    which is the same rule as ``arr[np.ix_(y_values, x_values)].ravel()``:
    "row-major" describes the nesting, while the order within each axis is
    the caller's.  A word spans several grid positions and so is reached
    more than once; taking the first arrival is what makes its place in the
    result well defined.

    Ordering by traversal rather than by word ID matters whenever word IDs
    are not monotone in the grid, which is the normal case for interleaved
    CZ layouts: scanning x visits words 0, 1, 2, 3, 0, 1, 2, 3, ..., so a
    partial x-selection reaches them out of ID order.  An AOD transport is
    separable and order-preserving — each x-tone and y-tone sweeps
    independently and tones cannot cross — so it maps the source tones onto
    the destination tones in the same order.  Selecting both endpoints the
    same way therefore pairs ``src[i]`` with ``dst[i]`` correctly, including
    for a compression or expansion, where the displacement differs per
    column and no single translation describes the bus.

    Returns a plain ``list[int]`` of zone-local word IDs for intra-zone
    operations like ``add_word_bus`` and ``add_entangling_pairs``.  For
    cross-zone operations that need a name-qualified reference (e.g.,
    ``ArchBuilder.connect``), use ``zone[...]`` directly — that form
    returns ``(name, list[int])``.
    """

    def __init__(self, zone: ZoneBuilder):
        self._zone = zone

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> list[int]:
        x_idx, y_idx = key
        xs = _query_axis(x_idx, self._zone._grid.num_x, "x", "grid index")
        ys = _query_axis(y_idx, self._zone._grid.num_y, "y", "grid index")
        result: list[int] = []
        seen: set[int] = set()
        for y in ys:
            for x in xs:
                word_id = self._zone._position_to_word.get((x, y))
                if word_id is not None and word_id not in seen:
                    seen.add(word_id)
                    result.append(word_id)
        return result


# ── ZoneBuilder ──


class ZoneBuilder:
    """Build a single zone with its words, grid, and buses.

    All indices are zone-local. Words are placed on the zone's grid
    and validated for shape and overlap. Buses are validated for AOD
    Cartesian product compliance.
    """

    def __init__(
        self,
        name: str,
        grid: _RustGrid,
        word_shape: tuple[int, int],
        *,
        x_clearance: float,
        y_clearance: float,
    ):
        """Initialize a zone.

        Args:
            name: Human-readable zone name (stored in Rust Zone).
            grid: Coordinate grid for this zone.  Every x- and y-position
                must be representable at 1 nm precision (i.e., at most 3
                decimal places when given in µm).
            word_shape: (num_x_sites, num_y_sites) — uniform shape for
                all words in this zone. sites_per_word = product of shape.
            x_clearance: Minimum physical distance (> 0, µm) from grid
                lines that path waypoints must maintain on the x-axis.
                Must be representable at 1 nm precision.
            y_clearance: Same as ``x_clearance``, applied to the y-axis.
                Allowing separate values is useful when row and column
                spacings differ substantially (e.g., tight intra-pair x
                spacing but wide row spacing).

        Raises:
            ValueError: If either clearance is not positive or any
                position / clearance value is not nm-precise.
        """
        if x_clearance <= 0:
            raise ValueError(f"x_clearance must be positive, got {x_clearance}")
        if y_clearance <= 0:
            raise ValueError(f"y_clearance must be positive, got {y_clearance}")
        self._name = name
        self._grid = grid
        self._word_shape = word_shape
        # Internal nm-integer representation for exact path search.
        self._x_clearance_nm = _to_nm(x_clearance, "x_clearance")
        self._y_clearance_nm = _to_nm(y_clearance, "y_clearance")
        self._grid_x_nm: list[int] = [
            _to_nm(x, "grid x-position") for x in grid.x_positions
        ]
        self._grid_y_nm: list[int] = [
            _to_nm(y, "grid y-position") for y in grid.y_positions
        ]
        self._words: list[list[tuple[int, int]]] = []
        self._word_has_site_bus: list[bool] = []
        self._position_to_word: dict[tuple[int, int], int] = {}
        self._site_buses: list[tuple[list[int], list[int]]] = []
        self._word_buses: list[tuple[list[int], list[int]]] = []
        self._entangling_pairs: list[tuple[int, int]] = []
        self._blockade_radius_nm: int | None = None

    @classmethod
    def from_positions(
        cls,
        name: str,
        x_coordinates: Sequence[float | int],
        y_coordinates: Sequence[float | int],
        word_shape: tuple[int, int],
        *,
        x_clearance: float,
        y_clearance: float,
    ) -> ZoneBuilder:
        """Build a zone from explicit x/y coordinate arrays.

        Convenience constructor that builds the coordinate grid from
        ``x_coordinates`` and ``y_coordinates`` (via ``Grid.from_positions``)
        and forwards the remaining arguments to the default constructor.

        Args:
            name: Human-readable zone name (stored in Rust Zone).
            x_coordinates: X-coordinates of the grid points (at least one).
                Every position must be representable at 1 nm precision.
            y_coordinates: Y-coordinates of the grid points (at least one).
            word_shape: (num_x_sites, num_y_sites) — uniform shape for
                all words in this zone.
            x_clearance: Minimum x-axis clearance (> 0, µm) from grid lines
                that path waypoints must maintain.
            y_clearance: Same as ``x_clearance``, applied to the y-axis.

        Returns:
            ZoneBuilder: The constructed zone.
        """
        grid = _RustGrid.from_positions(list(x_coordinates), list(y_coordinates))
        return cls(
            name,
            grid,
            word_shape,
            x_clearance=x_clearance,
            y_clearance=y_clearance,
        )

    @property
    def name(self) -> str:
        """Zone name."""
        return self._name

    @property
    def word_shape(self) -> tuple[int, int]:
        """(num_x_sites, num_y_sites) for each word."""
        return self._word_shape

    @property
    def sites_per_word(self) -> int:
        """Total sites per word (product of word_shape)."""
        return self._word_shape[0] * self._word_shape[1]

    @property
    def x_clearance(self) -> float:
        """Minimum x-axis clearance (µm) from grid lines for waypoints."""
        return self._x_clearance_nm / _NM_PER_UM

    @property
    def y_clearance(self) -> float:
        """Minimum y-axis clearance (µm) from grid lines for waypoints."""
        return self._y_clearance_nm / _NM_PER_UM

    @property
    def num_words(self) -> int:
        """Number of words added so far."""
        return len(self._words)

    def add_word(
        self,
        x_sites: slice | Sequence[int],
        y_sites: slice | Sequence[int],
        *,
        has_site_bus: bool = True,
    ) -> int:
        """Add a word occupying the given grid positions.

        The number of x-indices and y-indices must match word_shape.
        Grid positions must not overlap with any existing word.

        Args:
            x_sites: Grid x-indices for the word's sites.
            y_sites: Grid y-indices for the word's sites.
            has_site_bus: Whether this word participates in site-bus
                transport. Feeds the zone-level
                ``words_with_site_buses`` list on the final ``ArchSpec``
                — only words with ``has_site_bus=True`` are eligible to
                have site buses applied to them. Defaults to ``True``,
                which preserves the historical "all words opt-in"
                behavior. Set to ``False`` on storage words that
                shouldn't participate in site-level routing.

        Returns:
            Zone-local word index.

        Raises:
            ValueError: If the index counts do not match ``word_shape``, an
                index is repeated (two sites would share one grid
                position), or a position already belongs to another word.
            IndexError: If an index is out of range for this zone's grid.
        """
        xs = _normalize_index(x_sites, self._grid.num_x)
        ys = _normalize_index(y_sites, self._grid.num_y)

        if len(xs) != self._word_shape[0]:
            raise ValueError(
                f"x_sites has {len(xs)} indices but word_shape requires "
                f"{self._word_shape[0]}"
            )
        if len(ys) != self._word_shape[1]:
            raise ValueError(
                f"y_sites has {len(ys)} indices but word_shape requires "
                f"{self._word_shape[1]}"
            )

        # A repeated index satisfies the count check above while collapsing
        # two sites onto one grid position, producing a word that claims
        # sites_per_word atoms but occupies fewer — the overlap check below
        # cannot see it, since it runs before any of them is recorded.
        for axis, values in (("x_sites", xs), ("y_sites", ys)):
            repeated = sorted(i for i, n in Counter(values).items() if n > 1)
            if repeated:
                raise ValueError(
                    f"{axis} {repeated} repeated in {values}; each site of a "
                    "word must occupy its own grid position"
                )

        for x in xs:
            if x < 0 or x >= self._grid.num_x:
                raise IndexError(
                    f"x index {x} out of range for grid with "
                    f"{self._grid.num_x} x-positions"
                )
        for y in ys:
            if y < 0 or y >= self._grid.num_y:
                raise IndexError(
                    f"y index {y} out of range for grid with "
                    f"{self._grid.num_y} y-positions"
                )

        positions = [(x, y) for y in ys for x in xs]
        for pos in positions:
            if pos in self._position_to_word:
                owner = self._position_to_word[pos]
                raise ValueError(
                    f"Grid position (x={pos[0]}, y={pos[1]}) "
                    f"already belongs to word {owner}"
                )

        word_id = len(self._words)
        self._words.append(positions)
        self._word_has_site_bus.append(has_site_bus)
        for pos in positions:
            self._position_to_word[pos] = word_id
        return word_id

    def add_site_bus(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a site bus (intra-word movement).

        ``src[i]`` moves to ``dst[i]``; both are site indices within
        ``word_shape`` (``0..sites_per_word``).

        Args:
            src: Source site indices. Must be non-empty, in range, and form
                an AOD Cartesian product on the word grid.
            dst: Destination site indices, same length as ``src`` and
                likewise a Cartesian product.

        Raises:
            ValueError: If the two differ in length, either is empty, an
                index is repeated, an index is out of range, or either
                endpoint is not a Cartesian product.

        Note:
            Unlike :meth:`add_word_bus`, this does not check that the bus is
            AOD-realizable.  The atoms it carries are
            ``words_with_site_buses x src``, and :meth:`add_word` can extend
            that set afterwards, so the set is not known here.
        """
        if len(src) != len(dst):
            raise ValueError(
                f"Site bus src has {len(src)} entries but dst has {len(dst)}"
            )
        if not src:
            # An empty bus carries nothing, and the path search indexes
            # src[0] for its reference atom.
            raise ValueError(
                f"Site bus on zone '{self._name}' is empty; a bus must move "
                "at least one site"
            )
        total = self.sites_per_word
        nx = self._word_shape[0]
        for s in src:
            if s < 0 or s >= total:
                raise ValueError(f"site index {s} out of range [0, {total})")
        for d in dst:
            if d < 0 or d >= total:
                raise ValueError(f"site index {d} out of range [0, {total})")

        _reject_duplicate_endpoints(src, dst, "Site", self._name, "site")

        src_positions = [(s % nx, s // nx) for s in src]
        dst_positions = [(d % nx, d // nx) for d in dst]
        _validate_aod_rectangle(src_positions, "Site bus src")
        _validate_aod_rectangle(dst_positions, "Site bus dst")
        # No AOD-realizability check here.  A site bus carries
        # ``words_with_site_buses x src``, and ``add_word`` can extend that
        # set after the fact, so the carried-atom set is not known at this
        # call — the question is undecidable at add time.  ``_compute_paths``
        # still declines to route a bus it cannot realize.
        self._site_buses.append((list(src), list(dst)))

    def _site_bus_words(self) -> list[int]:
        """Words that take part in site-bus transport.

        Only these appear in the zone's ``words_with_site_buses``, so only
        these are carried when a site bus fires.
        """
        return [w for w in range(self.num_words) if self._word_has_site_bus[w]]

    def add_word_bus(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a word bus (intra-zone movement).

        ``src[i]`` moves to ``dst[i]``; both are zone-local word indices.

        Args:
            src: Source word indices. Must be non-empty, in range, and form
                an AOD Cartesian product on the zone grid.
            dst: Destination word indices, same length as ``src`` and
                likewise a Cartesian product.

        Raises:
            ValueError: If the two differ in length, either is empty, an
                index is repeated, an index is out of range, either
                endpoint is not a Cartesian product, or the transport is
                not AOD-realizable — see
                :meth:`_check_aod_compatible`.  Unlike a site bus, a word
                bus's carried atoms are fixed at this call: it names its
                words explicitly and ``_words`` is append-only.
        """
        if len(src) != len(dst):
            raise ValueError(
                f"Word bus src has {len(src)} entries but dst has {len(dst)}"
            )
        if not src:
            # An empty bus carries nothing, and the path search indexes
            # src_words[0] for its reference atom.
            raise ValueError(
                f"Word bus on zone '{self._name}' is empty; a bus must move "
                "at least one word"
            )
        n = len(self._words)
        for s in src:
            if s < 0 or s >= n:
                raise ValueError(f"word index {s} out of range [0, {n})")
        for d in dst:
            if d < 0 or d >= n:
                raise ValueError(f"word index {d} out of range [0, {n})")

        _reject_duplicate_endpoints(src, dst, "Word", self._name, "word")

        src_positions = [self._word_origin(s) for s in src]
        dst_positions = [self._word_origin(d) for d in dst]
        _validate_aod_rectangle(src_positions, "Word bus src")
        _validate_aod_rectangle(dst_positions, "Word bus dst")
        # Every site of every src word moves to the matching site of its dst
        # word; check that whole atom set is AOD-realizable — separable and
        # order-preserving, which permits per-column displacement.  All sites
        # are used rather than sampling site 0, so a zone whose words differ
        # in internal geometry is caught too.
        self._check_aod_compatible(
            [
                (self._site_nm(sw, s), self._site_nm(dw, s))
                for sw, dw in zip(src, dst)
                for s in range(self.sites_per_word)
            ],
            "Word",
            len(self._word_buses),
        )
        self._word_buses.append((list(src), list(dst)))

    def _check_aod_compatible(
        self,
        pairs: list[tuple[tuple[int, int], tuple[int, int]]],
        kind: str,
        bus_id: int,
    ) -> None:
        """Reject a bus no AOD operation can perform.

        An AOD is a crossed pair of deflectors: the traps sit at the
        Cartesian product of its x-tones and y-tones, and each tone is
        swept independently.  A transport is therefore realizable exactly
        when it is **separable and order-preserving**:

        * every atom's x-displacement depends only on which column it is
          in, and its y-displacement only on its row — an atom cannot leave
          its own tone;
        * tones keep their relative order, since two frequencies cannot
          pass through each other.

        Note this is weaker than a uniform translation: compressing or
        expanding a rectangle gives each column a different displacement
        and is still one AOD operation.  ``_compute_paths`` is the stricter
        party — it searches one reference path and applies its deltas to
        every lane — so it still declines to route a non-uniform bus.  That
        is a limit of the path search, not of the hardware.

        Args:
            pairs: ``(src_position, dst_position)`` in nm, one entry per
                atom the bus carries.
            kind: ``"Word"`` or ``"Site"``, for the error message.
            bus_id: Index the bus would receive, for the error message.
        """
        if not pairs:
            return
        # The AOD traps at the Cartesian product of its tones, so every atom
        # sitting at an intersection is carried whether or not it belongs to
        # this bus.  Requiring the bus's own atoms to fill that product
        # exactly is what guarantees no foreign atom rides along: with every
        # intersection occupied by a bus atom, none is left for an outsider.
        # The earlier ``_validate_aod_rectangle`` calls see only word origins
        # or site-template indices, so an L-shaped set of participating words
        # reaches here intact.
        prefix = f"{kind} bus {bus_id} on zone '{self._name}'"
        _require_complete_tone_grid([p[0] for p in pairs], prefix, "source")
        _require_complete_tone_grid([p[1] for p in pairs], prefix, "destination")
        src_x = sorted({p[0][0] for p in pairs})
        src_y = sorted({p[0][1] for p in pairs})
        dst_x = sorted({p[1][0] for p in pairs})
        dst_y = sorted({p[1][1] for p in pairs})
        if len(src_x) != len(dst_x) or len(src_y) != len(dst_y):
            raise ValueError(
                f"{prefix}: the bus spans a {len(src_x)}x{len(src_y)} grid of "
                f"tones at its source but {len(dst_x)}x{len(dst_y)} at its "
                "destination. An AOD cannot add or drop a tone mid-transport."
            )
        column = {v: i for i, v in enumerate(src_x)}
        row = {v: i for i, v in enumerate(src_y)}
        for (sx, sy), (dx, dy) in pairs:
            want = (dst_x[column[sx]], dst_y[row[sy]])
            if (dx, dy) != want:
                raise ValueError(
                    f"{prefix}: the atom at "
                    f"{(sx / _NM_PER_UM, sy / _NM_PER_UM)} µm is sent to "
                    f"{(dx / _NM_PER_UM, dy / _NM_PER_UM)} µm, but its column "
                    f"and row land at {(want[0] / _NM_PER_UM, want[1] / _NM_PER_UM)}"
                    " µm. An AOD sweeps whole x- and y-tones, which cannot "
                    "cross or trade places, so every atom must stay on its own."
                )

    def add_entangling_pairs(
        self, words_a: Sequence[int], words_b: Sequence[int]
    ) -> None:
        """Mark paired zone-local words as CZ pairs.

        ``words_a[i]`` is paired with ``words_b[i]``. The two sequences
        must have the same length.

        For most users, prefer :meth:`set_blockade_radius` — it derives
        the pair list directly from geometry and validates the word
        layout against the CZ-pairing convention.

        Any ``blockade_radius`` previously recorded on this zone (via
        :meth:`set_blockade_radius`) is cleared, since a manual append
        means the pair list is no longer purely radius-derived.
        """
        if len(words_a) != len(words_b):
            raise ValueError(
                f"words_a has {len(words_a)} entries but words_b has " f"{len(words_b)}"
            )
        n = len(self._words)
        for a, b in zip(words_a, words_b):
            if a < 0 or a >= n:
                raise ValueError(f"word index {a} out of range [0, {n})")
            if b < 0 or b >= n:
                raise ValueError(f"word index {b} out of range [0, {n})")
            self._entangling_pairs.append((a, b))
        # Pair list was just manually modified; the cached radius no
        # longer describes its full state.
        self._blockade_radius_nm = None

    @property
    def blockade_radius(self) -> float | None:
        """Rydberg blockade radius (µm) used to derive entangling pairs, or None."""
        if self._blockade_radius_nm is None:
            return None
        return self._blockade_radius_nm / _NM_PER_UM

    def set_blockade_radius(self, radius: float) -> None:
        """Derive entangling word pairs from the Rydberg blockade radius.

        Scans every pair of distinct words in the zone and classifies
        each under the matching-site-index CZ convention:

        * All matching-index site distances ``<= radius`` and all
          non-matching-index site distances ``> radius``: valid CZ pair.
        * Some matching-index distances within radius, some outside:
          ``ValueError`` (partial blockade — the word layout doesn't
          cleanly map onto the CZ-pairing convention).
        * Any non-matching-index site distance within radius (regardless
          of whether the matching-index distances also fall within):
          ``ValueError`` (crossed-index — two words are arranged such
          that site ``i`` of one word sits next to site ``j != i`` of
          the other, violating the exclusivity the convention requires).
        * All distances outside radius: words ignore each other, no
          pair recorded.

        After classification, every word must appear in **at most one**
        valid pair; multiple partners raise ``ValueError``.

        This call **overwrites** ``_entangling_pairs`` with the scan
        result and stores the radius on the zone.  To have it flow into
        the final ``ArchSpec.blockade_radius``, either call
        :meth:`ArchBuilder.set_blockade_radius` (which applies to every
        zone and records the value at builder scope) or, for a single
        zone already set via ``ZoneBuilder.set_blockade_radius``,
        ``ArchBuilder.build()`` will pick up a consistent zone-level
        radius automatically.

        Args:
            radius: Blockade radius in micrometers. Must be positive and
                representable at 1 nm precision.

        Raises:
            ValueError: if the layout is inconsistent with the radius
                (partial blockade / crossed-index / multi-partner) or
                if ``radius`` is not positive / nm-precise.
        """
        if radius <= 0:
            raise ValueError(f"blockade_radius must be positive, got {radius}")
        radius_nm = _to_nm(radius, "blockade_radius")
        self._entangling_pairs = self._scan_blockade_pairs(radius_nm)
        self._blockade_radius_nm = radius_nm

    def _scan_blockade_pairs(self, radius_nm: int) -> list[tuple[int, int]]:
        """Scan word pairs and classify under the matching-index CZ rule.

        Delegates to :func:`._geometry.scan_blockade_pairs`; see that
        function for the classification rules and the errors it raises.
        """
        return scan_blockade_pairs(self._geometry(), radius_nm)

    @property
    def words(self) -> _WordGridQuery:
        """Query word indices by grid region for intra-zone use.

        Returns a plain ``list[int]`` of zone-local word indices whose
        sites intersect the queried region — suitable for passing
        directly to ``add_word_bus`` / ``add_entangling_pairs``.

        For cross-zone references (e.g. ``ArchBuilder.connect``), index
        the zone itself (``zone[region]``) to get a name-qualified
        ``(name, list[int])`` tuple.
        """
        return _WordGridQuery(self)

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> tuple[str, list[int]]:
        """Query word indices by grid region, name-qualified for cross-zone use.

        Returns ``(self.name, zone.words[key])`` so the result can be
        passed directly to ``ArchBuilder.connect(src=..., dst=...)``,
        which expects ``(zone_name, zone_local_indices)``.

        For intra-zone use (passing indices to ``add_word_bus`` etc.),
        use ``zone.words[key]`` which returns just the index list.
        """
        return (self._name, self.words[key])

    @property
    def sites(self) -> _SiteGridQuery:
        """Query site indices within the word shape."""
        return _SiteGridQuery(self._word_shape)

    def _word_origin(self, word_id: int) -> tuple[int, int]:
        """Get the (min_x, min_y) origin of a word for AOD validation."""
        positions = self._words[word_id]
        return (min(p[0] for p in positions), min(p[1] for p in positions))

    # ── Path computation ──
    #
    # The geometry itself lives in :mod:`._geometry`, shared with the
    # redesigned builder.  These methods resolve this zone into a
    # :class:`ZoneGeometry` and delegate; they stay on the class because
    # callers and tests reach for them directly.

    def _geometry(self) -> ZoneGeometry:
        """Resolve this zone's grid and word template into a geometry value."""
        return ZoneGeometry(
            name=self._name,
            grid_x_nm=tuple(self._grid_x_nm),
            grid_y_nm=tuple(self._grid_y_nm),
            x_clearance_nm=self._x_clearance_nm,
            y_clearance_nm=self._y_clearance_nm,
            words=tuple(tuple(w) for w in self._words),
        )

    def _site_nm(self, word_id: int, site_id: int) -> tuple[int, int]:
        """Physical (x, y) position of a site, in nm integers."""
        x_idx, y_idx = self._words[word_id][site_id]
        return (self._grid_x_nm[x_idx], self._grid_y_nm[y_idx])

    def _enumerate_safe_positions(
        self,
        grid_positions: list[int],
        source_positions: list[int],
        min_cl_nm: int,
    ) -> list[int]:
        """Delegate to :func:`._geometry.enumerate_safe_positions`."""
        return enumerate_safe_positions(grid_positions, source_positions, min_cl_nm)

    def _search_path(
        self,
        ref_src: tuple[int, int],
        ref_dst: tuple[int, int],
        bus_src_positions: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...] | None:
        """Delegate to :func:`._geometry.search_path`."""
        return search_path(self._geometry(), ref_src, ref_dst, bus_src_positions)

    def _compute_paths(
        self, zone_id: int
    ) -> dict[LaneAddress, tuple[tuple[float, float], ...]]:
        """Delegate to :func:`._geometry.compute_transport_paths`."""
        return compute_transport_paths(
            self._geometry(),
            site_buses=self._site_buses,
            word_buses=self._word_buses,
            site_bus_words=self._site_bus_words(),
            zone_id=zone_id,
            warn_stacklevel=4,
        )


# ── ArchBuilder ──


class ArchBuilder:
    """Compose ``ZoneBuilder``s into a complete ``ArchSpec``.

    All zones share ONE word template: ``word_id`` is zone-local
    (``0..Nw-1``) and reused identically by every zone, disambiguated by
    ``zone_id`` (the LocationAddress convention). Inter-zone connections go
    into ``zone_buses``. Calls Rust validation on ``build()``.
    """

    def __init__(self) -> None:
        self._zones: list[ZoneBuilder] = []
        self._zone_name_to_id: dict[str, int] = {}
        self._connections: list[
            tuple[tuple[str, Sequence[int]], tuple[str, Sequence[int]]]
        ] = []
        self._modes: list[tuple[str, list[str]]] = []
        self._blockade_radius: float | None = None

    def add_zone(self, zone: ZoneBuilder) -> int:
        """Add a zone. Returns zone_id.

        Validates that sites_per_word matches across all zones. The word
        template itself is validated for cross-zone identity at
        :meth:`build` time.
        """
        if zone.name in self._zone_name_to_id:
            raise ValueError(f"Duplicate zone name: '{zone.name}'")
        if self._zones:
            existing_spw = self._zones[0].sites_per_word
            if zone.sites_per_word != existing_spw:
                raise ValueError(
                    f"sites_per_word mismatch: zone '{zone.name}' has "
                    f"{zone.sites_per_word} but existing zones have "
                    f"{existing_spw}"
                )
        zone_id = len(self._zones)
        self._zone_name_to_id[zone.name] = zone_id
        self._zones.append(zone)
        return zone_id

    def connect(
        self,
        src: tuple[str, Sequence[int]],
        dst: tuple[str, Sequence[int]],
    ) -> None:
        """Add an inter-zone bus (zone_buses).

        Args:
            src: ``(zone_name, zone_local_word_indices)`` — typically
                from ``zone[...]`` (indexing the zone itself, which
                returns a name-qualified tuple).
            dst: ``(zone_name, zone_local_word_indices)`` — same format.

        Raises:
            ValueError: If either zone name is unknown or a word index is
                out of range for its zone.

        Validates AOD Cartesian product across the two zone grids.
        """
        src_name, src_words = src
        dst_name, dst_words = dst
        if src_name not in self._zone_name_to_id:
            raise ValueError(f"Unknown zone: '{src_name}'")
        if dst_name not in self._zone_name_to_id:
            raise ValueError(f"Unknown zone: '{dst_name}'")

        src_zone = self._zones[self._zone_name_to_id[src_name]]
        dst_zone = self._zones[self._zone_name_to_id[dst_name]]
        # Range-check before resolving positions.  ``_word_origin`` indexes
        # ``_words`` directly, so a negative index silently resolves to a
        # different word and the Cartesian-product check below would then
        # validate geometry the caller never named.  ``add_word_bus``
        # rejects the same index.
        for side, zone, words in (
            ("src", src_zone, src_words),
            ("dst", dst_zone, dst_words),
        ):
            for w in words:
                if w < 0 or w >= zone.num_words:
                    raise ValueError(
                        f"Zone bus {side} word index {w} out of range "
                        f"[0, {zone.num_words}) for zone '{zone.name}'"
                    )

        src_positions = [src_zone._word_origin(w) for w in src_words]
        dst_positions = [dst_zone._word_origin(w) for w in dst_words]
        _validate_aod_rectangle(src_positions, "Zone bus src")
        _validate_aod_rectangle(dst_positions, "Zone bus dst")

        self._connections.append((src, dst))

    def add_mode(self, name: str, zones: Sequence[str]) -> None:
        """Add an operational mode.

        Args:
            name: Mode name (e.g. "all", "gate", "measure").
            zones: Zone names to include in this mode.
        """
        for z in zones:
            if z not in self._zone_name_to_id:
                raise ValueError(f"Unknown zone: '{z}'")
        self._modes.append((name, list(zones)))

    def set_blockade_radius(self, radius: float) -> None:
        """Apply ``radius`` to every zone by calling
        :meth:`ZoneBuilder.set_blockade_radius` on each.

        Overwrites every zone's entangling pairs with the scan result.
        The radius is stored on the builder and flows to
        ``ArchSpec.blockade_radius`` at :meth:`build` time.

        The radius is validated up-front (positive, finite, nm-precise)
        *before* any zone is touched, and the scan is run two-phase:
        every zone is scanned before any pair list is overwritten, so
        a layout error in a later zone cannot leave earlier zones in a
        partially-updated state.

        Args:
            radius: Rydberg blockade radius in micrometers.

        Raises:
            ValueError: if ``radius`` itself is invalid (non-positive,
                non-finite, sub-nm), or if any zone's layout is
                inconsistent with the radius. The error message
                includes the zone name and offending word IDs.
        """
        if radius <= 0:
            raise ValueError(f"blockade_radius must be positive, got {radius}")
        radius_nm = _to_nm(radius, "blockade_radius")

        # Phase 1: scan every zone.  Any zone-level failure raises here
        # before we've mutated anything.
        scan_results: list[list[tuple[int, int]]] = [
            zone._scan_blockade_pairs(radius_nm) for zone in self._zones
        ]

        # Phase 2: commit.  No further failures possible.
        for zone, pairs in zip(self._zones, scan_results):
            zone._entangling_pairs = pairs
            zone._blockade_radius_nm = radius_nm
        self._blockade_radius = radius

    @property
    def blockade_radius(self) -> float | None:
        """Rydberg blockade radius (µm) applied to all zones, or None."""
        return self._blockade_radius

    def build(
        self,
        feed_forward: bool = False,
        atom_reloading: bool = False,
        blockade_radius: float | None = None,
    ) -> ArchSpec:
        """Assemble the ArchSpec and validate via Rust.

        Args:
            feed_forward: Whether the device supports feed-forward.
            atom_reloading: Whether the device supports atom reloading.
            blockade_radius: Explicit blockade radius (µm). If provided,
                overrides both builder-level and zone-level radii.

        Raises:
            ValueError: If Rust validation fails.
        """
        # 1. All zones share ONE word template (LocationAddress convention):
        # word_id is zone-local (0..Nw-1) and reused identically by every
        # zone; zone_id disambiguates. Validate that every zone declares the
        # same template, then emit that single Nw-length list.
        all_words: list[Word] = []
        if self._zones:
            template = self._zones[0]._words
            for zone in self._zones[1:]:
                if zone._words != template:
                    raise ValueError(
                        f"zone '{zone.name}' declares a different word "
                        f"template than zone '{self._zones[0].name}'; all "
                        f"zones must share the same word template (word_id is "
                        f"zone-local and reused identically across zones)."
                    )
            all_words = [Word(tuple(positions)) for positions in template]

        # 2. Build Rust Zone objects.
        # Word IDs are zone-local (0..Nw-1): every zone addresses the same
        # shared template, so no offset translation is applied. zone_id
        # disambiguates identical word_ids across zones.
        rust_zones: list[_RustZone] = []
        for zone in self._zones:
            site_buses = [_RustSiteBus(src=s, dst=d) for s, d in zone._site_buses]
            word_buses = [
                _RustWordBus(src=list(s), dst=list(d)) for s, d in zone._word_buses
            ]
            # Only words flagged at add_word(has_site_bus=True) are
            # eligible for site-bus transport. Default is True, so the
            # historical "all words opt-in when any site bus exists"
            # behavior is preserved unless the caller overrides.
            words_with_site_buses = (
                [w for w in range(zone.num_words) if zone._word_has_site_bus[w]]
                if site_buses
                else []
            )
            sites_with_word_buses = (
                list(range(zone.sites_per_word)) if word_buses else []
            )
            entangling_pairs = list(zone._entangling_pairs)
            rust_zones.append(
                _RustZone(
                    name=zone.name,
                    grid=zone._grid,
                    site_buses=site_buses,
                    word_buses=word_buses,
                    words_with_site_buses=words_with_site_buses,
                    sites_with_word_buses=sites_with_word_buses,
                    entangling_pairs=entangling_pairs,
                )
            )

        # 3. Build zone_buses from connect() calls.
        # word_ids stay zone-local; the (zone_id, word_id) pair addresses the
        # shared template within each endpoint zone.
        zone_buses: list[_RustZoneBus] = []
        for (src_name, src_words), (dst_name, dst_words) in self._connections:
            src_zid = self._zone_name_to_id[src_name]
            dst_zid = self._zone_name_to_id[dst_name]
            zone_buses.append(
                _RustZoneBus(
                    src=[(src_zid, w) for w in src_words],
                    dst=[(dst_zid, w) for w in dst_words],
                )
            )

        # 4. Build modes.
        modes: list[_RustMode] = []
        for mode_name, zone_names in self._modes:
            zone_ids = [self._zone_name_to_id[z] for z in zone_names]
            bitstring_order: list[_RustLocAddr] = []
            for zid in zone_ids:
                zone = self._zones[zid]
                for w in range(zone.num_words):
                    for s in range(zone.sites_per_word):
                        bitstring_order.append(_RustLocAddr(zid, w, s))
            modes.append(
                _RustMode(
                    name=mode_name,
                    zones=zone_ids,
                    bitstring_order=bitstring_order,
                )
            )

        # 5. Compute AOD waypoint paths for site and word buses.
        all_paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
        for zone_idx, zone in enumerate(self._zones):
            all_paths.update(zone._compute_paths(zone_idx))

        # 6. Determine the blockade radius to record on the ArchSpec.
        # Precedence: explicit build(blockade_radius=...) argument >
        # builder-level set_blockade_radius > zone-level agreement > None.
        resolved_radius = (
            blockade_radius
            if blockade_radius is not None
            else self._resolve_blockade_radius()
        )

        # 7. Assemble and validate.
        return ArchSpec.from_components(
            words=tuple(all_words),
            zones=tuple(rust_zones),
            modes=modes,
            zone_buses=zone_buses,
            paths=all_paths or None,
            feed_forward=feed_forward,
            atom_reloading=atom_reloading,
            blockade_radius=resolved_radius,
        )

    def _resolve_blockade_radius(self) -> float | None:
        """Pick the blockade_radius value for the final ArchSpec.

        Precedence:

        1. ``self._blockade_radius`` (builder-scope value from
           :meth:`set_blockade_radius`) — always authoritative when set.
        2. A unique radius shared by all zones that have one set; zones
           without a radius must also agree (i.e., no zone opt-out) for
           this branch to apply.
        3. ``None`` otherwise.

        Raises ``ValueError`` if zones disagree on a radius or if some
        zones have a radius set and others don't.
        """
        if self._blockade_radius is not None:
            return self._blockade_radius
        zone_radii = [zone.blockade_radius for zone in self._zones]
        if all(r is None for r in zone_radii):
            return None
        missing = [z.name for z, r in zip(self._zones, zone_radii) if r is None]
        if missing:
            raise ValueError(
                "blockade_radius is set on some zones but not others; "
                f"missing on: {missing}. Either call "
                "ArchBuilder.set_blockade_radius to apply uniformly, "
                "or leave it unset on every zone."
            )
        # Every zone has a non-None radius here.
        present = [r for r in zone_radii if r is not None]
        unique = sorted(set(present))
        if len(unique) > 1:
            raise ValueError(
                f"Zones disagree on blockade_radius: {unique}. "
                "The ArchSpec can only carry a single value."
            )
        return unique[0]
