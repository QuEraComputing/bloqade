"""Architecture blueprint + builder entry point.

Provides the declarative schema (``ZoneSpec``, ``DeviceLayout``,
``ArchBlueprint``) and the high-level ``build_arch`` function that
turns a blueprint into a validated ``ArchSpec``.

Internally uses ``ZoneBuilder`` + ``ArchBuilder`` (from
``.imperative``) for correct-by-construction zone assembly. One
blueprint zone = one Rust zone (no splitting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.lanes.bytecode._native import Grid as _RustGrid
from bloqade.lanes.arch.spec import ArchSpec

from .imperative import ArchBuilder, ZoneBuilder
from .word_factory import WordGrid, create_zone_words

if TYPE_CHECKING:
    from .topology import InterZoneTopology, SiteTopology, WordTopology


# ── Declarative schema (formerly arch/zone.py) ──


@dataclass(frozen=True)
class ZoneSpec:
    """Specification for a single zone in a multi-zone architecture.

    Words are arranged in a 2D grid (num_rows x num_cols). Horizontally
    adjacent word pairs form CZ entangling pairs in entangling zones.

    Args:
        num_rows: Number of rows in the word grid. Must be >= 1.
        num_cols: Number of columns in the word grid. Must be >= 2 and even.
        entangling: Whether this zone supports CZ entangling gates.
        measurement: Whether this zone supports measurement.
    """

    num_rows: int
    num_cols: int
    entangling: bool = False
    measurement: bool = True
    word_topology: WordTopology | None = None
    site_topology: SiteTopology | None = None

    def __post_init__(self) -> None:
        if self.num_rows < 1:
            raise ValueError(f"num_rows must be >= 1, got {self.num_rows}")
        if self.num_cols < 2:
            raise ValueError(f"num_cols must be >= 2, got {self.num_cols}")
        if self.num_cols % 2 != 0:
            raise ValueError(f"num_cols must be even, got {self.num_cols}")

    @property
    def num_words(self) -> int:
        """Total number of words in this zone."""
        return self.num_rows * self.num_cols


@dataclass(frozen=True)
class DeviceLayout:
    """Physical layout parameters for word and site placement.

    Words are horizontal rows with interleaved CZ pairs. Within a pair,
    sites alternate: even word at x=0, 2s, 4s, ... and odd word at
    x=s, 3s, 5s, ... where s = site_spacing (blockade radius).

    Args:
        sites_per_word: Number of sites per word.
        site_spacing: Distance between adjacent atoms (micrometers).
            Also determines the CZ blockade distance between paired sites.
        pair_spacing: Horizontal gap between adjacent CZ pairs (micrometers).
        row_spacing: Vertical distance between word grid rows (micrometers).
        zone_gap: Additional vertical gap between zones (micrometers).
        x_clearance: Minimum x-axis clearance (µm) between AOD path
            waypoints and grid lines.
        y_clearance: Minimum y-axis clearance (µm) between AOD path
            waypoints and grid lines.  Separate x/y values are useful
            when row and column spacings differ substantially.
    """

    sites_per_word: int = 5
    site_spacing: float = 10.0
    pair_spacing: float = 10.0
    row_spacing: float = 20.0
    zone_gap: float = 20.0
    x_clearance: float = 3.0
    y_clearance: float = 3.0

    def __post_init__(self) -> None:
        if self.sites_per_word < 1:
            raise ValueError(f"sites_per_word must be >= 1, got {self.sites_per_word}")
        if self.x_clearance <= 0:
            raise ValueError(f"x_clearance must be positive, got {self.x_clearance}")
        if self.y_clearance <= 0:
            raise ValueError(f"y_clearance must be positive, got {self.y_clearance}")
        for name in ("site_spacing", "pair_spacing", "row_spacing", "zone_gap"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative, got {getattr(self, name)}"
                )


@dataclass(frozen=True)
class ArchBlueprint:
    """High-level architecture definition composed of named zones and layout.

    Zones are ordered by insertion order of the ``zones`` dict, which
    determines ``zone_id`` assignment and vertical layout (top to bottom).
    All zones share one word template addressed by zone-local word IDs
    (``0..Nw-1``); ``zone_id`` disambiguates.

    Args:
        zones: Named zones with their specifications.
        layout: Physical layout parameters for word/site placement.
        feed_forward: Whether the device supports feed-forward operations.
        atom_reloading: Whether the device supports atom reloading.
        blockade_radius: Rydberg blockade radius (µm) to record on the
            ArchSpec. When set, this value is passed directly to
            ``ArchSpec.from_components`` and overrides any radius derived
            from ``ArchBuilder.set_blockade_radius`` or zone-level scans.
    """

    zones: dict[str, ZoneSpec]
    layout: DeviceLayout = field(default_factory=DeviceLayout)
    feed_forward: bool = False
    atom_reloading: bool = False
    blockade_radius: float | None = None

    def __post_init__(self) -> None:
        if not self.zones:
            raise ValueError("At least one zone must be specified")
        dims = {(s.num_rows, s.num_cols) for s in self.zones.values()}
        if len(dims) > 1:
            raise ValueError(
                f"All zones must have the same grid dimensions, got {dims}"
            )

    @property
    def words_per_zone(self) -> int:
        """Number of words per zone (all zones have equal grid dimensions).

        Under the shared-template convention this is also the size of the
        single global word list (``len(ArchSpec.words)``): every zone reuses
        the same ``0..Nw-1`` template.
        """
        return next(iter(self.zones.values())).num_words

    @property
    def zone_names(self) -> tuple[str, ...]:
        """Zone names in definition order."""
        return tuple(self.zones.keys())


# ── Builder entry point (formerly arch/builder.py) ──


@dataclass(frozen=True)
class ArchResult:
    """Result of build_arch(), containing the ArchSpec and metadata."""

    arch: ArchSpec
    zone_grids: dict[str, WordGrid]
    zone_indices: dict[str, int]


def _build_zone_grid(
    zone_spec: ZoneSpec,
    layout: DeviceLayout,
    n: int,
    s: float,
    y_offset: float = 0.0,
) -> _RustGrid:
    """Build a Rust Grid covering all columns and rows in a zone.

    The grid must have enough x-positions to cover all interleaved CZ pairs
    and enough y-positions for all rows. ``y_offset`` shifts the zone
    vertically so that stacked zones do not overlap in physical space.
    """
    num_rows = zone_spec.num_rows
    num_cols = zone_spec.num_cols
    pair_width = (2 * n - 1) * s

    x_positions: list[float] = []
    num_pairs = num_cols // 2
    for pair_idx in range(num_pairs):
        pair_x = pair_idx * (pair_width + layout.pair_spacing)
        for i in range(n):
            x_positions.append(pair_x + 2.0 * s * i)
            x_positions.append(pair_x + s + 2.0 * s * i)

    x_pos_sorted = sorted(set(x_positions))
    y_positions = [y_offset + row * layout.row_spacing for row in range(num_rows)]
    if not y_positions:
        y_positions = [y_offset]

    return _RustGrid.from_positions(x_pos_sorted, y_positions)


def build_arch(
    blueprint: ArchBlueprint,
    connections: dict[tuple[str, str], InterZoneTopology] | None = None,
) -> ArchResult:
    """Build an ArchSpec from a blueprint and inter-zone connections.

    One blueprint zone maps to one Rust zone. Entangling pairs are
    metadata on the zone, not a reason to split into sub-zones.

    Args:
        blueprint: Architecture blueprint with zones and layout.
        connections: Inter-zone connectivity. Keys are (zone_a, zone_b) name
            pairs, values are InterZoneTopology instances.

    Returns:
        ArchResult with the validated ArchSpec and metadata.
    """
    connections = connections or {}
    layout = blueprint.layout
    n = layout.sites_per_word
    s = layout.site_spacing

    # Validate connections reference valid zones.
    for zone_a, zone_b in connections:
        if zone_a == zone_b:
            raise ValueError(
                f"Self-connection not allowed: '{zone_a}'. "
                "Use word_topology on ZoneSpec for intra-zone connectivity."
            )
        if zone_a not in blueprint.zones:
            raise ValueError(f"Unknown zone '{zone_a}' in connection")
        if zone_b not in blueprint.zones:
            raise ValueError(f"Unknown zone '{zone_b}' in connection")

    # 1. Create word grids (preserves row/col structure for topology generators).
    # Word IDs are zone-local (0..Nw-1) and shared identically across zones.
    zone_grids: dict[str, WordGrid] = {}
    for zone_name, zone_spec in blueprint.zones.items():
        zone_grids[zone_name] = create_zone_words(zone_spec, layout)

    # 2. Build ZoneBuilders from blueprint zones.
    zone_builders: dict[str, ZoneBuilder] = {}
    y_offset = 0.0

    for zone_name, zone_spec in blueprint.zones.items():
        word_grid = zone_grids[zone_name]
        rust_grid = _build_zone_grid(zone_spec, layout, n, s, y_offset=y_offset)
        # Advance y_offset past this zone's rows + the inter-zone gap.
        zone_height = max(0, zone_spec.num_rows - 1) * layout.row_spacing
        y_offset += zone_height + layout.zone_gap
        word_shape = _word_shape_from_layout(zone_spec, layout)

        zone = ZoneBuilder(
            zone_name,
            rust_grid,
            word_shape,
            x_clearance=layout.x_clearance,
            y_clearance=layout.y_clearance,
        )

        # Place words on the grid using the same index pattern as create_zone_words.
        for row in range(zone_spec.num_rows):
            for col in range(zone_spec.num_cols):
                word = word_grid.word_at(row, col)
                # Extract x and y indices from the word's site positions.
                x_indices = sorted({site[0] for site in word.site_indices})
                y_indices = sorted({site[1] for site in word.site_indices})
                zone.add_word(x_indices, y_indices)

        # Site buses from topology.
        if zone_spec.site_topology is not None:
            for bus in zone_spec.site_topology.generate_site_buses(
                layout.sites_per_word
            ):
                zone.add_site_bus(list(bus.src), list(bus.dst))

        # Intra-zone word buses from topology (already zone-local IDs).
        if zone_spec.word_topology is not None:
            for bus in zone_spec.word_topology.generate_word_buses(word_grid):
                zone.add_word_bus(src=list(bus.src), dst=list(bus.dst))

        # Entangling pairs (cz_pairs yields zone-local word IDs).
        if zone_spec.entangling:
            pairs = list(word_grid.cz_pairs())
            zone.add_entangling_pairs(
                [a for a, _ in pairs],
                [b for _, b in pairs],
            )

        zone_builders[zone_name] = zone

    # 3. Compose zones into ArchBuilder.
    arch_builder = ArchBuilder()
    zone_indices: dict[str, int] = {}

    for zone_name, zone in zone_builders.items():
        zid = arch_builder.add_zone(zone)
        zone_indices[zone_name] = zid

    # 4. Inter-zone connections → zone_buses.
    for (zone_a_name, zone_b_name), topology in connections.items():
        grid_a = zone_grids[zone_a_name]
        grid_b = zone_grids[zone_b_name]

        for bus in topology.generate_word_buses(grid_a, grid_b):
            # Topology generators already emit zone-local word IDs.
            arch_builder.connect(
                src=(zone_a_name, list(bus.src)),
                dst=(zone_b_name, list(bus.dst)),
            )

    # 5. Modes.
    all_zone_names = list(blueprint.zones.keys())
    arch_builder.add_mode("all", all_zone_names)

    for name, spec in blueprint.zones.items():
        if spec.measurement:
            arch_builder.add_mode(name, [name])

    # 6. Build and return.
    arch = arch_builder.build(
        feed_forward=blueprint.feed_forward,
        atom_reloading=blueprint.atom_reloading,
        blockade_radius=blueprint.blockade_radius,
    )

    return ArchResult(
        arch=arch,
        zone_grids=zone_grids,
        zone_indices=zone_indices,
    )


def _word_shape_from_layout(
    zone_spec: ZoneSpec, layout: DeviceLayout
) -> tuple[int, int]:
    """Derive word_shape from zone spec and layout.

    For interleaved CZ pairs, each word occupies ``sites_per_word`` x-positions
    and 1 y-position (row). This assumes 1D words; 2D word shapes would require
    extending ``ZoneSpec`` with a word shape parameter.
    """
    _ = zone_spec  # reserved for future 2D word shapes
    n = layout.sites_per_word
    return (n, 1)
