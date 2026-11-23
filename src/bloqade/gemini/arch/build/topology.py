"""Topology protocols and implementations for zone-based architectures.

Topologies define connectivity patterns (buses) for words and sites.
All topologies are 2D-grid-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bloqade.lanes.bytecode._native import SiteBus, WordBus

if TYPE_CHECKING:
    from .word_factory import WordGrid


# ── Protocols ──


@runtime_checkable
class SiteTopology(Protocol):
    """Generates site buses for movement within a single word (row of sites)."""

    def generate_site_buses(self, num_sites: int) -> tuple[SiteBus, ...]: ...


@runtime_checkable
class WordTopology(Protocol):
    """Generates word buses for movement between words in a 2D grid."""

    def generate_word_buses(self, grid: WordGrid) -> tuple[WordBus, ...]: ...


@runtime_checkable
class InterZoneTopology(Protocol):
    """Generates word buses connecting words across two zones."""

    def generate_word_buses(
        self, grid_a: WordGrid, grid_b: WordGrid
    ) -> tuple[WordBus, ...]: ...


# ── Helpers ──


def _check_power_of_two(n: int, name: str) -> int:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"{name} must be a power of 2, got {n}")
    return int(log2(n))


def _next_power_of_two(n: int) -> tuple[int, int]:
    """Return (padded, dims) where padded is the smallest power of 2 >= n."""
    if n < 1:
        raise ValueError(f"num_sites must be >= 1, got {n}")
    if n == 1:
        return (1, 0)
    dims = (n - 1).bit_length()
    return (1 << dims, dims)


# ── Site topologies ──


@dataclass(frozen=True)
class HypercubeSiteTopology:
    """Hypercube site connectivity within a single word.

    For N = 2^k sites, produces k buses. Bus for dimension d connects
    sites that differ in bit d: src = [sites with bit d=0],
    dst = [sites with bit d=1]. Each bus has N/2 parallel moves.

    For non-power-of-2 N, rounds up to the next power of 2 and filters
    out site indices >= N. Higher-indexed sites get fewer connections
    (e.g. for N=17, site 16 connects only to site 0 via dimension 4).
    """

    def generate_site_buses(self, num_sites: int) -> tuple[SiteBus, ...]:
        padded, dims = _next_power_of_two(num_sites)
        buses: list[SiteBus] = []
        for d in range(dims):
            mask = 1 << d
            src: list[int] = []
            dst: list[int] = []
            for i in range(padded):
                if i & mask == 0:
                    j = i | mask
                    if i < num_sites and j < num_sites:
                        src.append(i)
                        dst.append(j)
            if src:
                buses.append(SiteBus(src=src, dst=dst))
        return tuple(buses)


@dataclass(frozen=True)
class AllToAllSiteTopology:
    """All-to-all site connectivity: one bus per (src, dst) pair.

    For N sites, produces N*(N-1)/2 single-element buses allowing
    any site to reach any other site directly.
    """

    def generate_site_buses(self, num_sites: int) -> tuple[SiteBus, ...]:
        return tuple(
            SiteBus(src=[i], dst=[j])
            for i in range(num_sites)
            for j in range(i + 1, num_sites)
        )


# ── Word topologies ──


@dataclass(frozen=True)
class HypercubeWordTopology:
    """Hypercube word connectivity along both row and column dimensions.

    For a grid of (R x C) words where R = 2^r and C = 2^c, produces
    r + c buses. Row dimension d connects word(row, col) to
    word(row ^ 2^d, col). Column dimension d connects word(row, col)
    to word(row, col ^ 2^d).
    """

    def generate_word_buses(self, grid: WordGrid) -> tuple[WordBus, ...]:
        row_dims = _check_power_of_two(grid.num_rows, "num_rows")
        col_dims = _check_power_of_two(grid.num_cols, "num_cols")
        buses: list[WordBus] = []

        for d in range(row_dims):
            mask = 1 << d
            src: list[int] = []
            dst: list[int] = []
            for r in range(grid.num_rows):
                if r & mask != 0:
                    continue
                for c in range(grid.num_cols):
                    src.append(grid.word_id_at(r, c))
                    dst.append(grid.word_id_at(r | mask, c))
            buses.append(WordBus(src=src, dst=dst))

        for d in range(col_dims):
            mask = 1 << d
            src = []
            dst = []
            for r in range(grid.num_rows):
                for c in range(grid.num_cols):
                    if c & mask != 0:
                        continue
                    src.append(grid.word_id_at(r, c))
                    dst.append(grid.word_id_at(r, c | mask))
            buses.append(WordBus(src=src, dst=dst))

        return tuple(buses)


@dataclass(frozen=True)
class DiagonalWordTopology:
    """Diagonal word connectivity between adjacent column pairs.

    For a grid of (N rows x C cols), applies diagonal connectivity
    between each adjacent column pair (col_i, col_{i+1}). Per pair,
    produces 2*N - 1 buses. Total buses: (C-1) * (2*N - 1).

    Per column pair:
    - Group 1 (shift 0..N-1): col_a[r] -> col_b[r + shift]
      for r in 0..N-1-shift
    - Group 2 (shift 1..N-1): col_a[r + shift] -> col_b[r]
      for r in 0..N-1-shift (reverse diagonal)

    This gives full connectivity between all word pairs in adjacent
    columns, organized by diagonal. Non-adjacent columns are reachable
    via multi-hop.
    """

    def generate_word_buses(self, grid: WordGrid) -> tuple[WordBus, ...]:
        n = grid.num_rows
        buses: list[WordBus] = []

        for col_a in range(grid.num_cols - 1):
            col_b = col_a + 1

            # Group 1: col_a[r] -> col_b[r + shift]
            for shift in range(n):
                src = [grid.word_id_at(r, col_a) for r in range(n - shift)]
                dst = [grid.word_id_at(r + shift, col_b) for r in range(n - shift)]
                buses.append(WordBus(src=src, dst=dst))

            # Group 2: col_a[r + shift] -> col_b[r] (reverse diagonals)
            for diff in range(1, n):
                shift = n - diff
                src = [grid.word_id_at(r + shift, col_a) for r in range(n - shift)]
                dst = [grid.word_id_at(r, col_b) for r in range(n - shift)]
                buses.append(WordBus(src=src, dst=dst))

        return tuple(buses)


@dataclass(frozen=True)
class TransversalSiteTopology:
    """Physical site topology derived from a logical site topology via code expansion.

    Sites are organized in groups of ``code_distance``.  Group *g* contains
    physical sites ``[g*d, g*d+1, ..., g*d+d-1]`` where ``d = code_distance``.

    Produces two kinds of buses (transversal buses first so that logical
    bus IDs are preserved):

    1. **Transversal buses** -- each logical bus is "inflated" so that every
       ``(src, dst)`` element becomes ``d`` parallel physical elements.
       Logical bus *B* becomes physical bus *B* with the same index.

    2. **Intra-group buses** (optional) -- generated by ``intra_group_topology``
       for ``d`` sites, then replicated and offset for each group.  These
       support non-transversal operations within a code word (e.g. Steane
       code initialisation, syndrome extraction).
    """

    logical_topology: SiteTopology
    code_distance: int
    intra_group_topology: SiteTopology | None = None

    def generate_site_buses(self, num_sites: int) -> tuple[SiteBus, ...]:
        d = self.code_distance
        logical_sites = num_sites // d

        if logical_sites < 1:
            raise ValueError(f"num_sites={num_sites} too small for code_distance={d}")

        buses: list[SiteBus] = []

        # 1. Transversal buses: inflate each logical bus by d
        for logical_bus in self.logical_topology.generate_site_buses(logical_sites):
            expanded_src: list[int] = []
            expanded_dst: list[int] = []
            for s, t in zip(logical_bus.src, logical_bus.dst):
                for offset in range(d):
                    expanded_src.append(s * d + offset)
                    expanded_dst.append(t * d + offset)
            buses.append(SiteBus(src=expanded_src, dst=expanded_dst))

        # 2. Intra-group buses: replicated per group, offset by group base
        if self.intra_group_topology is not None:
            group_buses = self.intra_group_topology.generate_site_buses(d)
            for group in range(logical_sites):
                base = group * d
                for bus in group_buses:
                    buses.append(
                        SiteBus(
                            src=[s + base for s in bus.src],
                            dst=[t + base for t in bus.dst],
                        )
                    )

        return tuple(buses)


# ── Inter-zone topologies ──


@dataclass(frozen=True)
class MatchingTopology:
    """1:1 matching of words by grid position across two zones.

    Produces a single bus pairing grid_a(r, c) with grid_b(r, c)
    for all (r, c). Requires both grids to have the same dimensions.
    """

    def generate_word_buses(
        self, grid_a: WordGrid, grid_b: WordGrid
    ) -> tuple[WordBus, ...]:
        if (grid_a.num_rows, grid_a.num_cols) != (grid_b.num_rows, grid_b.num_cols):
            raise ValueError(
                f"Grid dimensions must match: "
                f"({grid_a.num_rows}, {grid_a.num_cols}) vs "
                f"({grid_b.num_rows}, {grid_b.num_cols})"
            )
        src: list[int] = []
        dst: list[int] = []
        for r in range(grid_a.num_rows):
            for c in range(grid_a.num_cols):
                src.append(grid_a.word_id_at(r, c))
                dst.append(grid_b.word_id_at(r, c))
        return (WordBus(src=src, dst=dst),)
