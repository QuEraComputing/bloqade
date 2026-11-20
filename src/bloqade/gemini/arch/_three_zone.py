"""Shared layout of the Gemini full and generic full example architectures."""

from itertools import pairwise

from bloqade.gemini.arch import (
    ArchResult,
    ArchBuilder,
    DiagonalWordTopology,
    HypercubeSiteTopology,
    create_zone_words,
)


def three_zone_arch(
    *, num_rows: int, num_cols: int, sites_per_word: int, storage_buses: bool
) -> ArchResult:
    grid = create_zone_words(
        num_rows=num_rows, num_cols=num_cols, sites_per_word=sites_per_word
    )
    site_spacing = 10.0  # µm
    row_spacing = 20.0  # µm, including the gap between zones
    columns = [site_spacing * col for col in range(num_cols * sites_per_word)]
    builder = ArchBuilder(
        grid_shape=(num_rows, len(columns)), word_shape=(1, sites_per_word)
    )
    for word in grid.words:
        row = word.site_indices[0][1]
        builder.add_word(rows=row, columns=[col for col, _ in word.site_indices])

    zone_names = ("storage_top", "entangling", "storage_bottom")
    zone_indices = {}
    site_buses = HypercubeSiteTopology().generate_site_buses(sites_per_word)
    word_buses = DiagonalWordTopology().generate_word_buses(grid)
    for i, name in enumerate(zone_names):
        zone_indices[name] = builder.add_zone(
            name,
            rows=[row_spacing * (i * num_rows + row) for row in range(num_rows)],
            columns=columns,
            x_clearance=3.0,
            y_clearance=3.0,
        )
        if storage_buses or name == "entangling":
            for bus in site_buses:
                builder.add_site_bus(name, bus.src, bus.dst)
            for bus in word_buses:
                builder.add_word_bus(name, bus.src, bus.dst)

    words_a, words_b = zip(*grid.cz_pairs())
    builder.add_entangling_pairs("entangling", words_a, words_b)
    for src, dst in pairwise(zone_names):
        builder.connect((src, grid.all_word_ids), (dst, grid.all_word_ids))
    builder.add_mode("all", zone_names)
    builder.add_mode("entangling", ["entangling"])
    return ArchResult(
        arch=builder.build(
            feed_forward=True, atom_reloading=True, blockade_radius=site_spacing
        ),
        zone_grids=dict.fromkeys(zone_names, grid),
        zone_indices=zone_indices,
    )
