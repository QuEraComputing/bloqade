"""Matplotlib visualizations for Gemini logical qubits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from bloqade.gemini.steane_defaults import (
    STEANE7_DETECTOR_MATRIX,
    STEANE7_OBSERVABLE_MATRIX,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def render_steane_code_qubit(
    ax: Axes | None = None, center: tuple[float, float] = (0, 0)
) -> Axes:
    """Draw the Steane qubit's detector regions and logical observable."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim(center[0] - 2, center[0] + 2)
        ax.set_ylim(center[1] - 2, center[1] + 2)
        ax.axis("off")

    ring_angles = np.linspace(0, 2 * np.pi, 7)[:6]
    positions = np.empty((7, 2), dtype=float)
    positions[0] = center
    positions[1:] = (
        np.column_stack((1.5 * np.cos(ring_angles), 1.5 * np.sin(ring_angles))) + center
    )
    physical_ids = np.array([2, 0, 3, 6, 4, 5, 1])

    def selected_positions(row: np.ndarray) -> np.ndarray:
        return positions[np.asarray(row, dtype=bool)[physical_ids]]

    for detector, color in zip(
        STEANE7_DETECTOR_MATRIX,
        ("#670EFF", "#57BC13", "#EF2F55"),
    ):
        points = selected_positions(detector)
        ax.fill(*points[ConvexHull(points).vertices].T, color=color)

    for observable in STEANE7_OBSERVABLE_MATRIX:
        points = selected_positions(observable)
        edges = minimum_spanning_tree(cdist(points, points)).tocoo()
        for src, dst in zip(edges.row, edges.col):
            ax.plot(*points[[src, dst]].T, color="black", linewidth=5, zorder=50)

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        color="white",
        s=800,
        zorder=100,
        edgecolors="black",
    )
    for (x, y), label in zip(positions, physical_ids):
        ax.text(x, y, str(label), ha="center", va="center", zorder=200)

    return ax
