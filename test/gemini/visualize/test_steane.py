"""Tests for the Gemini Steane-code Matplotlib diagram."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from bloqade.gemini.visualize import render_steane_code_qubit
from bloqade.gemini.steane_defaults import (
    STEANE7_DETECTOR_MATRIX,
    STEANE7_OBSERVABLE_MATRIX,
)


def _labels_at_vertices(ax, vertices: np.ndarray) -> set[int]:
    return {
        int(text.get_text())
        for text in ax.texts
        if any(np.allclose(text.get_position(), vertex) for vertex in vertices)
    }


def test_steane_diagram_uses_detector_and_observable_matrices() -> None:
    ax = render_steane_code_qubit()
    try:
        assert {int(text.get_text()) for text in ax.texts} == set(range(7))
        assert len(ax.patches) == STEANE7_DETECTOR_MATRIX.shape[0]
        assert len(ax.lines) == 2 * STEANE7_OBSERVABLE_MATRIX.shape[0]

        for patch, detector, color in zip(
            ax.patches,
            STEANE7_DETECTOR_MATRIX,
            ("#670EFF", "#57BC13", "#EF2F55"),
        ):
            assert isinstance(patch, Polygon)
            assert _labels_at_vertices(ax, np.asarray(patch.get_xy())) == set(
                np.flatnonzero(detector)
            )
            assert to_rgba(patch.get_facecolor()) == to_rgba(color)

        observable_edges = {
            frozenset(_labels_at_vertices(ax, np.asarray(line.get_xydata())))
            for line in ax.lines
        }
        assert observable_edges == {frozenset((0, 1)), frozenset((1, 5))}
        assert set().union(*observable_edges) == set(
            np.flatnonzero(STEANE7_OBSERVABLE_MATRIX[0])
        )
        assert all(line.get_linewidth() == 5 for line in ax.lines)
        assert ax.get_xlim() == (-2.0, 2.0)
        assert ax.get_ylim() == (-2.0, 2.0)
        assert not ax.axison
    finally:
        figure = ax.figure
        assert isinstance(figure, Figure)
        plt.close(figure)


def test_steane_diagram_draws_on_provided_axes_at_requested_center() -> None:
    figure, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    try:
        assert render_steane_code_qubit(ax=ax, center=(4, -3)) is ax
        center_label = next(text for text in ax.texts if text.get_text() == "2")
        assert center_label.get_position() == (4, -3)
        assert ax.get_xlim() == (-10.0, 10.0)
        assert ax.get_ylim() == (-10.0, 10.0)
    finally:
        plt.close(figure)
