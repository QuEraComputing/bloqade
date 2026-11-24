"""Helpers for logical magic-state-distillation experiments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bloqade.gemini.logical.experiments.postselection import (
        PostselectionCurveData,
    )

MSD_TARGET_BLOCH: NDArray[np.float64] = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
"""Bloch vector of the magic state that the MSD circuits distill."""

msd_postselection_obs: NDArray[np.uint8] = np.array([[1, 0, 1, 1]], dtype=np.uint8)
"""Accepted ancilla pattern for 5-to-1 MSD, for
``PostSelectionExperiment.decode_and_postselect``."""


def bloch_to_u3(bloch: NDArray) -> tuple[float, float, float]:
    """Return U3 angles ``(theta, phi, lam)`` that prepare ``bloch`` from ``|0>``."""
    x, y, z = bloch[0], bloch[1], bloch[2]
    theta = math.atan2(math.hypot(x, y), z)
    phi = math.atan2(y, x)
    lam = 0.0
    return (theta, phi, lam)


def plot_decoder_curves(
    curves: Mapping[str, PostselectionCurveData],
    *,
    injected_summary: float | None = None,
    min_accepted_fraction: float = 0.04,
    ax: object | None = None,
    title: str | None = None,
    log: bool = True,
):
    """Plot point-estimate decoder threshold curves."""

    if log:
        print("Plotting decoder curves...")

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    if ax is None:
        _, ax = plt.subplots()
    ax = cast(Axes, ax)
    fig = ax.figure

    for label, curve in curves.items():
        accepted = np.asarray(curve.accepted_fraction, dtype=np.float64)
        fidelity = np.asarray(curve.fidelity, dtype=np.float64)
        if len(accepted) > 0:
            ax.plot(accepted, fidelity, marker="o", linewidth=1.5, label=label)

    if injected_summary is not None:
        ax.axhline(
            float(injected_summary),
            linestyle="--",
            linewidth=1.2,
            color="black",
            label="Injected baseline",
        )

    ax.set_xscale("log")
    ax.set_xlim(left=min_accepted_fraction)
    ax.set_xlabel("Accepted fraction")
    ax.set_ylabel("Fidelity")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    return fig, ax


__all__ = [
    "MSD_TARGET_BLOCH",
    "bloch_to_u3",
    "msd_postselection_obs",
    "plot_decoder_curves",
]
