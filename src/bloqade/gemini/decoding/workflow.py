from __future__ import annotations

from typing import cast
from collections.abc import Mapping

import numpy as np

from bloqade.gemini.decoding.postselection import PostselectionCurveData


def _plot_decoder_curves(
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
