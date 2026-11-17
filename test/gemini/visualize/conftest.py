"""Shared fixtures for the visualization tests.

Plotly is optional -- it arrives with the ``atom-vis`` extra, and a plain
``uv sync`` leaves it out. Without this, every test here that builds a figure
fails with ``ModuleNotFoundError: No module named 'plotly'``, which reads as a
regression rather than as a missing extra.

A module-level ``pytest.importorskip`` would be wrong here. These files are
mixed: many of their tests need no Plotly at all and cover matplotlib bounds,
circuit-column layout, argument validation and renderer selection. Skipping a
whole file would stop running those for everyone without the extra, which is
how coverage quietly disappears. So the skip is per test, via the ``plotly``
fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def plotly():
    """Skip the requesting test unless the ``atom-vis`` extra is installed.

    Request this from any test that builds a figure. Tests that only exercise
    pure-Python helpers should not, so they keep running without the extra.
    """
    return pytest.importorskip("plotly")
