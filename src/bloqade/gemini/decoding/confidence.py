"""Gurobi-backed MILP decoding."""

from __future__ import annotations

from typing import Any

from bloqade.decoders import MILPDecoder


class GurobiDecoder(MILPDecoder):
    """Construct a MILP decoder using PuLP's GUROBI backend.

    Takes a Stim detector error model and optional solver keyword arguments.
    Implements ``BaseDecoder.decode`` and ``BaseDecoder.decode_confidence``;
    confidence uses the upstream normalized logical-gap score.
    Install ``bloqade[decoding]`` for PuLP and Gurobi support.
    """

    def _instantiate(self, verbose: bool = False, **kwargs: Any) -> None:
        # Use the setup hook so BaseDecoder.instantiate also selects Gurobi.
        super()._instantiate(verbose=verbose, solver="GUROBI", **kwargs)


__all__ = ["GurobiDecoder"]
