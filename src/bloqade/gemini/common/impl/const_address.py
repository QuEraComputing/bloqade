"""Gemini method tables for compiler validation."""

from __future__ import annotations

from kirin import interp
from kirin.lattice.empty import EmptyLattice
from bloqade.lanes.dialects import arch as qubit
from kirin.analysis.forward import ForwardFrame
from bloqade.lanes.validation.const_address import (
    _expect_const_int,
    _ConstAddressValidationAnalysis,
)


@qubit.dialect.register(key="lanes.arch.const_address")
class _NewAtConstAddressMethods(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_const_address(
        self,
        _interp: _ConstAddressValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):
        # Every arg is checked, not just the first bad one, so a kernel that
        # gets two of the three wrong hears about both in one pass.
        _expect_const_int(node.zone_id, "zone_id", node, _interp)
        _expect_const_int(node.word_id, "word_id", node, _interp)
        _expect_const_int(node.site_id, "site_id", node, _interp)

        return (EmptyLattice.bottom(),)
