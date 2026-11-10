"""Gemini method tables for compiler validation."""

from __future__ import annotations

from kirin import interp
from kirin.lattice.empty import EmptyLattice
from bloqade.lanes.dialects import arch as qubit
from kirin.analysis.forward import ForwardFrame
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.validation.address import _ValidationAnalysis
from bloqade.lanes.validation.const_address import _expect_const_int


@qubit.dialect.register(key="move.address.validation")
class _NewAtValidation(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_new_at(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):
        z = _expect_const_int(node.zone_id, "zone_id", node, _interp)
        w = _expect_const_int(node.word_id, "word_id", node, _interp)
        s = _expect_const_int(node.site_id, "site_id", node, _interp)

        if z is None or w is None or s is None:
            return (EmptyLattice.bottom(),)

        candidate = LocationAddress(word_id=w, site_id=s, zone_id=z)
        _interp.report_location_errors(node, (candidate,))

        return (EmptyLattice.bottom(),)
