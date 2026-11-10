"""Gemini method tables for compiler validation."""

from __future__ import annotations

from kirin import ir, interp
from kirin.lattice.empty import EmptyLattice
from bloqade.lanes.dialects import arch as qubit
from kirin.analysis.forward import ForwardFrame
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.validation.duplicate_address import (
    _DuplicateAddressValidationAnalysis,
)


@qubit.dialect.register(key="lanes.arch.duplicates")
class _NewAtDuplicateMethods(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_duplicate(
        self,
        _interp: _DuplicateAddressValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):

        z = _interp.maybe_const(node.zone_id, int)
        w = _interp.maybe_const(node.word_id, int)
        s = _interp.maybe_const(node.site_id, int)

        if z is None or w is None or s is None:
            # No address to compare, so nothing to say -- ConstAddressValidation
            # reports the non-const arg. Skip this statement rather than raise:
            # raising aborts the whole forward walk, which used to make one
            # non-const NewAt silently hide every duplicate after it.
            return (EmptyLattice.bottom(),)

        addr = LocationAddress(word_id=w, site_id=s, zone_id=z)
        if addr in _interp.seen:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"address (zone={z}, word={w}, site={s}) is pinned by two "
                    f"operations.new_at calls",
                ),
            )
        else:
            _interp.seen[addr] = node

        return (EmptyLattice.bottom(),)
