"""Gemini method tables for compiler validation."""

from __future__ import annotations

from kirin import ir, interp
from kirin.dialects import ilist
from kirin.lattice.empty import EmptyLattice
from kirin.analysis.forward import ForwardFrame
from bloqade.lanes.dialects.arch import Loc
from bloqade.lanes.dialects.arrange import dialect as arrange_dialect
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.validation.address import _ValidationAnalysis
from bloqade.lanes.dialects.arrange.stmts import MoveTo, Permute
from bloqade.lanes.validation.const_address import _expect_const_int


@arrange_dialect.register(key="move.address.validation")
class _MoveToValidationMethods(interp.MethodTable):
    @interp.impl(MoveTo)
    def check_move_to(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: MoveTo,
    ):
        # qubits must be an ilist.New result so we can count them;
        # ResultValue gives us the owning statement via .owner
        from kirin.ir import ResultValue
        from kirin.analysis import const

        if not isinstance(node.qubits, ResultValue):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "move_to: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubits_owner = node.qubits.owner
        if not isinstance(qubits_owner, ilist.New):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "move_to: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubit_values = qubits_owner.values

        # Failure 2: locations must be compile-time constants
        locs_hint = node.locations.hints.get("const")
        if not isinstance(locs_hint, const.Value):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "move_to: locations must be compile-time constants "
                    "(pass a literal list of LocationAddress values)",
                ),
            )
            return (EmptyLattice.bottom(),)

        location_values: tuple[LocationAddress, ...] = tuple(locs_hint.data)

        # Failure 1: length mismatch
        if len(qubit_values) != len(location_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"move_to: len(qubits)={len(qubit_values)} != "
                    f"len(locations)={len(location_values)}",
                ),
            )
            return (EmptyLattice.bottom(),)

        # Failure 3: out-of-range LocationAddress (delegated to arch spec)
        _interp.report_location_errors(node, location_values)

        # Failure 4: duplicate destination addresses
        if len(set(location_values)) != len(location_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "move_to: duplicate destination LocationAddress within one call",
                ),
            )

        # Failure 5: duplicate qubit SSA values
        seen_ids: set[int] = set()
        for qv in qubit_values:
            if id(qv) in seen_ids:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        "move_to: same Qubit SSA value appears more than once "
                        "in qubits list",
                    ),
                )
                break
            seen_ids.add(id(qv))

        return (EmptyLattice.bottom(),)

    @interp.impl(Permute)
    def check_permute(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: Permute,
    ):
        from kirin.ir import ResultValue
        from kirin.analysis import const

        if not isinstance(node.qubits, ResultValue) or not isinstance(
            qubits_owner := node.qubits.owner, ilist.New
        ):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "permute: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubit_values = qubits_owner.values

        # P1: perm must be compile-time constant
        perm_hint = node.perm.hints.get("const")
        if not isinstance(perm_hint, const.Value):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "permute: perm must be compile-time constants "
                    "(pass a literal list of ints)",
                ),
            )
            return (EmptyLattice.bottom(),)

        perm_values = tuple(int(p) for p in perm_hint.data)

        # P2: length mismatch
        if len(qubit_values) != len(perm_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"permute: len(qubits)={len(qubit_values)} != "
                    f"len(perm)={len(perm_values)}",
                ),
            )
            return (EmptyLattice.bottom(),)

        # P3: perm must be a bijection of range(n)
        if sorted(perm_values) != list(range(len(perm_values))):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "permute: perm must be a permutation of range(len(qubits)) "
                    "(each index 0..n-1 exactly once)",
                ),
            )

        # P4: duplicate qubit SSA values
        seen_ids: set[int] = set()
        for qv in qubit_values:
            if id(qv) in seen_ids:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        "permute: same Qubit SSA value appears more than once "
                        "in qubits list",
                    ),
                )
                break
            seen_ids.add(id(qv))

        return (EmptyLattice.bottom(),)

    @interp.impl(Loc)
    def check_loc(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: Loc,
    ):
        """Validate a residual (non-const-folded) ``arch.loc(zone, row, col)``.

        A constant ``loc`` const-folds to a ``LocationAddress`` during
        ``AggressiveUnroll`` and never reaches validation; this fires only on a
        residual ``Loc`` — coordinates that are not compile-time constants, or
        (defensively) that don't resolve to a location in the arch spec. The
        ``(zone, row, col)`` resolution against ``ArchSpec.location_at`` is the
        in-range check for the coordinate addressing scheme.
        """

        zone = _expect_const_int(node.zone, "zone", node, _interp)
        row = _expect_const_int(node.row, "row", node, _interp)
        col = _expect_const_int(node.col, "col", node, _interp)

        if zone is None or row is None or col is None:
            return (EmptyLattice.bottom(),)

        if _interp.arch_spec.location_at(zone, row, col) is None:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"loc: no location at (zone={zone}, row={row}, col={col}) "
                    "in the architecture spec",
                ),
            )

        return (EmptyLattice.bottom(),)
