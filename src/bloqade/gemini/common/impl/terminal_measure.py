"""Gemini method tables for compiler validation."""

from __future__ import annotations

import bloqade.qubit as qubit
from kirin import ir, interp as _interp
from kirin.analysis import ForwardFrame
from bloqade.lanes.validation.terminal_measure import (
    _collect_qubit_ids,
    _PhysicalTerminalMeasurementAnalysis,
)


@qubit.dialect.register(key="lanes.validate.physical.terminal_measure")
class _PhysicalMeasureValidation(_interp.MethodTable):
    @_interp.impl(qubit.stmts.Measure)
    def measure(
        self,
        interp: _PhysicalTerminalMeasurementAnalysis,
        frame: ForwardFrame,
        stmt: qubit.stmts.Measure,
    ):
        if interp.measure_count > 0:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Multiple terminal measurements are not allowed in physical circuits; "
                    "only one qubit.stmts.Measure is permitted.",
                ),
            )
        else:
            measured_ids = _collect_qubit_ids(interp.address_frame.get(stmt.qubits))
            expected_ids = set(range(interp.total_qubits))
            if measured_ids != expected_ids:
                interp.add_validation_error(
                    stmt,
                    ir.ValidationError(
                        stmt,
                        f"Terminal measure covers {len(measured_ids)} qubit(s) but "
                        f"{interp.total_qubits} were allocated; all qubits must be measured.",
                    ),
                )

        interp.measure_count += 1
        return (interp.lattice.bottom(),)
