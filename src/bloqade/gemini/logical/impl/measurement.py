"""Gemini method tables for compiler validation."""

from __future__ import annotations

import bloqade.qubit as qubit
from kirin import ir, interp as _interp
from bloqade.squin import gate
from kirin.analysis import ForwardFrame
from kirin.dialects import func
from bloqade.lanes.dialects import steane713 as operations
from bloqade.analysis.address.impls import Func as AddressFuncMethodTable
from bloqade.lanes.dialects.steane713 import extensions as extensions
from bloqade.analysis.measure_id.lattice import MeasureIdTuple
from bloqade.lanes.validation.terminal_measurement import (
    _SteaneTerminalMeasurementValidationAnalysis,
)


@operations.dialect.register(key="lanes.validate.terminal_measurement")
class __GeminiLogicalMeasurementValidation(_interp.MethodTable):
    @_interp.impl(operations.stmts.TerminalLogicalMeasurement)
    def terminal_measure(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: operations.stmts.TerminalLogicalMeasurement,
    ):

        # should only be one terminal measurement EVER
        if not interp.terminal_measurement_encountered:
            interp.terminal_measurement_encountered = True
        else:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Multiple terminal measurements are not allowed in Gemini logical programs!",
                ),
            )
            return (interp.lattice.bottom(),)

        measurement_analysis_results = interp.measurement_analysis_results
        total_qubits_allocated = interp.unique_qubits_allocated

        # NOTE: `entries` rather than `Frame.get`, which *raises* on a missing
        # key. The analysis has no entry for this value when the statement sits
        # inside a callee that was not inlined -- it was evaluated in a nested
        # frame -- and the branch below already handles "no usable result", so a
        # miss belongs there rather than escaping as an `InterpreterError` for
        # `ValidationSuite` to re-report as "Validation pass '...' failed:" plus
        # a traceback. Reading `entries` directly also skips `get`'s parent-frame
        # fallback, which is unreachable here: this is the root frame the suite
        # cached, so its `parent` is None.
        measure_lattice_element = measurement_analysis_results.entries.get(stmt.result)

        if not isinstance(measure_lattice_element, MeasureIdTuple):
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Measurement ID Analysis failed to produce the necessary results needed for validation.",
                ),
            )
            return (interp.lattice.bottom(),)

        if len(measure_lattice_element.data) != total_qubits_allocated:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "The number of qubits in the terminal measurement does not match the number of total qubits allocated! "
                    + f"{total_qubits_allocated} qubits were allocated but only {len(measure_lattice_element.data)} were measured.",
                ),
            )
            return (interp.lattice.bottom(),)

        return (interp.lattice.bottom(),)


@extensions.dialect.register(key="lanes.validate.terminal_measurement")
class __GeminiLogicalExtensionsMeasurementValidation(_interp.MethodTable):
    @_interp.impl(extensions.stmts.StarRz)
    def star_rz(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: extensions.stmts.StarRz,
    ):
        interp.check_gate_after_measurement(stmt)
        return ()


@qubit.dialect.register(key="lanes.validate.terminal_measurement")
class _QubitGeminiMeasurementValidation(_interp.MethodTable):
    # This is a non-logical measurement, can safely flag as invalid
    @_interp.impl(qubit.stmts.Measure)
    def measure(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: qubit.stmts.Measure,
    ):

        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt,
                "Non-terminal measurements are not allowed in Gemini programs!",
            ),
        )

        return (interp.lattice.bottom(),)


@gate.dialect.register(key="lanes.validate.terminal_measurement")
class _GateGeminiMeasurementValidation(_interp.MethodTable):
    # NOTE: every concrete statement of the gate dialect must be listed here;
    # a gate missing from this list falls through to `eval_fallback` and is
    # silently accepted after the terminal measurement. The test suite checks
    # this list against `gate.dialect.stmts`.
    @_interp.impl(gate.stmts.X)
    @_interp.impl(gate.stmts.Y)
    @_interp.impl(gate.stmts.Z)
    @_interp.impl(gate.stmts.H)
    @_interp.impl(gate.stmts.T)
    @_interp.impl(gate.stmts.S)
    @_interp.impl(gate.stmts.SqrtX)
    @_interp.impl(gate.stmts.SqrtY)
    @_interp.impl(gate.stmts.Rx)
    @_interp.impl(gate.stmts.Ry)
    @_interp.impl(gate.stmts.Rz)
    @_interp.impl(gate.stmts.U3)
    @_interp.impl(gate.stmts.PhasedXZ)
    @_interp.impl(gate.stmts.CX)
    @_interp.impl(gate.stmts.CY)
    @_interp.impl(gate.stmts.CZ)
    @_interp.impl(gate.stmts.CCZ)
    @_interp.impl(gate.stmts.Swap)
    def gate_after_measurement(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.Gate,
    ):
        interp.check_gate_after_measurement(stmt)
        return ()


@func.dialect.register(key="lanes.validate.terminal_measurement")
class Func(AddressFuncMethodTable):
    pass
