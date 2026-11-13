"""Cirq conversion for the supported Gemini logical circuit subset.

Imported lazily by ``bloqade.cirq_utils`` so importing Gemini logical kernels
does not require the optional Cirq dependency.

Use ``GeminiLogicalQubit(index, pin=(zone, word, site))`` for pinned qubits and
``cirq.LineQubit(index)`` for unpinned qubits. ``GeminiLogicalQubit`` is an alias
of the shared, pinned-only ``GeminiQubit`` type.
"""

from typing import Any

import cirq
from kirin import ir, interp, lowering
from kirin.dialects import func
from bloqade.cirq_utils.emit.base import EmitCirq, EmitCirqFrame
from bloqade.lanes.dialects.steane713 import stmts as logical_ops
from bloqade.lanes.dialects.steane713._dialect import dialect as logical_dialect
from bloqade.lanes.dialects.steane713.extensions import stmts as logical_extensions
from bloqade.lanes.dialects.steane713.extensions._dialect import (
    dialect as logical_extensions_dialect,
)

from bloqade.gemini.common.cirq_conversion import GeminiQubit, GeminiCirqLowerer

GeminiLogicalQubit = GeminiQubit


class GeminiLogicalCirqLowerer(GeminiCirqLowerer):
    """Import terminal-measurement Cirq circuits as Gemini logical IR.

    Cirq's final measurement is a structural marker here: its ideal bit values
    are not the seven physical readouts of logical.terminal_measure.
    """

    def run(
        self,
        stmt: cirq.Circuit,
        *,
        register_as_argument: bool = False,
        **kwargs: Any,
    ) -> ir.Region:
        if register_as_argument:
            raise lowering.BuildError(
                "Gemini logical Cirq loading does not support register_as_argument"
            )
        return super().run(stmt, register_as_argument=register_as_argument, **kwargs)

    def visit_MeasurementGate(
        self, state: lowering.State[cirq.Circuit], node: cirq.GateOperation
    ) -> ir.Statement:
        gate = node.gate
        assert isinstance(gate, cirq.MeasurementGate)
        if any(gate.invert_mask) or gate.confusion_map:
            raise lowering.BuildError(
                "Gemini logical terminal measurement does not support "
                "Cirq invert masks or confusion maps"
            )
        return state.current_frame.push(
            logical_ops.TerminalLogicalMeasurement(
                qubits=self.lower_qubit_getindices(state, node.qubits)
            )
        )


@logical_dialect.register(key="emit.cirq")
class _LogicalCirqMethods(interp.MethodTable):
    @interp.impl(logical_ops.TerminalLogicalMeasurement)
    def terminal_measure(
        self,
        emit: EmitCirq,
        frame: EmitCirqFrame,
        stmt: logical_ops.TerminalLogicalMeasurement,
    ) -> tuple[None]:
        # A direct return is discarded by emit_circuit(ignore_returns=True).
        # Other consumers (e.g. detector/observable post-processing) cannot be
        # represented by an ordinary Cirq measurement.
        if any(not isinstance(use.stmt, func.Return) for use in stmt.result.uses):
            raise interp.exceptions.InterpreterError(
                "Cirq emission cannot preserve logical measurement results or "
                "their detector/observable post-processing"
            )
        qids = frame.get(stmt.qubits)
        emit.circuit.append(cirq.measure(*qids), strategy=cirq.InsertStrategy.NEW)
        return (None,)

    @interp.impl(logical_ops.Initialize)
    def unsupported(
        self, emit: EmitCirq, frame: EmitCirqFrame, stmt: ir.Statement
    ) -> tuple[Any, ...]:
        raise interp.exceptions.InterpreterError(
            f"Cirq emission does not yet support {stmt.name}"
        )


@logical_extensions_dialect.register(key="emit.cirq")
class _LogicalExtensionsCirqMethods(interp.MethodTable):
    @interp.impl(logical_extensions.StarRz)
    def unsupported(
        self, emit: EmitCirq, frame: EmitCirqFrame, stmt: logical_extensions.StarRz
    ) -> tuple[Any, ...]:
        raise interp.exceptions.InterpreterError(
            f"Cirq emission does not yet support {stmt.name}"
        )
