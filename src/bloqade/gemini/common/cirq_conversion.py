"""Shared Cirq support for Gemini qubits with pinned architecture addresses.

Import this module lazily so Cirq remains an optional dependency of Gemini.
"""

from typing import Any
from dataclasses import dataclass

import cirq
from kirin import ir, interp, lowering
from kirin.dialects import py, ilist
from bloqade.cirq_utils.lowering import Squin
from bloqade.lanes.dialects.arch import stmts as gemini_qubit
from bloqade.cirq_utils.emit.base import EmitCirq, EmitCirqFrame
from bloqade.lanes.dialects.arch._dialect import dialect as gemini_qubit_dialect

from bloqade import qubit


@dataclass(frozen=True, eq=False, repr=False)
class GeminiQubit(cirq.Qid):
    """A Cirq qubit pinned to a required Gemini architecture address.

    ``pin`` is the nonnegative integer triple ``(zone, word, site)``. Use
    ``cirq.LineQubit(index)`` for an unpinned qubit instead. During conversion,
    GeminiQubit maps to ``new_at`` and LineQubit maps to ordinary ``qubit.New``;
    emission preserves those pinned and unpinned forms, respectively. As with
    the standard Squin loader, input Qids are sorted and renumbered in allocation
    order: ``index`` is a Cirq label, not a required internal register index.
    Pin coordinates are preserved independently of that renumbering.
    """

    index: int
    pin: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Gemini qubit indices must be nonnegative integers")
        if self.pin is None:
            raise ValueError(
                "A pinned address is required; use cirq.LineQubit for unpinned qubits"
            )
        if len(self.pin) != 3 or any(not isinstance(value, int) for value in self.pin):
            raise ValueError("A pinned address must be three integer coordinates")
        if any(value < 0 for value in self.pin):
            raise ValueError("Pinned address coordinates must be nonnegative")
        object.__setattr__(self, "pin", tuple(self.pin))

    @property
    def dimension(self) -> int:
        return 2

    def _comparison_key(self) -> tuple[int, tuple[int, int, int]]:
        return (self.index, self.pin)

    def __str__(self) -> str:
        zone, word, site = self.pin
        return f"Q{self.index}@({zone},{word},{site})"

    def __repr__(self) -> str:
        return f"GeminiQubit(index={self.index!r}, pin={self.pin!r})"


class GeminiCirqLowerer(Squin):
    """Load pinned Qids as ``new_at`` and other Qids as ordinary qubits.

    Qid ordering and dense register indices come from the standard Squin
    lowerer. Sparse labels and mixed Qid types are accepted whether or not
    any qubit is pinned; the original labels need not survive emission.
    """

    def visit_CircuitOperation(
        self, state: lowering.State[cirq.Circuit], node: cirq.CircuitOperation
    ) -> lowering.Result:
        if node.qubit_map:
            if node.repetitions != 1:
                raise lowering.BuildError(
                    "Mapped CircuitOperations with repetitions are not supported"
                )
            return self.visit(state, node.mapped_circuit())
        return super().visit_CircuitOperation(state, node)

    def run(
        self,
        stmt: cirq.Circuit,
        *,
        register_as_argument: bool = False,
        **kwargs: Any,
    ) -> ir.Region:
        if register_as_argument and any(
            isinstance(qid, GeminiQubit) for qid in self.circuit.all_qubits()
        ):
            raise lowering.BuildError(
                "A pinned Gemini qubit cannot be loaded as a register argument"
            )
        return super().run(stmt, register_as_argument=register_as_argument, **kwargs)

    def allocate_register(self, state: lowering.State[cirq.Circuit]) -> ir.SSAValue:
        values: list[ir.SSAValue] = []
        for qid in self.qreg_index:
            if isinstance(qid, GeminiQubit):
                coords = [
                    state.current_frame.push(py.Constant(value)).result
                    for value in qid.pin
                ]
                stmt = gemini_qubit.NewAt(*coords)
            else:
                stmt = qubit.stmts.New()
            values.append(state.current_frame.push(stmt).results[0])
        return state.current_frame.push(ilist.New(values=values)).result


@gemini_qubit_dialect.register(key="emit.cirq")
class _GeminiQubitCirqMethods(interp.MethodTable):
    @interp.impl(gemini_qubit.NewAt)
    def new_at(
        self, emit: EmitCirq, frame: EmitCirqFrame, stmt: gemini_qubit.NewAt
    ) -> tuple[GeminiQubit]:
        pin = (
            frame.get(stmt.zone_id),
            frame.get(stmt.word_id),
            frame.get(stmt.site_id),
        )
        if not all(isinstance(coord, int) for coord in pin):
            raise interp.exceptions.InterpreterError(
                "Pinned Gemini addresses must be compile-time integers"
            )
        qid = GeminiQubit(frame.qubit_index, pin=pin)
        if frame.qubits is not None:
            supplied = frame.qubits[frame.qubit_index]
            if supplied != qid:
                raise interp.exceptions.InterpreterError(
                    "A circuit_qubits entry for NewAt must be the matching "
                    "GeminiQubit with the same index and pinned address"
                )
        frame.qubit_index += 1
        return (qid,)
