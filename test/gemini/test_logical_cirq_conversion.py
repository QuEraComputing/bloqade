from typing import Any, cast

import cirq
import pytest
from kirin import ir, interp, lowering
from kirin.dialects import py, ilist
from bloqade.cirq_utils import emit_circuit, load_circuit
from kirin.ir.exception import ValidationErrorGroup
from bloqade.cirq_utils.lowering import Squin
from bloqade.lanes.dialects.arch import new_at
from bloqade.cirq_utils.emit.base import EmitCirq, EmitCirqFrame
from bloqade.lanes.dialects.steane713 import stmts as logical_ops
from bloqade.lanes.dialects.arch.stmts import NewAt
from bloqade.lanes.dialects.steane713.stmts import TerminalLogicalMeasurement

from bloqade import qubit, squin
from bloqade.gemini import logical, physical
from bloqade.gemini.device import GeminiLogicalSimulator
from bloqade.gemini.common.cirq_conversion import (
    GeminiQubit,
    GeminiCirqLowerer,
    _GeminiQubitCirqMethods,
)
from bloqade.gemini.logical.cirq_conversion import (
    GeminiLogicalQubit,
    _LogicalCirqMethods,
)


def test_load_plain_cirq_as_logical_kernel():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1))

    method = load_circuit(circuit, dialects=logical.kernel)

    assert method.dialects is logical.kernel
    assert (
        sum(
            isinstance(stmt, TerminalLogicalMeasurement)
            for stmt in method.callable_region.walk()
        )
        == 1
    )
    assert (
        sum(isinstance(stmt, qubit.stmts.New) for stmt in method.callable_region.walk())
        == 2
    )
    method.verify()
    method.verify_type()


def test_logical_loader_accepts_unmeasured_subcircuit_with_final_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    body = cirq.FrozenCircuit(cirq.H(q0), cirq.CX(q0, q1))
    circuit = cirq.Circuit(cirq.CircuitOperation(body), cirq.measure(q0, q1))

    method = load_circuit(circuit, dialects=logical.kernel)

    assert (
        sum(
            isinstance(stmt, TerminalLogicalMeasurement)
            for stmt in method.callable_region.walk()
        )
        == 1
    )


def test_logical_loader_preserves_subcircuit_qubit_mapping():
    q0, q1 = cirq.LineQubit.range(2)
    operation = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(q0)))
    operation = operation.with_qubit_mapping({q0: q1})
    circuit = cirq.Circuit(cirq.X(q0), operation, cirq.measure(q0, q1))

    method = load_circuit(circuit, dialects=logical.kernel)
    emitted = emit_circuit(method)

    assert cirq.H(q1) in emitted.all_operations()
    assert cirq.H(q0) not in emitted.all_operations()


def test_logical_loader_preserves_terminal_measurement_order():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q1, q0))

    method = load_circuit(circuit, dialects=logical.kernel)

    assert emit_circuit(method) == circuit


def test_pinned_logical_qubit_round_trip():
    @logical.kernel
    def program():
        register = ilist.IList([new_at(1, 2, 3), qubit.new()])
        squin.h(register[0])
        squin.cx(register[0], register[1])
        logical.terminal_measure(register)

    circuit = emit_circuit(program)
    qids = circuit.all_qubits()
    assert GeminiLogicalQubit(0, (1, 2, 3)) in qids
    assert cirq.LineQubit(1) in qids

    reloaded = load_circuit(circuit, dialects=logical.kernel)
    pins = [stmt for stmt in reloaded.callable_region.walk() if isinstance(stmt, NewAt)]
    assert len(pins) == 1
    for coordinate, expected in zip(
        (pins[0].zone_id, pins[0].word_id, pins[0].site_id), (1, 2, 3), strict=True
    ):
        assert isinstance(coordinate.owner, py.Constant)
        assert coordinate.owner.value.unwrap() == expected
    assert (
        sum(
            isinstance(stmt, TerminalLogicalMeasurement)
            for stmt in reloaded.callable_region.walk()
        )
        == 1
    )
    reloaded.verify()
    reloaded.verify_type()
    assert emit_circuit(reloaded) == circuit


@pytest.mark.parametrize(
    "supplied", [cirq.LineQubit(0), GeminiLogicalQubit(0, (3, 2, 1))]
)
def test_pinned_emission_rejects_mismatched_circuit_qubits(supplied: cirq.Qid):
    @logical.kernel
    def program():
        register = ilist.IList([new_at(1, 2, 3)])
        logical.terminal_measure(register)

    with pytest.raises(interp.exceptions.InterpreterError, match="matching"):
        emit_circuit(program, circuit_qubits=[supplied])


def test_logical_loader_rejects_mid_circuit_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0), cirq.H(q1), cirq.measure(q1))

    with pytest.raises(ValidationErrorGroup, match="Multiple terminal measurements"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_partial_terminal_measurement():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.measure(q0))

    with pytest.raises(ValidationErrorGroup, match="only 1 were measured"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_missing_terminal_measurement():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0))

    with pytest.raises(ValidationErrorGroup, match="exactly one"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_inverted_measurement():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q, invert_mask=(True,)))

    with pytest.raises(lowering.BuildError, match="invert masks"):
        load_circuit(circuit, dialects=logical.kernel)


def test_logical_loader_rejects_register_argument_mode():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q))

    with pytest.raises(lowering.BuildError, match="register_as_argument"):
        load_circuit(circuit, dialects=logical.kernel, register_as_argument=True)


def test_emit_rejects_logical_measurement_postprocessing():
    @logical.kernel
    def program():
        register = qubit.qalloc(1)
        return logical.default_post_processing(register)

    with pytest.raises(
        interp.exceptions.InterpreterError, match="detector/observable post-processing"
    ):
        emit_circuit(program, ignore_returns=True)


def test_emit_rejects_logical_star_rz_extension():
    @logical.kernel
    def program():
        register = qubit.qalloc(1)
        logical.extensions.star_rz(0.1, register[0])
        logical.terminal_measure(register)

    with pytest.raises(interp.exceptions.InterpreterError, match="star_rz"):
        emit_circuit(program, ignore_returns=True)


def test_emit_bell_kernel_returning_terminal_measurement():
    @logical.kernel
    def logical_bell():
        register = squin.qalloc(2)
        squin.h(register[0])
        squin.cx(register[0], register[1])
        return logical.terminal_measure(register)

    circuit = emit_circuit(logical_bell, ignore_returns=True)
    q0, q1 = cirq.LineQubit.range(2)
    assert circuit == cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1))


def test_emit_compiled_bell_has_one_physical_measurement():
    @logical.kernel
    def logical_bell():
        register = squin.qalloc(2)
        squin.h(register[0])
        squin.cx(register[0], register[1])
        return logical.terminal_measure(register)

    physical = (
        GeminiLogicalSimulator().task(logical_bell).noiseless_physical_squin_kernel
    )
    physical_measurements = [
        stmt
        for stmt in physical.callable_region.walk()
        if isinstance(stmt, qubit.stmts.Measure)
    ]
    assert len(physical_measurements) == 1
    assert isinstance(physical_measurements[0].qubits.owner, ilist.New)
    assert len(physical_measurements[0].qubits.owner.values) == 14

    circuit = emit_circuit(physical, ignore_returns=True)
    cirq_measurements = [
        op
        for op in circuit.all_operations()
        if isinstance(op.gate, cirq.MeasurementGate)
    ]
    assert len(cirq_measurements) == 1
    assert len(cirq_measurements[0].qubits) == 14


def test_gemini_qid_rejects_negative_index_and_invalid_pin():
    with pytest.raises(ValueError, match="nonnegative"):
        GeminiQubit(-1, (0, 0, 0))
    with pytest.raises(ValueError, match="three integer coordinates"):
        GeminiQubit(0, (0, 1))  # type: ignore[arg-type]


def test_gemini_qid_requires_pin():
    with pytest.raises(TypeError, match="pin"):
        cast(Any, GeminiQubit)(0)


def test_gemini_qid_rejects_none_pin():
    with pytest.raises(ValueError, match="LineQubit"):
        GeminiQubit(0, cast(tuple[int, int, int], None))


@pytest.mark.parametrize("index", [0.5, "0", None])
def test_gemini_qid_rejects_noninteger_index(index: object):
    with pytest.raises(ValueError, match="integer"):
        GeminiQubit(cast(int, index), (0, 0, 0))


@pytest.mark.parametrize("pin", [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (-1, -1, -1)])
def test_gemini_qid_rejects_negative_pin(pin: tuple[int, int, int]):
    with pytest.raises(ValueError, match="nonnegative"):
        GeminiQubit(0, pin)


@pytest.mark.parametrize("pin", [(0, 1, 2.5), (0, "1", 2)])
def test_gemini_qid_rejects_noninteger_pin(pin: object):
    with pytest.raises(ValueError, match="three integer coordinates"):
        GeminiQubit(0, cast(tuple[int, int, int], pin))


def test_gemini_qid_normalizes_list_pin_to_immutable_tuple():
    pin = [0, 1, 2]
    qid = GeminiQubit(0, cast(tuple[int, int, int], pin))
    expected = GeminiQubit(0, (0, 1, 2))

    assert qid.pin == (0, 1, 2)
    assert qid == expected
    assert hash(qid) == hash(expected)
    assert len({qid, expected}) == 1

    pin[0] = 3
    assert qid.pin == (0, 1, 2)
    assert qid == expected
    assert hash(qid) == hash(expected)


def test_gemini_qid_ordering_uses_index_and_pin():
    ordered = [
        GeminiQubit(0, (0, 0, 0)),
        GeminiQubit(0, (0, 0, 1)),
        GeminiQubit(0, (0, 1, 0)),
        GeminiQubit(1, (0, 0, 0)),
    ]

    assert sorted([ordered[2], ordered[0], ordered[3], ordered[1]]) == ordered
    assert len(set(ordered)) == len(ordered)


def test_gemini_qid_has_readable_pinned_name():
    pinned = GeminiQubit(1, (0, 2, 3))

    assert str(pinned) == "Q1@(0,2,3)"
    assert repr(pinned) == "GeminiQubit(index=1, pin=(0, 2, 3))"
    assert GeminiLogicalQubit is GeminiQubit


@pytest.mark.parametrize(
    "dialects", [logical.kernel, physical.kernel], ids=["logical", "physical"]
)
@pytest.mark.parametrize(
    "qids",
    [
        (cirq.LineQubit(0), cirq.LineQubit(5), cirq.LineQubit(9)),
        (cirq.LineQubit(0), GeminiQubit(1, (0, 2, 3)), cirq.LineQubit(5)),
        (cirq.GridQubit(0, 0), cirq.GridQubit(3, 4), cirq.GridQubit(1, 5)),
        (cirq.GridQubit(0, 1), GeminiQubit(7, (0, 2, 3)), cirq.NamedQubit("anc")),
        (GeminiQubit(5, (0, 0, 0)), cirq.LineQubit(5), GeminiQubit(5, (0, 1, 0))),
    ],
    ids=[
        "sparse-lines",
        "sparse-pinned",
        "grids",
        "mixed-types",
        "shared-numeric-labels",
    ],
)
def test_gemini_loader_uses_standard_squin_qubit_order(
    dialects: ir.DialectGroup, qids: tuple[cirq.Qid, cirq.Qid, cirq.Qid]
):
    q0, q1, q2 = qids
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q2, q0, q1))
    indices = Squin(squin.kernel, circuit).qreg_index

    assert GeminiCirqLowerer(dialects, circuit).qreg_index == indices

    method = load_circuit(circuit, dialects=dialects)
    renamed = {
        qid: (
            GeminiQubit(index, qid.pin)
            if isinstance(qid, GeminiQubit)
            else cirq.LineQubit(index)
        )
        for qid, index in indices.items()
    }
    expected = cirq.Circuit(
        cirq.H(renamed[q0]),
        cirq.CX(renamed[q0], renamed[q1]),
        cirq.measure(renamed[q2], renamed[q0], renamed[q1]),
    )

    method.verify()
    method.verify_type()
    assert emit_circuit(method) == expected


def test_pinned_emission_rejects_noninteger_address():
    zone, word, site = (ir.TestValue() for _ in range(3))
    stmt = NewAt(zone, word, site)
    frame = EmitCirqFrame(stmt, entries={zone: 0, word: 1.5, site: 0}, qubit_index=0)

    with pytest.raises(
        interp.exceptions.InterpreterError, match="compile-time integers"
    ):
        _GeminiQubitCirqMethods().new_at(EmitCirq(), frame, stmt)


def test_emit_rejects_logical_initialize_statement():
    stmt = logical_ops.Initialize(*(ir.TestValue() for _ in range(4)))

    with pytest.raises(interp.exceptions.InterpreterError, match="initialize"):
        _LogicalCirqMethods().unsupported(EmitCirq(), EmitCirqFrame(stmt), stmt)
