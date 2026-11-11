import os
import sys
import textwrap
import subprocess

import cirq
import pytest
from kirin import lowering
from kirin.dialects import py
from bloqade.cirq_utils import emit_circuit, load_circuit
from bloqade.lanes.dialects.arch.stmts import NewAt

from bloqade.gemini import physical


def test_physical_pinned_qubit_round_trip():
    from bloqade.gemini.common.cirq_conversion import GeminiQubit

    pinned = GeminiQubit(0, pin=(1, 2, 3))
    circuit = cirq.Circuit(cirq.H(pinned), cirq.measure(pinned))

    method = load_circuit(circuit, dialects=physical.kernel)

    pins = [stmt for stmt in method.callable_region.walk() if isinstance(stmt, NewAt)]
    assert len(pins) == 1
    for coordinate, expected in zip(
        (pins[0].zone_id, pins[0].word_id, pins[0].site_id), (1, 2, 3), strict=True
    ):
        assert isinstance(coordinate.owner, py.Constant)
        assert coordinate.owner.value.unwrap() == expected
    assert emit_circuit(method) == circuit


def test_physical_unpinned_line_qubit_round_trip():
    unpinned = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(unpinned), cirq.measure(unpinned))

    method = load_circuit(circuit, dialects=physical.kernel)

    assert not any(isinstance(stmt, NewAt) for stmt in method.callable_region.walk())
    assert emit_circuit(method) == circuit


def test_physical_mixed_pinned_and_unpinned_qubit_round_trip():
    from bloqade.gemini.common.cirq_conversion import GeminiQubit

    pinned = GeminiQubit(0, (1, 2, 3))
    unpinned = cirq.LineQubit(1)
    circuit = cirq.Circuit(
        cirq.H(pinned), cirq.CX(pinned, unpinned), cirq.measure(pinned, unpinned)
    )

    method = load_circuit(circuit, dialects=physical.kernel)

    assert sum(isinstance(stmt, NewAt) for stmt in method.callable_region.walk()) == 1
    assert emit_circuit(method) == circuit


def test_physical_pinned_loader_rejects_register_as_argument():
    from bloqade.gemini.common.cirq_conversion import GeminiQubit

    pinned = GeminiQubit(0, pin=(1, 2, 3))
    circuit = cirq.Circuit(cirq.measure(pinned))

    with pytest.raises(lowering.BuildError, match="pinned"):
        load_circuit(circuit, dialects=physical.kernel, register_as_argument=True)


def test_physical_new_at_emits_without_importing_logical_converter(tmp_path):
    script = tmp_path / "physical_cirq_emission.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys

            from kirin.dialects import ilist
            from bloqade import squin
            from bloqade.cirq_utils import emit_circuit
            from bloqade.gemini import physical
            from bloqade.gemini import new_at

            @physical.kernel()
            def program():
                qubit = new_at(1, 2, 3)
                squin.h(qubit)
                return squin.broadcast.measure(ilist.IList([qubit]))

            assert "bloqade.gemini.logical.cirq_conversion" not in sys.modules
            circuit = emit_circuit(program, ignore_returns=True)
            assert "bloqade.gemini.logical.cirq_conversion" not in sys.modules
            print(repr(next(iter(circuit.all_qubits()))))
            """
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path)},
    )

    assert completed.returncode == 0, completed.stderr
    assert "GeminiQubit(index=0, pin=(1, 2, 3))" in completed.stdout
