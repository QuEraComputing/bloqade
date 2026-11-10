"""Public APIs remain available while compiler definitions belong to lanes."""

import sys
import subprocess

import pytest
from kirin.dialects import ilist
from kirin.serialization import JSONSerializer
from bloqade.lanes.dialects import arch, steane713
from bloqade.lanes.dialects.steane713 import extensions

from bloqade.gemini import logical
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.gemini.common.dialects.qubit.stmts import NewAt
from bloqade.gemini.logical.dialects.operations.stmts import TerminalLogicalMeasurement


def test_compatibility_aliases_keep_identity():
    assert new_at is arch.new_at
    assert NewAt is arch.NewAt
    assert TerminalLogicalMeasurement is steane713.stmts.TerminalLogicalMeasurement
    assert logical.terminal_measure is steane713.terminal_measure
    assert extensions.dialect is not steane713.dialect


def test_new_dialect_serialization_roundtrip():
    @logical.kernel(verify=False)
    def program():
        q = logical.qalloc_at(ilist.IList([2]))
        logical.extensions.star_rz(0.1, q[0])
        return logical.terminal_measure(q)

    serializer = JSONSerializer()
    payload = serializer.encode(program.dialects.encode(program, version="1.0.0"))
    restored = program.dialects.decode(serializer.decode(payload))
    assert (
        serializer.encode(restored.dialects.encode(restored, version="1.0.0"))
        == payload
    )


@pytest.mark.parametrize("first", ["bloqade.lanes.transform", "bloqade.gemini"])
def test_import_order(first):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({first!r}); import bloqade.gemini; import bloqade.lanes.transform",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("frontend", ["logical", "physical"])
def test_frontend_registers_method_tables(frontend):
    # A fresh process distinguishes explicit registration from test import order.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
from bloqade.lanes.dialects import arch, steane713, arrange
from bloqade import qubit
from bloqade.squin import gate
from kirin.dialects import func, scf
from bloqade.lanes.dialects.steane713 import extensions
requirements = [
    (arch.dialect, 'qubit.address'),
    (arch.dialect, 'lanes.arch.const_address'),
    (arch.dialect, 'lanes.arch.duplicates'),
    (arch.dialect, 'move.address.validation'),
    (arrange.dialect, 'move.address.validation'),
    (qubit.dialect, 'lanes.validate.physical.terminal_measure'),
    (steane713.dialect, 'typeinfer'),
    (steane713.dialect, 'measure_id'),
]
requirements += [(d, 'lanes.validate.logical') for d in
    (qubit.dialect, gate.dialect, scf.dialect, extensions.dialect)]
requirements += [(d, 'lanes.validate.terminal_measurement') for d in
    (qubit.dialect, gate.dialect, func.dialect, steane713.dialect, extensions.dialect)]
assert all(key not in dialect.interps for dialect, key in requirements)
import bloqade.gemini.{frontend}
for dialect, key in requirements:
    assert key in dialect.interps, (dialect.name, key)
    assert type(dialect.interps[key]).__module__.startswith('bloqade.gemini.')
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
