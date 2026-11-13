"""Public APIs remain available while compiler definitions belong to lanes."""

import sys
import subprocess

import pytest
from kirin.dialects import ilist
from kirin.serialization import JSONSerializer
from bloqade.lanes.dialects import arch, steane713
from bloqade.lanes.dialects.steane713 import extensions

from bloqade.gemini import new_at, logical


def test_public_frontend_exports_keep_identity():
    assert new_at is arch.new_at
    assert logical.terminal_measure is steane713.terminal_measure
    assert extensions.dialect is not steane713.dialect


def test_public_terminal_measure_lowers_to_steane_statement():
    @logical.kernel
    def program():
        q = logical.qalloc(1)
        return logical.terminal_measure(q)

    assert any(
        isinstance(stmt, steane713.stmts.TerminalLogicalMeasurement)
        for stmt in program.callable_region.walk()
    )


def test_new_dialect_serialization_roundtrip():
    @logical.kernel(verify=False)
    def program():
        q = logical.qalloc_at(ilist.IList([2]))
        logical.extensions.star_rz(0.1, q[0])
        return logical.terminal_measure(q)

    serializer = JSONSerializer()
    payload = serializer.encode(program.dialects.encode(program, version="1.0.0"))
    restored = program.dialects.decode(serializer.decode(payload))
    # Deserialization may intern equal type attributes, changing JSON reference
    # IDs without changing the IR. Compare the decoded program structurally.
    assert program.code.is_structurally_equal(restored.code)


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
def test_frontend_reuses_lanes_method_tables(frontend):
    # Importing either frontend must not replace compiler-owned registrations.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
from bloqade.lanes.dialects import arch, steane713, arrange
from bloqade.lanes.validation import (
    address, const_address, duplicate_address, steane, terminal_measure,
    terminal_measurement,
)
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
tables = [dialect.interps[key] for dialect, key in requirements]
assert all(type(table).__module__.startswith("bloqade.lanes.") for table in tables)
import bloqade.gemini.{frontend}
for (dialect, key), table in zip(requirements, tables):
    assert dialect.interps[key] is table, (dialect.name, key)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
