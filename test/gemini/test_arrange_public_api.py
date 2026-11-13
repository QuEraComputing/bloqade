"""The Gemini movement API lowers to lanes without exposing dialect internals."""

import pytest
from kirin import ir
from bloqade.lanes.dialects.arch.stmts import NewAt
from bloqade.lanes.dialects.arrange.stmts import MoveTo, Permute

from bloqade import gemini


def test_public_movement_helpers_lower_to_lanes_statements():
    @gemini.physical.kernel(verify=False, typeinfer=False, fold=False, inline=False)
    def program(qubits, locations):
        gemini.move_to(qubits, locations)
        gemini.arrange.permute(qubits, [1, 0])

    statements = list(program.code.walk())
    assert sum(isinstance(stmt, MoveTo) for stmt in statements) == 1
    assert sum(isinstance(stmt, Permute) for stmt in statements) == 1


def test_arrange_exposes_frontend_operations_only():
    assert gemini.arrange.__all__ == ["permute"]
    for name in ("dialect", "stmts", "Len", "move_to"):
        assert not hasattr(gemini.arrange, name)
    assert not hasattr(gemini.common, "dialects")


@pytest.mark.parametrize("kernel", [gemini.logical.kernel, gemini.physical.kernel])
def test_public_new_at_lowers_to_lanes_statement(kernel: ir.DialectGroup):
    @kernel(
        verify=False,
        typeinfer=False,
        fold=False,
        inline=False,
        aggressive_unroll=False,
    )
    def program():
        return gemini.new_at(0, 1, 0)

    assert sum(isinstance(stmt, NewAt) for stmt in program.code.walk()) == 1
