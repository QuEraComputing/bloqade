"""Debug logical move programs (before transversal expansion)."""

import pytest
from bloqade.lanes.dialects import move, steane713
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.arch.gemini.logical import get_arch_spec

import bloqade.gemini as gemini
from bloqade import squin
from bloqade.gemini.visualize import plotly_debugger
from bloqade.gemini.visualize.artist import collect_debug_steps


@gemini.logical.kernel
def _one_logical_qubit():
    reg = squin.qalloc(1)
    squin.x(reg[0])
    return steane713.terminal_measure(reg)


@pytest.fixture(scope="module")
def logical_move_program():
    arch_spec = get_arch_spec()
    program = LogicalPipeline(arch_spec=arch_spec, transversal_rewrite=False).emit(
        _one_logical_qubit
    )
    return program, arch_spec


def test_logical_move_program_is_interpretable(logical_move_program):
    program, arch_spec = logical_move_program
    fills = [
        stmt for stmt in program.callable_region.walk() if isinstance(stmt, move.Fill)
    ]
    assert len(fills) == 1
    assert len(fills[0].location_addresses) == 1
    assert collect_debug_steps(program, arch_spec)


def test_plotly_debugger_can_render_logical_move_program(logical_move_program):
    program, arch_spec = logical_move_program
    figure = plotly_debugger(program, arch_spec, show=False)

    assert figure is not None
    assert figure.frames
