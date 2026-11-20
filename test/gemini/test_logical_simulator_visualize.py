"""Visualization entry points on compiled Gemini logical simulator tasks."""

from unittest.mock import Mock

import pytest
from bloqade.lanes.dialects import move

from bloqade import squin
from bloqade.gemini import logical
from bloqade.gemini.device import GeminiLogicalSimulator
from bloqade.gemini.visualize import plotly_debugger
from bloqade.gemini.visualize.artist import collect_debug_steps


@logical.kernel(aggressive_unroll=True)
def _one_logical_qubit():
    reg = squin.qalloc(1)
    squin.x(reg[0])
    return logical.terminal_measure(reg)


@pytest.fixture(scope="module")
def logical_task():
    return GeminiLogicalSimulator().task(_one_logical_qubit)


def test_logical_move_program_is_cached_and_interpretable(logical_task):
    logical_move = logical_task.logical_move_kernel

    assert logical_move is logical_task.logical_move_kernel
    assert logical_move is not logical_task.physical_move_kernel
    assert logical_task.logical_arch_spec is logical_task.logical_arch_spec
    fills = [
        stmt
        for stmt in logical_move.callable_region.walk()
        if isinstance(stmt, move.Fill)
    ]
    assert len(fills) == 1
    assert len(fills[0].location_addresses) == 1
    assert collect_debug_steps(logical_move, logical_task.logical_arch_spec)


def test_plotly_debugger_can_render_logical_move_program(logical_task):
    figure = plotly_debugger(
        logical_task.logical_move_kernel,
        logical_task.logical_arch_spec,
        show=False,
    )

    assert figure is not None
    assert figure.frames


@pytest.mark.parametrize(
    ("animated", "arch_vis", "selected"),
    [
        (False, False, "debugger"),
        (True, False, "animated_debugger"),
        (False, True, "plotly_debugger"),
        (True, True, "plotly_debugger"),
    ],
)
def test_visualize_logical_uses_logical_program_and_architecture(
    logical_task, monkeypatch, animated: bool, arch_vis: bool, selected: str
):
    import bloqade.gemini.visualize as visualize

    mocks = {
        "debugger": Mock(),
        "animated_debugger": Mock(),
        "plotly_debugger": Mock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(visualize, name, mock)

    logical_task.visualize_logical(
        animated=animated, interactive=False, arch_vis=arch_vis
    )

    mocks[selected].assert_called_once_with(
        logical_task.logical_move_kernel,
        logical_task.logical_arch_spec,
        interactive=False,
        # A logical program has no PhysicalInitialize to step through.
        initialize_kernel=None,
        to_mp4=None,
    )
    for name, mock in mocks.items():
        if name != selected:
            mock.assert_not_called()


def test_visualize_still_uses_physical_program_and_architecture(
    logical_task, monkeypatch
):
    import bloqade.gemini.visualize as visualize

    physical_debugger = Mock()
    monkeypatch.setattr(visualize, "debugger", physical_debugger)

    logical_task.visualize(interactive=False)

    physical_debugger.assert_called_once_with(
        logical_task.physical_move_kernel,
        logical_task.physical_arch_spec,
        interactive=False,
        initialize_kernel=logical_task._initialize_kernel,
        to_mp4=None,
    )


@pytest.mark.parametrize("logical", [False, True])
@pytest.mark.parametrize("arch_vis", [False, True])
def test_task_visualization_forwards_mp4_path(
    logical_task, monkeypatch, tmp_path, logical: bool, arch_vis: bool
):
    import bloqade.gemini.visualize as visualize

    selected = "plotly_debugger" if arch_vis else "animated_debugger"
    draw = Mock()
    monkeypatch.setattr(visualize, selected, draw)
    output = str(tmp_path / "moves.mp4")

    method = logical_task.visualize_logical if logical else logical_task.visualize
    method(animated=True, arch_vis=arch_vis, to_mp4=output)

    draw.assert_called_once_with(
        (
            logical_task.logical_move_kernel
            if logical
            else logical_task.physical_move_kernel
        ),
        logical_task.logical_arch_spec if logical else logical_task.physical_arch_spec,
        interactive=True,
        initialize_kernel=None if logical else logical_task._initialize_kernel,
        to_mp4=output,
    )
