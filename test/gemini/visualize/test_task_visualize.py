"""``task.visualize()`` drives the debuggers in ``bloqade.gemini.visualize``."""

from unittest.mock import MagicMock

import pytest
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.lanes.arch.gemini.physical import get_arch_spec

import bloqade.gemini.visualize as gemini_visualize
from bloqade.gemini.device import PhysicalSimulatorTask, GeminiLogicalSimulatorTask
from bloqade.gemini.visualize.initialize import InitializeCZ, initialize_operations


def _task(task_type, **attributes):
    task = object.__new__(task_type)
    object.__setattr__(task, "physical_move_kernel", MagicMock())
    object.__setattr__(task, "physical_arch_spec", get_arch_spec())
    for name, value in attributes.items():
        object.__setattr__(task, name, value)
    return task


@pytest.mark.parametrize(
    ("options", "debugger_name"),
    [
        ({}, "debugger"),
        ({"animated": True}, "animated_debugger"),
        ({"arch_vis": True}, "plotly_debugger"),
    ],
)
def test_visualize_calls_the_selected_debugger(monkeypatch, options, debugger_name):
    debuggers = {
        name: MagicMock()
        for name in ("debugger", "animated_debugger", "plotly_debugger")
    }
    for name, debugger in debuggers.items():
        monkeypatch.setattr(gemini_visualize, name, debugger)
    task = _task(PhysicalSimulatorTask)

    task.visualize(interactive=False, **options)

    for name, debugger in debuggers.items():
        assert debugger.call_count == (1 if name == debugger_name else 0)
    (mt, arch_spec), kwargs = debuggers[debugger_name].call_args
    assert mt is task.physical_move_kernel
    assert arch_spec is task.physical_arch_spec
    # A physical program has no ``move.PhysicalInitialize`` to step through,
    # and nothing is exported unless ``to_mp4`` is given.
    assert kwargs == {"interactive": False, "initialize_kernel": None, "to_mp4": None}


def test_logical_task_steps_through_its_state_preparation(monkeypatch):
    debugger = MagicMock()
    monkeypatch.setattr(gemini_visualize, "debugger", debugger)
    noise_model = generate_logical_noise_model()
    _, noisy = noise_model.get_logical_initialize()

    _task(GeminiLogicalSimulatorTask, noise_model=noise_model).visualize()

    assert debugger.call_args.kwargs["initialize_kernel"] is noisy


def test_state_preparation_kernel_moves_the_atoms_the_noise_model_moves():
    """The debuggers move the controls of each CZ; see ``steane7_initialize_with_noise``."""
    _, noisy = generate_logical_noise_model().get_logical_initialize()

    operations = initialize_operations(noisy, [0.0], [0.0], [0.0], 7)

    assert [
        (operation.controls, operation.targets)
        for operation in operations
        if isinstance(operation, InitializeCZ)
    ] == [
        ((2, 4, 6), (1, 3, 5)),
        ((0, 2, 4), (3, 5, 6)),
        ((1, 3, 5), (0, 2, 4)),
    ]
