import json

import numpy as np
import pandas as pd
from bloqade.visual.animation.gate_event import GateEvent
from bloqade.visual.animation.runtime.aod import AODMoveEvent
from bloqade.visual.animation.runtime.atoms import AtomTrajectory
from bloqade.visual.animation.runtime.ppoly import PPoly
from bloqade.plugins.qbraid.simulation_result import QuEraSimulationResult
from bloqade.visual.animation.runtime.qpustate import AnimateQPUState

from .sample.test_noise_model import get_noise_model


def run_json_test(obj):
    assert hasattr(obj, "to_json")
    assert hasattr(type(obj), "from_json")

    obj_events_json = json.dumps(obj.to_json())
    obj_events_reconstructed = type(obj).from_json(json.loads(obj_events_json))
    assert obj == obj_events_reconstructed


def test_PPoly():
    x = np.array([0, 1, 2, 3, 4])
    c = np.random.rand(4, 5)
    ppoly = PPoly(c, x)

    run_json_test(ppoly)


def test_GateEvents():
    gate_events = GateEvent("Test", {"Test": 1}, 10.0)
    run_json_test(gate_events)


def test_AtomTrajectory():
    x_x = np.array([0, 1, 2, 3, 4])
    x_c = np.random.rand(4, 5)
    x = PPoly(x_c, x_x)

    y_x = np.array([0, 1, 2, 3, 4])
    y_c = np.random.rand(4, 5)
    y = PPoly(y_c, y_x)

    events = [(0.0, "Test"), (1.0, "Test")]

    atom_trajectory = AtomTrajectory(1, x, y, events)

    run_json_test(atom_trajectory)


def test_AODMoveEvent():
    x_x = np.array([0, 1, 2, 3, 4])
    x_c = np.random.rand(4, 5)
    x = PPoly(x_c, x_x)

    y_x = np.array([0, 1, 2, 3, 4])
    y_c = np.random.rand(4, 5)
    y = PPoly(y_c, y_x)

    aod_move_event = AODMoveEvent(1.0, 1.0, x, y)

    run_json_test(aod_move_event)


def test_AnimateQPUState():
    x_x = np.array([0, 1, 2, 3, 4])
    x_c = np.random.rand(4, 5)
    x = PPoly(x_c, x_x)

    y_x = np.array([0, 1, 2, 3, 4])
    y_c = np.random.rand(4, 5)
    y = PPoly(y_c, y_x)

    animate_qpu_state = AnimateQPUState(
        block_durations=[5.0],
        gate_events=[(3.0, GateEvent("Test", {"Test": 1}, 10.0))],
        atoms=[AtomTrajectory(1, x, y, [(0.0, "Test")])],
        slm_zone=[(0.0, 0.0)],
        aod_moves=[AODMoveEvent(1.0, 1.0, x, y)],
    )

    run_json_test(animate_qpu_state)


def test_simulation_result():
    noise_model = get_noise_model()

    x_x = np.array([0, 1, 2, 3, 4])
    x_c = np.random.rand(4, 5)
    x = PPoly(x_c, x_x)

    y_x = np.array([0, 1, 2, 3, 4])
    y_c = np.random.rand(4, 5)
    y = PPoly(y_c, y_x)

    animate_qpu_state = AnimateQPUState(
        block_durations=[5.0],
        gate_events=[(3.0, GateEvent("Test", {"Test": 1}, 10.0))],
        atoms=[AtomTrajectory(1, x, y, [(0.0, "Test")])],
        slm_zone=[(0.0, 0.0)],
        aod_moves=[AODMoveEvent(1.0, 1.0, x, y)],
    )

    counts = {}

    logs = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    quera_simulation_result = QuEraSimulationResult(
        flair_visual_version="0.0.1",
        counts=counts,
        logs=logs,
        atom_animation_state=animate_qpu_state,
        noise_model=noise_model,
    )

    obj_events_json = json.dumps(quera_simulation_result.to_json())
    obj_events_reconstructed = type(quera_simulation_result).from_json(
        json.loads(obj_events_json)
    )

    assert quera_simulation_result.noise_model == obj_events_reconstructed.noise_model
    assert (
        quera_simulation_result.flair_visual_version
        == obj_events_reconstructed.flair_visual_version
    )
    assert quera_simulation_result.counts == obj_events_reconstructed.counts
    assert quera_simulation_result.logs.equals(obj_events_reconstructed.logs)
    assert (
        quera_simulation_result.atom_animation_state
        == obj_events_reconstructed.atom_animation_state
    )
