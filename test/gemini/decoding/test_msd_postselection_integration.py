from __future__ import annotations

import numpy as np

from bloqade.gemini.device import TsimSimulatorBackend, GeminiLogicalSimulator
from bloqade.gemini.decoding import (
    PostSelectionExperiment,
    TableDecoderWithConfidence,
    empty_logical_circuit,
    magic_state_dist_steane,
    single_qubit_state_tomography,
)
from bloqade.gemini.decoding.experiments import _basis_dataset_from_task_result

_TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
_DECODER_SEED = 10
_SIMULATOR_SEED = 10
_DECODER_TRAINING_SHOTS = 100
_SIMULATION_SHOTS = 100


def _sample_tasks_sequentially(
    experiment: PostSelectionExperiment,
    simulator: GeminiLogicalSimulator,
) -> None:
    """Sample X/Y/Z in a deterministic order for this numerical regression test.

    ``PostSelectionExperiment.get_samples`` submits all basis tasks concurrently.
    A shared seeded backend currently assigns its child seeds when each worker
    begins sampling, so that path is scheduling-dependent. This test exercises
    the same compiled tasks and post-processing synchronously instead.
    """
    tasks = experiment.make_tasks(simulator)
    experiment._postselection_exp_cache.raw_results = {
        basis: _basis_dataset_from_task_result(tasks[basis].run(_SIMULATION_SHOTS))
        for basis in ("X", "Y", "Z")
    }


def _run_seeded_distilled_msd() -> float:
    nonclifford_prefix, clifford_circuit = magic_state_dist_steane(theta_offset=0.30)
    experiment = PostSelectionExperiment(
        nonclifford_prefix,
        clifford_circuit,
        single_qubit_state_tomography(),
        TableDecoderWithConfidence,
        decoder_init_args={
            "seed": _DECODER_SEED,
            "num_shots": _DECODER_TRAINING_SHOTS,
        },
    )
    experiment.kernels(num_logical_qubits=5)
    experiment.dem_circuits()
    experiment.dems()
    experiment.initialize_decoders()
    _sample_tasks_sequentially(
        experiment,
        GeminiLogicalSimulator(backend=TsimSimulatorBackend(seed=_SIMULATOR_SEED)),
    )
    experiment.decode_and_postselect(np.array([[1, 0, 1, 1]], dtype=np.uint8))
    return experiment.tomography_result(0.50).fidelity_bloch(_TARGET_BLOCH)


def _run_seeded_injected_msd() -> float:
    nonclifford_prefix, _ = magic_state_dist_steane(theta_offset=0.30)
    experiment = PostSelectionExperiment(
        nonclifford_prefix,
        empty_logical_circuit(),
        single_qubit_state_tomography(),
        TableDecoderWithConfidence,
        decoder_init_args={
            "seed": _DECODER_SEED,
            "num_shots": _DECODER_TRAINING_SHOTS,
        },
    )
    experiment.kernels(num_logical_qubits=1)
    experiment.dem_circuits()
    experiment.dems()
    experiment.initialize_decoders()
    _sample_tasks_sequentially(
        experiment,
        GeminiLogicalSimulator(backend=TsimSimulatorBackend(seed=_SIMULATOR_SEED)),
    )
    experiment.decode_and_postselect(np.array([[]], dtype=np.uint8))
    return experiment.tomography_result(1.0).fidelity_bloch(_TARGET_BLOCH)


def test_seeded_msd_postselection_fidelities():
    distilled = _run_seeded_distilled_msd()
    repeated_distilled = _run_seeded_distilled_msd()
    injected = _run_seeded_injected_msd()

    assert repeated_distilled == distilled
    assert np.isfinite(distilled)
    assert np.isfinite(injected)
