"""Long-running numerical regression for the seeded MSD demo workflow.

This test is selected only by the optional ``msd_numerical_regression`` CI job.
Its baselines are intentionally environment-specific because Stim's seeded sampling
is not portable across all machines and SIMD configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from bloqade.gemini.device import CliffTSimulatorBackend, GeminiLogicalSimulator
from bloqade.gemini.decoding import (
    PostSelectionExperiment,
    TableDecoder,
    empty_logical_circuit,
    magic_state_dist_steane,
    single_qubit_state_tomography,
)
from bloqade.gemini.decoding.experiments import _basis_dataset_from_task_result

_TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
_SEED = 10
_SIMULATION_SHOTS = 1_000_000
_EXPECTED_DISTILLED_FIDELITY = 0.9784569952548308
_EXPECTED_INJECTED_FIDELITY = 0.945448826690564
# Seeded Stim samples are stable enough for numerical-regression checks but not
# bit-identical across all supported runner configurations.
_FIDELITY_TOLERANCE = 1e-2


def _initialize_seeded_decoders(experiment: PostSelectionExperiment) -> None:
    # Upstream training is unseeded; supply deterministic samples via its public API.
    for pair in experiment.initialize_decoders().values():
        for decoder in pair:
            assert isinstance(decoder, TableDecoder)
            detectors, observables, _ = decoder.dem.compile_sampler(seed=_SEED).sample(
                10**7
            )
            decoder.update_det_obs_counts(
                np.concatenate([detectors, observables], axis=1)
            )


def _sample_tasks_sequentially(
    experiment: PostSelectionExperiment,
    simulator: GeminiLogicalSimulator,
) -> None:
    """Sample tomography bases in a deterministic child-seed order."""
    tasks = experiment.make_tasks(simulator)
    experiment._postselection_exp_cache.raw_results = {
        basis: _basis_dataset_from_task_result(tasks[basis].run(_SIMULATION_SHOTS))
        for basis in ("X", "Y", "Z")
    }


def _run_distilled_msd() -> float:
    nonclifford_prefix, clifford_circuit = magic_state_dist_steane(theta_offset=0.30)
    experiment = PostSelectionExperiment(
        nonclifford_prefix,
        clifford_circuit,
        single_qubit_state_tomography(),
        TableDecoder,
        decoder_init_args={"num_shots": 0},
    )
    experiment.kernels(num_logical_qubits=5)
    experiment.dem_circuits()
    experiment.dems()
    _initialize_seeded_decoders(experiment)
    _sample_tasks_sequentially(
        experiment,
        GeminiLogicalSimulator(backend=CliffTSimulatorBackend(seed=_SEED)),
    )
    experiment.decode_and_postselect(np.array([[1, 0, 1, 1]], dtype=np.uint8))
    return experiment.tomography_result(0.50).fidelity_bloch(_TARGET_BLOCH)


def _run_injected_msd() -> float:
    nonclifford_prefix, _ = magic_state_dist_steane(theta_offset=0.30)
    experiment = PostSelectionExperiment(
        nonclifford_prefix,
        empty_logical_circuit(),
        single_qubit_state_tomography(),
        TableDecoder,
        decoder_init_args={"num_shots": 0},
    )
    experiment.kernels(num_logical_qubits=1)
    experiment.dem_circuits()
    experiment.dems()
    _initialize_seeded_decoders(experiment)
    _sample_tasks_sequentially(
        experiment,
        GeminiLogicalSimulator(backend=CliffTSimulatorBackend(seed=_SEED)),
    )
    experiment.decode_and_postselect(np.array([[]], dtype=np.uint8))
    return experiment.tomography_result(1.0).fidelity_bloch(_TARGET_BLOCH)


@pytest.mark.msd_numerical_regression
def test_seeded_msd_postselection_numerical_regression():
    distilled = _run_distilled_msd()
    injected = _run_injected_msd()

    assert distilled == pytest.approx(
        _EXPECTED_DISTILLED_FIDELITY,
        abs=_FIDELITY_TOLERANCE,
    )
    assert injected == pytest.approx(
        _EXPECTED_INJECTED_FIDELITY,
        abs=_FIDELITY_TOLERANCE,
    )
