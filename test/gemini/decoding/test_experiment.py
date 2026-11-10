from __future__ import annotations

import math
from typing import Any, cast
from dataclasses import dataclass
from unittest.mock import Mock
from concurrent.futures import Future

import numpy as np
import pytest
from stim import DetectorErrorModel
from kirin.dialects import py, func
from bloqade.decoders import BaseDecoder
from bloqade.squin.gate import stmts as gate_stmts

from bloqade.gemini.device import DetectorResult, GeminiLogicalSimulator
from bloqade.gemini.decoding import (
    TomographyResult,
    PostSelectionExperiment,
    TableDecoderWithConfidence,
    GurobiDecoderWithConfidence,
    empty_logical_circuit,
    magic_state_dist_steane,
    single_qubit_state_tomography,
)
from bloqade.gemini.decoding.sampling import _BasisDataset
from bloqade.gemini.decoding.confidence import ConfidenceDecoder
from bloqade.gemini.decoding.experiments import _basis_dataset_from_task_result


def _invoke_names(kernel) -> list[str]:
    return [
        str(stmt.callee.sym_name)
        for stmt in kernel.callable_region.walk()
        if isinstance(stmt, func.Invoke) and stmt.callee.sym_name is not None
    ]


def _float_constants(kernel) -> list[float]:
    values: list[float] = []
    for stmt in kernel.callable_region.walk():
        value = cast(Any, stmt.value).data if isinstance(stmt, py.Constant) else None
        if isinstance(value, float):
            values.append(float(value))
    return values


def _gate_statements(kernel) -> list[gate_stmts.Gate]:
    return [
        stmt
        for stmt in kernel.callable_region.walk()
        if isinstance(stmt, gate_stmts.Gate)
    ]


def _u3_angles(kernel) -> tuple[float, float, float]:
    u3 = next(
        stmt for stmt in _gate_statements(kernel) if isinstance(stmt, gate_stmts.U3)
    )
    return (
        float(cast(Any, u3.theta.owner).value.data),
        float(cast(Any, u3.phi.owner).value.data),
        float(cast(Any, u3.lam.owner).value.data),
    )


@pytest.fixture(scope="module")
def msd_circuits():
    return magic_state_dist_steane(theta_offset=0.30)


@pytest.fixture(scope="module")
def tomography_circuits():
    return single_qubit_state_tomography()


@pytest.fixture(scope="module")
def msd_mld_exp(msd_circuits, tomography_circuits):
    nonclifford_prefix, clifford_circuit = msd_circuits
    return PostSelectionExperiment(
        nonclifford_prefix,
        clifford_circuit,
        tomography_circuits,
        TableDecoderWithConfidence,
        {
            "seed": 10,
            "num_shots": 100,
        },
    )


@pytest.fixture(scope="module")
def msd_mld_kernels(msd_mld_exp):
    return msd_mld_exp.kernels()


@pytest.fixture(scope="module")
def msd_mld_dem_circuits(msd_mld_exp, msd_mld_kernels):
    _ = msd_mld_kernels
    return msd_mld_exp.dem_circuits()


@pytest.fixture(scope="module")
def msd_mld_dems(msd_mld_exp, msd_mld_dem_circuits):
    _ = msd_mld_dem_circuits
    return msd_mld_exp.dems()


@pytest.fixture(scope="module")
def msd_mld_decoders(msd_mld_exp, msd_mld_dems):
    _ = msd_mld_dems
    return msd_mld_exp.initialize_decoders()


@pytest.fixture(scope="module")
def msd_mld_samples(msd_mld_exp, msd_mld_kernels):
    _ = msd_mld_kernels
    msd_mld_exp.make_tasks(GeminiLogicalSimulator())
    return msd_mld_exp.get_samples(num_shots=100)


@pytest.fixture(scope="module")
def msd_mld_decoded(msd_mld_exp, msd_mld_decoders, msd_mld_samples):
    _ = msd_mld_decoders, msd_mld_samples
    return msd_mld_exp.decode_and_postselect(
        np.array([[1, 0, 1, 1]], dtype=np.uint8),
        progress_label="MLD",
    )


def test_magic_state_dist_steane_default_angles_in_ir():
    nonclifford_prefix, _ = magic_state_dist_steane()

    expected_angles = [
        math.acos(1 / math.sqrt(3)),
        0.25 * math.pi,
        0.0,
    ]

    np.testing.assert_allclose(_float_constants(nonclifford_prefix), expected_angles)


def test_magic_state_dist_steane_clifford_has_three_cz_invokes():
    _, clifford_circuit = magic_state_dist_steane()

    assert _invoke_names(clifford_circuit).count("cz") == 3


def test_magic_state_dist_steane_offset_angles_in_ir():
    nonclifford_prefix, _ = magic_state_dist_steane(
        theta_offset=0.1,
        phi_offset=0.2,
        lam_offset=0.3,
    )

    expected_angles = [
        math.acos(1 / math.sqrt(3)) + 0.1,
        0.25 * math.pi + 0.2,
        0.3,
    ]

    np.testing.assert_allclose(_float_constants(nonclifford_prefix), expected_angles)


def test_single_qubit_state_tomography_keys(tomography_circuits):
    assert set(tomography_circuits) == {"X", "Y", "Z"}


def test_single_qubit_state_tomography_ir(tomography_circuits):
    assert _invoke_names(tomography_circuits["X"]) == ["h"]
    assert _invoke_names(tomography_circuits["Y"]) == ["sqrt_z_adj", "h"]
    assert _invoke_names(tomography_circuits["Z"]) == []


def test_empty_logical_circuit_has_no_non_return_ir():
    statements = list(empty_logical_circuit().callable_region.walk())

    assert [type(stmt) for stmt in statements] == [func.ConstantNone, func.Return]


def test_postselection_experiment_kernels_sets_cache(msd_mld_exp, msd_mld_kernels):
    assert msd_mld_exp._postselection_exp_cache.dem_kernels is msd_mld_kernels


def _new_mld_experiment(msd_circuits, tomography_circuits) -> PostSelectionExperiment:
    nonclifford_prefix, clifford_circuit = msd_circuits
    return PostSelectionExperiment(
        nonclifford_prefix,
        clifford_circuit,
        tomography_circuits,
        TableDecoderWithConfidence,
        {
            "seed": 10,
            "num_shots": 10,
        },
    )


def test_postselection_experiment_dem_circuits_requires_kernels(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError, match="kernels must be called before dem_circuits"
    ):
        exp.dem_circuits()


def test_postselection_experiment_dems_requires_dem_circuits(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(RuntimeError, match="dem_circuits must be called before dems"):
        exp.dems()


def test_postselection_experiment_initialize_decoders_requires_dems(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError, match="dems must be called before initialize_decoders"
    ):
        exp.initialize_decoders()


def test_postselection_experiment_make_tasks_requires_kernels(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(RuntimeError, match="kernels must be called before make_tasks"):
        exp.make_tasks(GeminiLogicalSimulator())


def test_postselection_experiment_get_samples_requires_make_tasks(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError, match="make_tasks must be called before get_samples"
    ):
        exp.get_samples(num_shots=10)


def test_postselection_experiment_decode_requires_samples(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError, match="get_samples must be called before decoding"
    ):
        exp.decode_and_postselect(np.array([[1, 0, 1, 1]]))


def test_postselection_experiment_decode_requires_decoders_after_samples(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)
    exp._postselection_exp_cache.raw_results = {
        "X": _BasisDataset(
            detectors=np.zeros((2, 15), dtype=np.uint8),
            observables=np.zeros((2, 5), dtype=np.uint8),
        )
    }

    with pytest.raises(
        RuntimeError,
        match="initialize_decoders must be called before decoding",
    ):
        exp.decode_and_postselect(np.array([[1, 0, 1, 1]]))


def test_postselection_experiment_analysis_requires_decode(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError, match="decode_and_postselect must be called before analysis"
    ):
        exp.analysis_f_vs_fraction()


def test_postselection_experiment_tomography_result_requires_decode(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError,
        match="decode_and_postselect must be called before tomography_result",
    ):
        exp.tomography_result(accepted_fraction=1.0)


def test_postselection_experiment_visualization_requires_analysis(
    msd_circuits,
    tomography_circuits,
):
    exp = _new_mld_experiment(msd_circuits, tomography_circuits)

    with pytest.raises(
        RuntimeError,
        match="analysis_f_vs_fraction must be called before visualization",
    ):
        exp.analysis_visualization()


def test_postselection_experiment_kernels_basis_keys_and_tomography_suffix(
    msd_mld_kernels,
):
    assert set(msd_mld_kernels) == {"X", "Y", "Z"}

    x_gates = _gate_statements(msd_mld_kernels["X"])
    y_gates = _gate_statements(msd_mld_kernels["Y"])
    z_gates = _gate_statements(msd_mld_kernels["Z"])

    assert isinstance(x_gates[-1], gate_stmts.H)
    assert isinstance(y_gates[-2], gate_stmts.S)
    assert y_gates[-2].adjoint is True
    assert isinstance(y_gates[-1], gate_stmts.H)
    assert not isinstance(z_gates[-1], gate_stmts.H)


def test_postselection_experiment_kernels_return_tuple(msd_mld_kernels):
    for kernel in msd_mld_kernels.values():
        statements = list(kernel.callable_region.walk())
        assert isinstance(statements[-2], py.tuple.New)
        assert isinstance(statements[-1], func.Return)


def test_postselection_experiment_kernels_have_u3_and_three_cz(msd_mld_kernels):
    for kernel in msd_mld_kernels.values():
        gates = _gate_statements(kernel)
        assert sum(isinstance(stmt, gate_stmts.U3) for stmt in gates) == 1
        assert sum(isinstance(stmt, gate_stmts.CZ) for stmt in gates) == 3


def test_postselection_experiment_dem_circuits_have_zero_reference_signs(
    msd_mld_dem_circuits,
):
    for dem_circ in msd_mld_dem_circuits.values():
        detector_signs, observable_signs = (
            dem_circ._stim_circ.reference_detector_and_observable_signs()
        )

        assert not np.any(detector_signs)
        assert not np.any(observable_signs)


def test_postselection_experiment_dems_have_expected_shape(msd_mld_dems):
    for dem in msd_mld_dems.values():
        assert dem.num_detectors == 15
        assert dem.num_observables == 5


def _assert_decoders_validate_detector_width(
    decoders: dict[str, tuple[ConfidenceDecoder, BaseDecoder]],
) -> None:
    factory_decoder, full_decoder = decoders["X"]
    factory_detector_count = int(cast(Any, factory_decoder).num_detectors)
    full_detector_count = int(cast(Any, full_decoder).num_detectors)

    for detector_bits in (
        np.zeros(factory_detector_count - 1, dtype=np.bool_),
        np.zeros(factory_detector_count + 1, dtype=np.bool_),
    ):
        with pytest.raises(ValueError, match="decode_with_confidence expects"):
            factory_decoder.decode_with_confidence(detector_bits)

    for detector_bits in (
        np.zeros(full_detector_count - 1, dtype=np.bool_),
        np.zeros(full_detector_count + 1, dtype=np.bool_),
    ):
        with pytest.raises(ValueError, match="decode expects"):
            full_decoder.decode(detector_bits)


def test_postselection_experiment_initialize_mld_decoders_validate_width(
    msd_mld_decoders,
):
    _assert_decoders_validate_detector_width(msd_mld_decoders)


def test_postselection_experiment_initialize_mle_decoders_validate_width(
    msd_circuits,
    tomography_circuits,
    msd_mld_dems,
):
    pytest.importorskip("gurobipy")
    nonclifford_prefix, clifford_circuit = msd_circuits
    exp = PostSelectionExperiment(
        nonclifford_prefix,
        clifford_circuit,
        tomography_circuits,
        GurobiDecoderWithConfidence,
    )
    exp._postselection_exp_cache.dems = msd_mld_dems

    _assert_decoders_validate_detector_width(exp.initialize_decoders())


def test_postselection_experiment_make_tasks_sets_cache(msd_mld_exp, msd_mld_kernels):
    _ = msd_mld_kernels
    tasks = msd_mld_exp.make_tasks(GeminiLogicalSimulator())

    assert msd_mld_exp._postselection_exp_cache.hardware_tasks is tasks


@dataclass
class _FakeTask:
    run_async: Mock


def test_basis_dataset_from_detector_result_preserves_samples():
    result = DetectorResult[Any](
        _detector_error_model=DetectorErrorModel(),
        _fidelity_min=0.9,
        _fidelity_max=1.0,
        _detectors=[[True, False], [False, True]],
        _observables=[[False], [True]],
    )

    dataset = _basis_dataset_from_task_result(result)

    np.testing.assert_array_equal(
        dataset.detectors,
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        dataset.observables,
        np.array([[0], [1]], dtype=np.uint8),
    )


def test_postselection_experiment_get_samples_calls_run_async_once_per_basis():
    exp = object.__new__(PostSelectionExperiment)
    cast(Any, exp)._postselection_exp_cache = type(
        "Cache",
        (),
        {
            "hardware_tasks": {},
            "raw_results": None,
        },
    )()
    dataset = _BasisDataset(
        detectors=np.zeros((100, 15), dtype=np.uint8),
        observables=np.zeros((100, 5), dtype=np.uint8),
    )
    tasks: dict[str, _FakeTask] = {}
    for basis in ("X", "Y", "Z"):
        future: Future[_BasisDataset] = Future()
        future.set_result(dataset)
        tasks[basis] = _FakeTask(run_async=Mock(return_value=future))
    cast(Any, exp._postselection_exp_cache).hardware_tasks = tasks

    exp.get_samples(num_shots=100)

    for task in tasks.values():
        task.run_async.assert_called_once_with(100)


def test_postselection_experiment_get_samples_shapes(msd_mld_samples):
    for dataset in msd_mld_samples.values():
        assert dataset.detectors.shape == (100, 15)
        assert dataset.observables.shape == (100, 5)


def test_postselection_experiment_decode_and_postselect_shapes(msd_mld_decoded):
    for result in msd_mld_decoded.values():
        assert result.observables.shape[1] == 1
        assert result.confidence.ndim == 1
        assert result.confidence.shape[0] == result.observables.shape[0]


def test_postselection_experiment_decode_and_postselect_reduces_shots(
    msd_mld_exp,
    msd_mld_decoded,
):
    raw_results = msd_mld_exp._postselection_exp_cache.raw_results
    assert raw_results is not None

    for basis, result in msd_mld_decoded.items():
        assert result.observables.shape[0] == result.confidence.shape[0]
        assert result.observables.shape[0] < raw_results[basis].observables.shape[0]


def test_postselection_experiment_decode_and_postselect_value_types(msd_mld_decoded):
    for result in msd_mld_decoded.values():
        assert set(np.unique(result.observables)).issubset({0, 1})
        assert np.issubdtype(result.confidence.dtype, np.floating)


def test_postselection_experiment_analysis_curve_shapes(msd_mld_exp, msd_mld_decoded):
    _ = msd_mld_decoded
    curve = msd_mld_exp.analysis_f_vs_fraction(
        target_bloch=np.ones(3, dtype=np.float64) / np.sqrt(3.0),
        min_accepted_per_basis=0,
    )

    assert curve.accepted_fraction.shape == curve.fidelity.shape


def test_postselection_experiment_tomography_result_matches_full_curve_point(
    msd_mld_exp,
    msd_mld_decoded,
):
    _ = msd_mld_decoded
    target_bloch = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
    curve = msd_mld_exp.analysis_f_vs_fraction(
        target_bloch=target_bloch,
        min_accepted_per_basis=0,
    )

    point = msd_mld_exp.tomography_result(1.0).fidelity_bloch(target_bloch)

    assert point == pytest.approx(curve.fidelity[-1])


def test_tomography_result_rejects_bloch_vector_with_norm_greater_than_one():
    result = TomographyResult(
        {
            "X": np.array([[0], [1]], dtype=np.uint8),
            "Y": np.array([[0], [1]], dtype=np.uint8),
            "Z": np.array([[0], [1]], dtype=np.uint8),
        }
    )

    with pytest.raises(ValueError, match="squared norm <= 1"):
        result.fidelity_bloch(np.array([2.0, 0.0, 0.0]))
