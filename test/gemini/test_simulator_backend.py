from __future__ import annotations

import gc
import sys
import inspect
import weakref
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from kirin import ir
from kirin.dialects import func

from bloqade import squin
from bloqade.gemini.device import (
    BackendSample,
    PPVMSimulatorBackend,
    TsimSimulatorBackend,
    CliffTSimulatorBackend,
    AbstractSimulatorBackend,
)
from bloqade.gemini.device.simulator_backend import (
    _ppvm,
    _get_tsim_circuit,
    _PyQrackSimulatorBackend,
    _ppvm_compatible_stim_text,
    _clifft_compatible_stim_text,
)


@squin.kernel
def _physical_kernel():
    reg = squin.qalloc(1)
    return squin.broadcast.measure(reg)


@squin.kernel
def _random_physical_kernel():
    reg = squin.qalloc(1)
    squin.h(reg[0])
    return squin.broadcast.measure(reg)


@squin.kernel
def _one_physical_kernel():
    reg = squin.qalloc(1)
    squin.x(reg[0])
    return squin.broadcast.measure(reg)


@squin.kernel
def _loss_physical_kernel():
    reg = squin.qalloc(8)
    squin.broadcast.qubit_loss(0.5, reg)
    return squin.broadcast.measure(reg)


def test_backend_contract_has_exactly_two_abstract_operations():
    assert AbstractSimulatorBackend.__abstractmethods__ == {
        "sample",
        "_detector_error_model",
    }


@pytest.mark.parametrize(
    "backend_type",
    [
        AbstractSimulatorBackend,
        TsimSimulatorBackend,
        CliffTSimulatorBackend,
        PPVMSimulatorBackend,
        _PyQrackSimulatorBackend,
    ],
)
def test_backend_sample_does_not_expose_detector_mode(backend_type):
    assert "run_detectors" not in inspect.signature(backend_type.sample).parameters
    assert "seed" not in inspect.signature(backend_type.sample).parameters


@pytest.mark.parametrize(
    "backend_type",
    [
        TsimSimulatorBackend,
        CliffTSimulatorBackend,
        PPVMSimulatorBackend,
        _PyQrackSimulatorBackend,
    ],
)
def test_backends_expose_only_private_detector_error_model(backend_type):
    backend = backend_type()
    assert callable(backend._detector_error_model)
    assert not hasattr(backend, "detector_error_model")


def test_backend_sample_defaults_to_no_payloads():
    assert BackendSample() == BackendSample(
        measurements=None, detectors=None, observables=None
    )


def test_tsim_backend_detector_mode_defaults_to_false():
    assert TsimSimulatorBackend().run_detectors is False


def test_tsim_backend_accepts_detector_mode_configuration():
    assert TsimSimulatorBackend(run_detectors=True).run_detectors is True


@pytest.mark.parametrize(
    "backend_type",
    [CliffTSimulatorBackend, PPVMSimulatorBackend, _PyQrackSimulatorBackend],
)
def test_composite_backend_exposes_only_private_tsim_backend(backend_type):
    backend = backend_type()
    assert isinstance(backend._tsim_backend, TsimSimulatorBackend)
    assert not hasattr(backend, "tsim_backend")


@pytest.mark.parametrize(
    "backend_type",
    [
        TsimSimulatorBackend,
        CliffTSimulatorBackend,
        PPVMSimulatorBackend,
        _PyQrackSimulatorBackend,
    ],
)
@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_backends_reject_invalid_public_seed(backend_type, seed):
    with pytest.raises(ValueError, match="seed must be"):
        backend_type(seed=seed)


def test_tsim_circuit_cache_reuses_live_kernel_and_drops_dead_kernel(monkeypatch):
    class Kernel:
        def similar(self):
            return _physical_kernel.similar()

    circuit_factory = MagicMock(side_effect=lambda _: object())
    monkeypatch.setattr(
        "bloqade.gemini.device.simulator_backend._tsim",
        lambda: SimpleNamespace(Circuit=circuit_factory),
    )
    backend = TsimSimulatorBackend()
    first_kernel = Kernel()
    second_kernel = Kernel()

    first_circuit = backend._tsim_circuit(cast(ir.Method, first_kernel))
    assert backend._tsim_circuit(cast(ir.Method, first_kernel)) is first_circuit
    assert backend._tsim_circuit(cast(ir.Method, second_kernel)) is not first_circuit
    assert circuit_factory.call_count == 2
    assert len(backend._circuits) == 2

    first_ref = weakref.ref(first_kernel)
    del first_kernel
    gc.collect()

    assert first_ref() is None
    assert len(backend._circuits) == 1


def test_tsim_conversion_removes_return_from_owned_kernel(monkeypatch):
    converted = []
    monkeypatch.setattr(
        "bloqade.gemini.device.simulator_backend._tsim",
        lambda: SimpleNamespace(
            Circuit=lambda kernel: converted.append(kernel) or object()
        ),
    )
    source_ir = _physical_kernel.print_str()

    TsimSimulatorBackend()._tsim_circuit(_physical_kernel)

    assert _physical_kernel.print_str() == source_ir
    converted_return = converted[0].callable_region.blocks[0].last_stmt
    assert isinstance(converted_return, func.Return)
    assert isinstance(converted_return.value.owner, func.ConstantNone)


def _backend_with_circuit(
    circuit: MagicMock, *, seed: int | None = None, run_detectors: bool = False
) -> TsimSimulatorBackend:
    backend = TsimSimulatorBackend(seed=seed, run_detectors=run_detectors)
    backend._tsim_circuit = MagicMock(return_value=circuit)  # type: ignore[method-assign]
    return backend


def _derived_seeds(seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 2**63)) for _ in range(count)]


def test_tsim_clifford_measurement_sampling_uses_stim():
    circuit = MagicMock(is_clifford=True)
    circuit.stim_circuit.compile_sampler.return_value.sample.return_value = [[0, 1]]
    backend = _backend_with_circuit(circuit)

    sample = backend.sample(_physical_kernel, shots=1)

    assert sample.measurements is not None
    assert np.array_equal(sample.measurements, [[False, True]])
    circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=None)
    circuit.compile_sampler.assert_not_called()


def test_tsim_nonclifford_measurement_sampling_uses_tsim():
    circuit = MagicMock(is_clifford=False)
    circuit.compile_sampler.return_value.sample.return_value = [[1]]
    backend = _backend_with_circuit(circuit)

    sample = backend.sample(_physical_kernel, shots=1)

    assert sample.measurements is not None
    assert np.array_equal(sample.measurements, [[True]])
    circuit.compile_sampler.assert_called_once_with(seed=None)
    circuit.stim_circuit.compile_sampler.assert_not_called()


@pytest.mark.parametrize("is_clifford", [True, False], ids=["stim", "tsim"])
def test_tsim_backend_seed_derives_child_seed_for_measurement_sampling(is_clifford):
    circuit = MagicMock(is_clifford=is_clifford)
    sampler = (
        circuit.stim_circuit.compile_sampler.return_value
        if is_clifford
        else circuit.compile_sampler.return_value
    )
    sampler.sample.return_value = [[1]]
    backend = _backend_with_circuit(circuit, seed=0)

    backend.sample(_physical_kernel, shots=1)

    expected_seed = _derived_seeds(0, 1)[0]
    if is_clifford:
        circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=expected_seed)
        circuit.compile_sampler.assert_not_called()
    else:
        circuit.compile_sampler.assert_called_once_with(seed=expected_seed)
        circuit.stim_circuit.compile_sampler.assert_not_called()


def test_tsim_clifford_detector_sampling_uses_measurement_converter():
    circuit = MagicMock(is_clifford=True)
    measurements = np.array([[True, False]])
    circuit.stim_circuit.compile_sampler.return_value.sample.return_value = measurements
    converter = circuit.compile_m2d_converter.return_value
    converter.convert.return_value = (np.array([[1]]), np.array([[0]]))
    backend = _backend_with_circuit(circuit, run_detectors=True)

    sample = backend.sample(_physical_kernel, shots=1)

    circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=None)
    circuit.compile_m2d_converter.assert_called_once_with(skip_reference_sample=True)
    converter.convert.assert_called_once_with(
        measurements=measurements, separate_observables=True
    )
    assert sample.detectors is not None
    assert sample.observables is not None
    assert np.array_equal(sample.detectors, [[True]])
    assert np.array_equal(sample.observables, [[False]])


def test_tsim_nonclifford_detector_sampling_uses_detector_sampler():
    circuit = MagicMock(is_clifford=False)
    sampler = circuit.compile_detector_sampler.return_value
    sampler.sample.return_value = (np.array([[0]]), np.array([[1]]))
    backend = _backend_with_circuit(circuit, run_detectors=True)

    sample = backend.sample(_physical_kernel, shots=1)

    circuit.compile_detector_sampler.assert_called_once_with(seed=None)
    sampler.sample.assert_called_once_with(shots=1, separate_observables=True)
    assert sample.detectors is not None
    assert sample.observables is not None
    assert np.array_equal(sample.detectors, [[False]])
    assert np.array_equal(sample.observables, [[True]])


@pytest.mark.parametrize("is_clifford", [True, False], ids=["stim", "tsim"])
def test_tsim_backend_seed_derives_child_seed_for_detector_sampling(is_clifford):
    circuit = MagicMock(is_clifford=is_clifford)
    backend = _backend_with_circuit(circuit, seed=0, run_detectors=True)
    if is_clifford:
        sampler = circuit.stim_circuit.compile_sampler.return_value
        sampler.sample.return_value = [[1]]
        circuit.compile_m2d_converter.return_value.convert.return_value = (
            [[1]],
            [[0]],
        )
    else:
        sampler = circuit.compile_detector_sampler.return_value
        sampler.sample.return_value = ([[1]], [[0]])

    backend.sample(_physical_kernel, shots=1)

    expected_seed = _derived_seeds(0, 1)[0]
    if is_clifford:
        circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=expected_seed)
        circuit.compile_detector_sampler.assert_not_called()
    else:
        circuit.compile_detector_sampler.assert_called_once_with(seed=expected_seed)
        circuit.stim_circuit.compile_sampler.assert_not_called()


def test_tsim_backend_seed_advances_child_seed_between_batches():
    circuit = MagicMock(is_clifford=True)
    circuit.stim_circuit.compile_sampler.return_value.sample.return_value = [[1]]
    backend = _backend_with_circuit(circuit, seed=17)

    backend.sample(_physical_kernel, shots=1)
    backend.sample(_physical_kernel, shots=1)

    assert circuit.stim_circuit.compile_sampler.call_args_list == [
        call(seed=child_seed) for child_seed in _derived_seeds(17, 2)
    ]


def test_tsim_seeded_backends_reproduce_batch_sequence():
    pytest.importorskip("tsim")
    first_backend = TsimSimulatorBackend(seed=17)
    second_backend = TsimSimulatorBackend(seed=17)

    first = first_backend.sample(_random_physical_kernel, shots=64)
    second = first_backend.sample(_random_physical_kernel, shots=64)
    matching_first = second_backend.sample(_random_physical_kernel, shots=64)
    matching_second = second_backend.sample(_random_physical_kernel, shots=64)

    assert first.measurements is not None
    assert second.measurements is not None
    assert matching_first.measurements is not None
    assert matching_second.measurements is not None
    assert np.array_equal(first.measurements, matching_first.measurements)
    assert np.array_equal(second.measurements, matching_second.measurements)


def test_tsim_detector_error_model_and_structural_capability():
    circuit = MagicMock()
    circuit.detector_error_model.return_value = "dem"
    backend = _backend_with_circuit(circuit)

    assert _get_tsim_circuit(backend, _physical_kernel) is circuit
    assert backend._detector_error_model(_physical_kernel) == "dem"
    circuit.detector_error_model.assert_called_once_with(
        approximate_disjoint_errors=True
    )


def _fake_clifft(monkeypatch, *, sample_result=None):
    clifft = ModuleType("clifft")
    clifft.compile = MagicMock(return_value="program")  # type: ignore[attr-defined]
    clifft.sample = MagicMock(  # type: ignore[attr-defined]
        return_value=sample_result
        or SimpleNamespace(
            measurements=np.array([[0, 1]], dtype=np.uint8),
            detectors=np.array([[1]], dtype=np.uint8),
            observables=np.array([[0]], dtype=np.uint8),
        )
    )
    monkeypatch.setitem(sys.modules, "clifft", clifft)
    return clifft


def test_clifft_compatible_stim_text_strips_instruction_tags_only():
    assert (
        _clifft_compatible_stim_text(
            "I_ERROR[loss](0)\nDETECTOR[coords](1)\n# [comment]"
        )
        == "I_ERROR(0)\nDETECTOR(1)\n# [comment]"
    )


def test_ppvm_compatible_stim_text_converts_paired_correlated_loss_events():
    stim_text = (
        "I_ERROR[correlated_loss:7](0.1) 0 1\n"
        "  I_ERROR[correlated_loss:8](0.2, 0.03, 0.04) 2 3 # paired loss\n"
        "I_ERROR[loss](0.05) 4"
    )

    assert _ppvm_compatible_stim_text(stim_text) == (
        "I_ERROR[correlated_loss](0.1) 0 1\n"
        "  I_ERROR[correlated_loss](0.2, 0.03, 0.04) 2 3 # paired loss\n"
        "I_ERROR[loss](0.05) 4"
    )


@pytest.mark.parametrize("targets", ["0", "0 1 2", "0 1 2 3"])
def test_ppvm_compatible_stim_text_rejects_non_pair_events(targets):
    stim_text = f"I_ERROR[correlated_loss:4](0.1) {targets}"

    with pytest.raises(ValueError, match="exactly two qubits"):
        _ppvm_compatible_stim_text(stim_text)


def _fake_ppvm(monkeypatch, *, samples):
    ppvm = ModuleType("ppvm")
    ppvm.StimProgram = SimpleNamespace(  # type: ignore[attr-defined]
        parse=MagicMock(return_value=SimpleNamespace(num_qubits=5))
    )
    ppvm.MeasurementResult = SimpleNamespace(  # type: ignore[attr-defined]
        ZERO=0,
        ONE=1,
        LOST=2,
    )
    ppvm.sample_stim = MagicMock(return_value=samples)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ppvm", ppvm)
    return ppvm


def test_ppvm_missing_dependency_has_installation_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "ppvm", None)

    with pytest.raises(
        ImportError,
        match=r"bloqade-ppvm.*required dependency of `bloqade`.*Reinstall `bloqade-ppvm`",
    ):
        _ppvm()


def test_ppvm_backend_caches_program_maps_losses_and_derives_child_seed(monkeypatch):
    ppvm = _fake_ppvm(monkeypatch, samples=[[0, 1, 2], [2, 0, 1]])
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = SimpleNamespace(
        stim_circuit="I_ERROR[correlated_loss:9](0.1) 0 1\nM 0 1 2"
    )
    backend = PPVMSimulatorBackend(seed=123, _tsim_backend=tsim_backend)

    first = backend.sample(_physical_kernel, shots=2)
    second = backend.sample(_physical_kernel, shots=2)

    assert first.measurements is not None
    assert first.measurements.dtype == object
    assert first.measurements.tolist() == [[False, True, None], [None, False, True]]
    assert second.measurements is not None
    ppvm.StimProgram.parse.assert_called_once_with(  # type: ignore[attr-defined]
        "I_ERROR[correlated_loss](0.1) 0 1\nM 0 1 2"
    )
    assert ppvm.sample_stim.call_args_list == [  # type: ignore[attr-defined]
        call(
            ppvm.StimProgram.parse.return_value,  # type: ignore[attr-defined]
            n_qubits=5,
            num_shots=2,
            seed=child_seed,
        )
        for child_seed in _derived_seeds(123, 2)
    ]
    assert tsim_backend._tsim_circuit.call_count == 1


def test_ppvm_backend_converts_lost_measurements_in_place():
    raw_measurements = np.array([[False, None], [None, True]], dtype=object)

    converted_measurements = PPVMSimulatorBackend._convert_loss_to_measurement(
        raw_measurements,
        loss_replace=True,
        loss=None,
    )

    assert raw_measurements.tolist() == [[False, True], [True, True]]
    assert converted_measurements.dtype == bool
    assert converted_measurements.tolist() == [[False, True], [True, True]]


def test_ppvm_backend_omits_unconfigured_seed_and_returns_measurements(monkeypatch):
    ppvm = _fake_ppvm(monkeypatch, samples=[[1]])
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = SimpleNamespace(stim_circuit="M 0")
    backend = PPVMSimulatorBackend(_tsim_backend=tsim_backend)

    sample = backend.sample(_physical_kernel, shots=1)

    assert sample.measurements is not None
    assert sample.measurements.tolist() == [[True]]
    ppvm.sample_stim.assert_called_once_with(  # type: ignore[attr-defined]
        ppvm.StimProgram.parse.return_value,  # type: ignore[attr-defined]
        n_qubits=5,
        num_shots=1,
        seed=None,
    )
    assert sample.detectors is None
    assert sample.observables is None


def test_ppvm_backend_delegates_tsim_circuit_and_dem_to_injected_backend():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "circuit"
    tsim_backend._detector_error_model.return_value = "dem"
    backend = PPVMSimulatorBackend(_tsim_backend=tsim_backend)

    assert _get_tsim_circuit(backend, _physical_kernel) == "circuit"
    assert backend._detector_error_model(_physical_kernel) == "dem"
    tsim_backend._tsim_circuit.assert_called_once_with(_physical_kernel)
    tsim_backend._detector_error_model.assert_called_once_with(_physical_kernel)


def test_clifft_backend_compiles_once_normalizes_measurements_and_derives_child_seed(
    monkeypatch,
):
    clifft = _fake_clifft(monkeypatch)
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "I_ERROR[loss](0)\nM 0"
    backend = CliffTSimulatorBackend(seed=123, _tsim_backend=tsim_backend)

    first = backend.sample(_physical_kernel, shots=2)
    second = backend.sample(_physical_kernel, shots=2)

    assert first.measurements is not None
    assert first.measurements.dtype == np.bool_
    assert np.array_equal(first.measurements, [[False, True]])
    assert second.measurements is not None
    clifft.compile.assert_called_once_with("I_ERROR(0)\nM 0")  # type: ignore[attr-defined]
    assert clifft.sample.call_count == 2  # type: ignore[attr-defined]
    assert clifft.sample.call_args_list == [  # type: ignore[attr-defined]
        call("program", shots=2, seed=child_seed)
        for child_seed in _derived_seeds(123, 2)
    ]


def test_clifft_backend_omits_unconfigured_seed_and_returns_measurements(monkeypatch):
    clifft = _fake_clifft(monkeypatch)
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "M 0"
    backend = CliffTSimulatorBackend(_tsim_backend=tsim_backend)

    sample = backend.sample(_physical_kernel, shots=2)

    clifft.sample.assert_called_once_with("program", shots=2)  # type: ignore[attr-defined]
    assert sample.measurements is not None
    assert np.array_equal(sample.measurements, [[False, True]])
    assert sample.detectors is None
    assert sample.observables is None


def test_clifft_backend_delegates_tsim_circuit_and_dem_to_injected_backend():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "circuit"
    tsim_backend._detector_error_model.return_value = "dem"
    backend = CliffTSimulatorBackend(_tsim_backend=tsim_backend)

    assert _get_tsim_circuit(backend, _physical_kernel) == "circuit"
    assert backend._detector_error_model(_physical_kernel) == "dem"
    tsim_backend._tsim_circuit.assert_called_once_with(_physical_kernel)
    tsim_backend._detector_error_model.assert_called_once_with(_physical_kernel)


def test_clifft_backend_reframes_missing_tsim_dependency(monkeypatch):
    _fake_clifft(monkeypatch)
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.side_effect = ImportError("missing tsim")
    tsim_backend._detector_error_model.side_effect = ImportError("missing tsim")
    backend = CliffTSimulatorBackend(_tsim_backend=tsim_backend)

    with pytest.raises(ImportError, match=r"CliffT.*bloqade\[sim\]"):
        backend.sample(_physical_kernel, shots=1)
    with pytest.raises(ImportError, match=r"CliffT.*bloqade\[sim\]"):
        backend._detector_error_model(_physical_kernel)


def test_clifft_program_cache_drops_dead_kernel(monkeypatch):
    _fake_clifft(monkeypatch)

    class Kernel:
        pass

    kernel = Kernel()
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "M 0"
    backend = CliffTSimulatorBackend(_tsim_backend=tsim_backend)

    backend._clifft_program(cast(ir.Method, kernel))
    assert len(backend._programs) == 1
    tsim_backend.reset_mock()
    kernel_ref = weakref.ref(kernel)
    del kernel
    gc.collect()

    assert kernel_ref() is None
    assert len(backend._programs) == 0


def test_clifft_backend_real_seeded_sampling_and_dem():
    pytest.importorskip("clifft")
    pytest.importorskip("tsim")
    first_backend = CliffTSimulatorBackend(seed=991)
    second_backend = CliffTSimulatorBackend(seed=991)

    first = first_backend.sample(_random_physical_kernel, shots=16)
    second = second_backend.sample(_random_physical_kernel, shots=16)

    assert first.measurements is not None
    assert second.measurements is not None
    assert first.measurements.shape == (16, 1)
    assert np.array_equal(first.measurements, second.measurements)
    assert first_backend._detector_error_model(_random_physical_kernel) is not None
    first_backend._programs.clear()
    second_backend._programs.clear()


def test_pyqrack_backend_real_deterministic_sampling_uses_fresh_shot_tasks():
    pytest.importorskip("pyqrack")
    backend = _PyQrackSimulatorBackend(seed=441)

    sample = backend.sample(_one_physical_kernel, shots=16)

    assert sample.measurements is not None
    assert sample.measurements.shape == (16, 1)
    assert sample.measurements.dtype == np.bool_
    assert np.all(sample.measurements)


def test_pyqrack_backend_seed_reproduces_native_seed_sequence(monkeypatch):
    pyqrack = pytest.importorskip("pyqrack")
    original_seed = pyqrack.QrackSimulator.seed
    recorded_seeds: list[int] = []

    def recording_seed(simulator, seed: int):
        recorded_seeds.append(seed)
        return original_seed(simulator, seed)

    monkeypatch.setattr(pyqrack.QrackSimulator, "seed", recording_seed)
    first_backend = _PyQrackSimulatorBackend(seed=441)
    second_backend = _PyQrackSimulatorBackend(seed=441)

    first_backend.sample(_random_physical_kernel, shots=4)
    second_backend.sample(_random_physical_kernel, shots=4)

    expected_rng = np.random.default_rng(441)
    expected_seeds = [int(expected_rng.integers(0, 2**63)) for _ in range(4)]
    assert recorded_seeds == expected_seeds * 2


@pytest.mark.xfail(
    reason=(
        "PyQrack native measurement sampling has not been reproducible from "
        "seeded backend instances on Linux CI."
    ),
)
def test_pyqrack_backend_seed_reproduces_native_measurements():
    pytest.importorskip("pyqrack")
    first_backend = _PyQrackSimulatorBackend(seed=441)
    second_backend = _PyQrackSimulatorBackend(seed=441)

    first = first_backend.sample(_random_physical_kernel, shots=64)
    second = second_backend.sample(_random_physical_kernel, shots=64)

    assert first.measurements is not None
    assert second.measurements is not None
    assert np.array_equal(first.measurements, second.measurements)


def test_pyqrack_backend_seed_reproduces_squin_loss_noise():
    pytest.importorskip("pyqrack")
    first_backend = _PyQrackSimulatorBackend(seed=442)
    second_backend = _PyQrackSimulatorBackend(seed=442)

    first = first_backend.sample(_loss_physical_kernel, shots=16)
    second = second_backend.sample(_loss_physical_kernel, shots=16)

    assert first.measurements is not None
    assert second.measurements is not None
    assert np.array_equal(first.measurements, second.measurements)
    assert np.any(first.measurements)
    assert np.any(~first.measurements)


def test_pyqrack_derives_fresh_native_seed_for_each_shot(monkeypatch):
    pyqrack = pytest.importorskip("pyqrack")
    original_seed = pyqrack.QrackSimulator.seed
    recorded_seeds: list[int] = []

    def recording_seed(simulator, seed: int):
        recorded_seeds.append(seed)
        return original_seed(simulator, seed)

    monkeypatch.setattr(pyqrack.QrackSimulator, "seed", recording_seed)
    backend_seed = 443

    _PyQrackSimulatorBackend(seed=backend_seed).sample(_one_physical_kernel, shots=4)

    expected_rng = np.random.default_rng(backend_seed)
    expected_seeds = [int(expected_rng.integers(0, 2**63)) for _ in range(4)]
    assert recorded_seeds == expected_seeds


def test_pyqrack_backend_seed_advances_persistent_seed_stream(monkeypatch):
    pyqrack = pytest.importorskip("pyqrack")

    original_seed = pyqrack.QrackSimulator.seed
    recorded_seeds: list[int] = []

    def recording_seed(simulator, seed: int):
        recorded_seeds.append(seed)
        return original_seed(simulator, seed)

    monkeypatch.setattr(pyqrack.QrackSimulator, "seed", recording_seed)
    backend = _PyQrackSimulatorBackend(seed=444)

    backend.sample(_random_physical_kernel, shots=4)
    backend.sample(_random_physical_kernel, shots=4)

    persistent_rng = np.random.default_rng(444)
    first_persistent = [int(persistent_rng.integers(0, 2**63)) for _ in range(4)]
    second_persistent = [int(persistent_rng.integers(0, 2**63)) for _ in range(4)]

    assert recorded_seeds == first_persistent + second_persistent


def test_pyqrack_backend_prepares_owned_annotation_free_kernel_once(monkeypatch):
    from bloqade.decoders.dialects.annotate.stmts import SetDetector

    from bloqade.gemini.device.physical_simulator import (
        append_measurements_and_annotations_physical,
    )

    annotated = _physical_kernel.similar()
    append_measurements_and_annotations_physical(annotated, m2dets=[[1]], m2obs=None)
    source_ir = annotated.print_str()
    similar = MagicMock(wraps=annotated.similar)
    monkeypatch.setattr(annotated, "similar", similar)

    sample = _PyQrackSimulatorBackend(seed=2).sample(annotated, shots=3)

    assert sample.measurements is not None
    assert sample.measurements.shape == (3, 1)
    assert similar.call_count == 1
    assert annotated.print_str() == source_ir
    assert any(
        isinstance(stmt, SetDetector) for stmt in annotated.callable_region.walk()
    )


def test_pyqrack_backend_delegates_tsim_circuit_and_dem_to_injected_backend():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "circuit"
    tsim_backend._detector_error_model.return_value = "dem"
    backend = _PyQrackSimulatorBackend(_tsim_backend=tsim_backend)

    assert _get_tsim_circuit(backend, _physical_kernel) == "circuit"
    assert backend._detector_error_model(_physical_kernel) == "dem"
    tsim_backend._tsim_circuit.assert_called_once_with(_physical_kernel)
    tsim_backend._detector_error_model.assert_called_once_with(_physical_kernel)


def test_pyqrack_backend_reframes_missing_tsim_dependency():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.side_effect = ImportError("missing tsim")
    tsim_backend._detector_error_model.side_effect = ImportError("missing tsim")
    backend = _PyQrackSimulatorBackend(_tsim_backend=tsim_backend)

    with pytest.raises(ImportError, match=r"PyQrack.*bloqade\[sim\]"):
        _get_tsim_circuit(backend, _physical_kernel)
    with pytest.raises(ImportError, match=r"PyQrack.*bloqade\[sim\]"):
        backend._detector_error_model(_physical_kernel)


def test_pyqrack_measurements_are_mapped_explicitly():
    values = SimpleNamespace(Zero=0, One=1, Lost=2)

    assert _PyQrackSimulatorBackend._measurement_to_bool(0, values) is False
    assert _PyQrackSimulatorBackend._measurement_to_bool(1, values) is True
    with pytest.raises(ValueError, match="Unsupported PyQrack measurement"):
        _PyQrackSimulatorBackend._measurement_to_bool(2, values)
