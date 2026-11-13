import math
import inspect
import importlib.util
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from concurrent.futures import Future

import numpy as np
import pytest
from stim import DetectorErrorModel
from kirin.dialects import ilist
from typing_extensions import assert_type
from bloqade.decoders.dialects import annotate
from bloqade.lanes.noise_model import generate_logical_noise_model

from bloqade import qubit, squin, types
from bloqade.gemini import (
    SimulatorResult as GeminiSimulatorResult,
    logical as gemini_logical,
)
from bloqade.gemini.cudaq import cudaq_to_squin
from bloqade.gemini.device import (
    Result,
    BackendSample,
    DetectorResult,
    SimulatorResult,
    TsimSimulatorBackend,
    CliffTSimulatorBackend,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
)
from bloqade.gemini.compile import append_measurements_and_annotations
from bloqade.gemini.steane_defaults import steane7_m2obs, steane7_m2dets
from bloqade.gemini.device.simulator import (
    Result as SimulatorResultImplementation,
    DetectorResult as SimulatorDetectorResult,
)
from bloqade.gemini.device.simulator_backend import _PyQrackSimulatorBackend

_HAS_CLIFFT = importlib.util.find_spec("clifft") is not None
"""clifft is an optional dependency gated to Python >= 3.12 (see the
msd-reprod extra); backends using it must be skipped when it is absent."""


@gemini_logical.kernel(verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0])
    annotate.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1])
    annotate.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2])


@gemini_logical.kernel(verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_observable([meas[0], meas[1], meas[5]])


@gemini_logical.kernel(aggressive_unroll=True)
def main():
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

    squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
    squin.sqrt_x_adj(reg[0])
    squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
    squin.broadcast.sqrt_y_adj(reg)

    measurements = gemini_logical.terminal_measure(reg)

    for i in range(len(reg)):
        set_detector(measurements[i])
        set_observable(measurements[i])


@gemini_logical.kernel(aggressive_unroll=True)
def small_backend_kernel():
    reg = qubit.qalloc(1)
    squin.x(reg[0])
    measurements = gemini_logical.terminal_measure(reg)
    set_detector(measurements[0])
    set_observable(measurements[0])


def _plain_callable():
    return None


@pytest.mark.slow
@pytest.mark.parametrize("size", [2, 6])
def test_physical_compilation(size: int):
    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        reg = qubit.qalloc(1)
        squin.h(reg[0])
        for _ in range(size):
            current = len(reg)
            missing = size - current
            if missing > current:
                num_alloc = current
            else:
                num_alloc = missing

            if num_alloc > 0:
                new_qubits = qubit.qalloc(num_alloc)
                squin.broadcast.cx(reg[-num_alloc:], new_qubits)
                reg = reg + new_qubits

        meas = gemini_logical.terminal_measure(reg)

        def set_observable(qubit_index: int):
            return squin.set_observable(
                [meas[qubit_index][0], meas[qubit_index][1], meas[qubit_index][5]]
            )

        return ilist.map(set_observable, ilist.range(len(reg)))

    result = GeminiLogicalSimulator().task(main).run(1000, with_noise=False)
    # checks to make sure logical GHZ state is created.
    assert all(len(set(rv)) == 1 for rv in result.observables)


@pytest.mark.slow
def test_run_default():
    """Test that default measurement sampling returns a Result."""
    sim = GeminiLogicalSimulator()
    result = sim.task(main).run(shots=5, with_noise=False)

    assert isinstance(result, Result)
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


@pytest.mark.slow
def test_run_with_detector_backend():
    """Test that a detector-configured backend returns a DetectorResult."""
    sim = GeminiLogicalSimulator(backend=TsimSimulatorBackend(run_detectors=True))
    result = sim.task(main).run(shots=10, with_noise=False)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


@pytest.mark.slow
def test_detector_backend_with_noise():
    """Test a detector-configured backend samples the noisy circuit."""
    sim = GeminiLogicalSimulator(backend=TsimSimulatorBackend(run_detectors=True))
    result = sim.task(main).run(shots=10, with_noise=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None


@pytest.mark.slow
def test_run_async_with_detector_backend():
    """Test a detector-configured backend returns a Future[DetectorResult]."""
    sim = GeminiLogicalSimulator(backend=TsimSimulatorBackend(run_detectors=True))
    future = sim.task(main).run_async(shots=5, with_noise=False)
    result = future.result()

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


@pytest.mark.slow
def test_detector_backend_via_task():
    """Test a detector-configured backend when running a task directly."""
    sim = GeminiLogicalSimulator(backend=TsimSimulatorBackend(run_detectors=True))
    task = sim.task(main)
    result = task.run(shots=5, with_noise=False)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


@pytest.mark.slow
def test_detector_backend_task_directly():
    """Test creating a task with a detector-configured backend."""
    sim = GeminiLogicalSimulator(
        noise_model=generate_logical_noise_model(),
        backend=TsimSimulatorBackend(run_detectors=True),
    )
    task = sim.task(main)
    result = task.run(shots=5, with_noise=False)
    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


@pytest.mark.slow
def test_detector_backend_task_async():
    """Test run_async directly on a task with a detector-configured backend."""
    sim = GeminiLogicalSimulator(
        noise_model=generate_logical_noise_model(),
        backend=TsimSimulatorBackend(run_detectors=True),
    )
    task = sim.task(main)
    future = task.run_async(shots=5, with_noise=False)
    result = future.result()
    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


def test_result_property_caching():
    """Test that Result properties return cached values on subsequent access."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def returning_kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        meas = gemini_logical.terminal_measure(reg)

        def set_observable(qubit_index: int):
            return squin.set_observable(
                [meas[qubit_index][0], meas[qubit_index][1], meas[qubit_index][5]]
            )

        return ilist.map(set_observable, ilist.range(len(reg)))

    sim = GeminiLogicalSimulator()
    result = sim.task(returning_kernel).run(shots=5, with_noise=False)

    # Access each property twice to exercise the caching path
    detectors_first = result.detectors
    detectors_second = result.detectors
    assert detectors_first is detectors_second

    observables_first = result.observables
    observables_second = result.observables
    assert observables_first is observables_second

    measurements_first = result.measurements
    measurements_second = result.measurements
    assert measurements_first is measurements_second

    return_values_first = result.return_values
    return_values_second = result.return_values
    assert return_values_first is return_values_second


def test_detector_result_rejects_unavailable_values():
    detector_error_model = DetectorErrorModel()
    result = DetectorResult[None](
        _detector_error_model=detector_error_model,
        _fidelity_min=0.9,
        _fidelity_max=1.0,
        _detectors=[[True]],
        _observables=[[False]],
    )

    assert result.fidelity_bounds() == (0.9, 1.0)
    assert result.detector_error_model is detector_error_model
    assert result.detectors == ((True,),)
    assert result.observables == ((False,),)
    with pytest.raises(ValueError, match="Raw measurements are unavailable"):
        _ = result.measurements
    with pytest.raises(ValueError, match="kernel return values are unavailable"):
        _ = result.return_values


def test_simulator_result_exports_are_public():
    assert SimulatorResult is GeminiSimulatorResult


def test_logical_simulator_result_aliases_are_public():
    assert SimulatorResultImplementation is Result
    assert SimulatorDetectorResult is DetectorResult


def test_noiseless_tsim_circuit_compiles_samplers():
    sim = GeminiLogicalSimulator()
    task = sim.task(main)

    noiseless = task.noiseless_tsim_circuit

    assert noiseless.compile_sampler() is not None
    assert noiseless.compile_detector_sampler() is not None


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TsimSimulatorBackend(), id="tsim"),
        pytest.param(
            CliffTSimulatorBackend(seed=17),
            id="clifft",
            marks=pytest.mark.skipif(
                not _HAS_CLIFFT, reason="clifft requires Python >= 3.12"
            ),
        ),
        pytest.param(_PyQrackSimulatorBackend(seed=17), id="pyqrack"),
    ],
)
def test_builtin_backends_run_logical_workflow_with_guaranteed_dem(backend):
    result = (
        GeminiLogicalSimulator(backend=backend)
        .task(small_backend_kernel)
        .run(shots=2, with_noise=False)
    )

    assert isinstance(result, Result)
    assert len(result.measurements) == 2
    assert result.detector_error_model is not None
    if isinstance(backend, CliffTSimulatorBackend):
        backend._programs.clear()


def test_pyqrack_logical_returns_measurement_backed_result():
    result = (
        GeminiLogicalSimulator(backend=_PyQrackSimulatorBackend(seed=18))
        .task(small_backend_kernel)
        .run(shots=2, with_noise=False)
    )

    assert isinstance(result, Result)
    assert len(result.detectors) == 2
    assert len(result.observables) == 2
    assert result.detector_error_model is not None


@pytest.mark.parametrize(
    "method",
    [
        GeminiLogicalSimulatorTask.run,
        GeminiLogicalSimulatorTask.run_async,
    ],
)
def test_logical_task_run_methods_do_not_expose_runtime_configuration(method):
    assert "run_detectors" not in inspect.signature(method).parameters
    assert "seed" not in inspect.signature(method).parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"m2dets": [[1]]}, id="m2dets"),
        pytest.param({"m2obs": [[1]]}, id="m2obs"),
    ],
)
def test_logical_simulator_constructor_rejects_measurement_matrices(kwargs):
    simulator_type: Any = GeminiLogicalSimulator

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulator_type(**kwargs)


def test_logical_simulator_task_rejects_non_squin_before_compilation(monkeypatch):
    import bloqade.gemini.compile as gemini_compile

    compile_task = MagicMock()
    monkeypatch.setattr(gemini_compile, "compile_task", compile_task)
    invalid_kernel: Any = _plain_callable

    with pytest.raises(TypeError, match="Squin ir.Method"):
        GeminiLogicalSimulator().task(invalid_kernel)

    compile_task.assert_not_called()


def test_logical_simulator_task_preserves_source_kernel():
    source_ir = small_backend_kernel.print_str()
    backend = TsimSimulatorBackend(run_detectors=True)

    task = GeminiLogicalSimulator(backend=backend).task(small_backend_kernel)

    assert small_backend_kernel.print_str() == source_ir
    assert task.logical_squin_kernel is not small_backend_kernel
    assert task._backend is backend
    assert not hasattr(task, "backend")
    assert not hasattr(task, "simulator_backend")
    assert not hasattr(task, "simulator")


def _steane_matrices(num_qubits: int):
    return steane7_m2dets(num_qubits), steane7_m2obs(num_qubits)


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TsimSimulatorBackend(), id="measurement"),
        pytest.param(TsimSimulatorBackend(run_detectors=True), id="detector"),
    ],
)
def test_logical_x_observable_is_one(backend: TsimSimulatorBackend):
    m2dets, m2obs = _steane_matrices(1)

    @gemini_logical.kernel(aggressive_unroll=True)
    def logical_x():
        reg = qubit.qalloc(1)
        squin.x(reg[0])
        gemini_logical.terminal_measure(reg)

    append_measurements_and_annotations(logical_x, m2dets, m2obs)
    task = GeminiLogicalSimulator(backend=backend).task(logical_x)
    result = task.run(10, with_noise=False)

    expected_observable_width = len(m2obs[0])
    assert expected_observable_width > 0
    assert len(result.observables) == 10
    assert all(
        len(observables) == expected_observable_width
        for observables in result.observables
    )
    assert all(all(obs) for obs in result.observables)


@pytest.mark.slow
@pytest.mark.parametrize(
    "use_dets, use_obs",
    [(True, True), (True, False), (False, True)],
    ids=["both", "dets_only", "obs_only"],
)
def test_explicit_annotations_on_kernel_with_terminal_measure(
    use_dets: bool, use_obs: bool
):
    """Explicitly append annotations to a kernel with a terminal measure."""
    num_qubits = 2
    m2dets, m2obs = _steane_matrices(num_qubits)

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel_with_measure():
        reg = qubit.qalloc(num_qubits)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    selected_m2dets = m2dets if use_dets else None
    selected_m2obs = m2obs if use_obs else None
    append_measurements_and_annotations(
        kernel_with_measure, selected_m2dets, selected_m2obs
    )
    task = GeminiLogicalSimulator().task(kernel_with_measure)
    result = task.run(10, with_noise=False)

    expected_detector_width = len(m2dets[0]) if use_dets else 0
    expected_observable_width = len(m2obs[0]) if use_obs else 0
    assert len(result.detectors) == 10
    assert all(len(det) == expected_detector_width for det in result.detectors)
    assert len(result.observables) == 10
    assert all(len(obs) == expected_observable_width for obs in result.observables)

    if use_dets:
        assert all(isinstance(b, bool) for det in result.detectors for b in det)

    if use_obs:
        assert all(isinstance(b, bool) for obs in result.observables for b in obs)


@pytest.mark.parametrize(
    "use_dets, use_obs",
    [(True, True), (True, False), (False, True)],
    ids=["both", "dets_only", "obs_only"],
)
def test_cudaq_kernel_explicit_conversion_and_annotation(use_dets: bool, use_obs: bool):
    cudaq = pytest.importorskip("cudaq")

    num_qubits = 2
    m2dets, m2obs = _steane_matrices(num_qubits)

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(num_qubits)
        h(q[0])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        cx(q[0], q[1])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

    squin_kernel = cudaq_to_squin(bell_pair)
    selected_m2dets = m2dets if use_dets else None
    selected_m2obs = m2obs if use_obs else None
    append_measurements_and_annotations(squin_kernel, selected_m2dets, selected_m2obs)
    task = GeminiLogicalSimulator().task(squin_kernel)
    result = task.run(10, with_noise=False)

    expected_detector_width = len(m2dets[0]) if use_dets else 0
    expected_observable_width = len(m2obs[0]) if use_obs else 0
    assert len(result.detectors) == 10
    assert all(len(det) == expected_detector_width for det in result.detectors)
    assert len(result.observables) == 10
    assert all(len(obs) == expected_observable_width for obs in result.observables)

    if use_dets:
        assert all(isinstance(b, bool) for det in result.detectors for b in det)

    if use_obs:
        assert all(isinstance(b, bool) for obs in result.observables for b in obs)


def _mock_task() -> Any:
    task = object.__new__(GeminiLogicalSimulatorTask)
    backend = MagicMock()
    backend._detector_error_model.return_value = "dem"
    backend._convert_loss_to_measurement.side_effect = (
        lambda shots_arr, **_kwargs: shots_arr
    )
    object.__setattr__(task, "physical_squin_kernel", "noisy-kernel")
    object.__setattr__(task, "noiseless_physical_squin_kernel", "noiseless-kernel")
    object.__setattr__(task, "_simulator_backend", backend)
    object.__setattr__(task, "fidelity_bounds", MagicMock(return_value=(0.5, 0.9)))
    object.__setattr__(task, "_post_processing", MagicMock())
    return task


def test_run_samples_noisy_kernel_through_backend_after_dem_generation():
    task = _mock_task()
    samples = np.array([[True, False]])
    task._backend.sample.return_value = BackendSample(measurements=samples)

    result = GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=True)

    task._backend._detector_error_model.assert_called_once_with("noisy-kernel")
    task._backend.sample.assert_called_once_with("noisy-kernel", shots=1)
    assert isinstance(result, Result)
    assert result._raw_measurements == samples.tolist()
    assert result.detector_error_model == "dem"


def test_run_samples_noiseless_kernel_through_backend():
    task = _mock_task()
    task._backend.sample.return_value = BackendSample(measurements=np.array([[True]]))

    GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=False)

    task._backend.sample.assert_called_once_with("noiseless-kernel", shots=1)


def test_run_uses_native_backend_detector_and_observable_samples():
    task = _mock_task()
    detectors, observables = np.array([[True]]), np.array([[False]])
    task._backend.sample.return_value = BackendSample(
        detectors=detectors, observables=observables
    )

    result = GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=True)

    task._backend.sample.assert_called_once_with("noisy-kernel", shots=1)
    assert isinstance(result, DetectorResult)
    assert result._detectors == detectors.tolist()
    assert result._observables == observables.tolist()


@pytest.mark.parametrize(
    "sample, message",
    [
        (BackendSample(measurements=np.array([True])), "two-dimensional"),
        (BackendSample(measurements=np.array([[True], [False]])), "rows"),
    ],
)
def test_run_rejects_invalid_backend_measurement_payloads(sample, message):
    task = _mock_task()
    task._backend.sample.return_value = sample

    with pytest.raises(ValueError, match=message):
        GeminiLogicalSimulatorTask.run(task, shots=1)


@pytest.mark.parametrize(
    "sample",
    [
        BackendSample(),
        BackendSample(detectors=np.array([[True]])),
        BackendSample(observables=np.array([[False]])),
        BackendSample(measurements=np.array([[True]]), detectors=np.array([[True]])),
        BackendSample(measurements=np.array([[True]]), observables=np.array([[False]])),
        BackendSample(
            measurements=np.array([[True]]),
            detectors=np.array([[True]]),
            observables=np.array([[False]]),
        ),
    ],
)
def test_run_rejects_nonexclusive_backend_payload_shapes(sample):
    task = _mock_task()
    task._backend.sample.return_value = sample

    with pytest.raises(
        ValueError, match="measurement-only or detector\\+observable-only"
    ):
        GeminiLogicalSimulatorTask.run(task, shots=1)


@pytest.mark.parametrize(
    "sample, message",
    [
        (
            BackendSample(detectors=np.array([True]), observables=np.array([[False]])),
            "detector samples must be a two-dimensional",
        ),
        (
            BackendSample(
                detectors=np.array([[True], [False]]),
                observables=np.array([[False], [True]]),
            ),
            "2 detector rows for 1 shots",
        ),
        (
            BackendSample(detectors=np.array([[True]]), observables=np.array([False])),
            "observable samples must be a two-dimensional",
        ),
        (
            BackendSample(
                detectors=np.array([[True]]),
                observables=np.array([[False], [True]]),
            ),
            "2 observable rows for 1 shots",
        ),
    ],
)
def test_run_rejects_invalid_backend_detector_payloads(sample, message):
    task = _mock_task()
    task._backend.sample.return_value = sample

    with pytest.raises(ValueError, match=message):
        GeminiLogicalSimulatorTask.run(task, shots=1)


def test_run_fails_on_dem_before_sampling():
    task = _mock_task()
    task._backend._detector_error_model.side_effect = ImportError("no tsim")

    with pytest.raises(ImportError, match="no tsim"):
        GeminiLogicalSimulatorTask.run(task, shots=1)

    task._backend.sample.assert_not_called()


def test_task_run_async_submits_run_without_runtime_configuration():
    task = _mock_task()
    executor = MagicMock()
    future = Future()
    future.set_result(object())
    executor.submit.return_value = future
    run = MagicMock()
    object.__setattr__(task, "_thread_pool_executor", executor)
    object.__setattr__(task, "run", run)

    result = GeminiLogicalSimulatorTask.run_async(task, shots=3, with_noise=False)

    assert result is future
    executor.submit.assert_called_once_with(run, 3, False)


if TYPE_CHECKING:

    def _check_simulator_result_protocol(result: Result[Any]) -> None:
        common_result: SimulatorResult[Any] = result
        detector_result: SimulatorResult[Any] = DetectorResult[Any](
            _detector_error_model=DetectorErrorModel(),
            _fidelity_min=0.9,
            _fidelity_max=1.0,
            _detectors=[[True]],
            _observables=[[False]],
        )
        assert common_result is not None
        assert detector_result is not None

    def _check_task_run_results(task: GeminiLogicalSimulatorTask[Any]) -> None:
        assert_type(task.run(), SimulatorResult[Any])
        assert_type(task.run_async(), Future[SimulatorResult[Any]])
