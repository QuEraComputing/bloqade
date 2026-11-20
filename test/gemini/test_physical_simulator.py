import inspect
import importlib.util
from typing import TYPE_CHECKING, Any
from dataclasses import is_dataclass
from unittest.mock import MagicMock
from concurrent.futures import Future

import numpy as np
import pytest
from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import func, ilist, ssacfg
from typing_extensions import assert_type
from bloqade.lanes.prelude import kernel as move_kernel
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import arch as gemini_qubit, move
from bloqade.lanes.arch.metrics import MoveMetricCalculator
from bloqade.lanes.bytecode.encoding import Direction, SiteLaneAddress
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable

from bloqade import squin, types as bloqade_types
from bloqade.gemini.device import (
    Result,
    BackendSample,
    DetectorResult,
    SimulatorResult,
    TsimSimulatorBackend,
    PPVMSimulatorBackend,
    CliffTSimulatorBackend,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
)
from bloqade.gemini.compile.task import _find_qubit_ssas
from bloqade.gemini.post_processing import build_post_processing
from bloqade.gemini.device._task_runtime import _SimulatorTaskBase
from bloqade.gemini.device.simulator_backend import _PyQrackSimulatorBackend
from bloqade.gemini.device.physical_simulator import (
    DetectorResult as PhysicalDetectorResult,
    PhysicalResult,
    PhysicalSimulator,
    PhysicalSimulatorTask,
    append_measurements_and_annotations_physical,
)

_HAS_CLIFFT = importlib.util.find_spec("clifft") is not None
"""clifft is an optional dependency gated to Python >= 3.12 (see the
msd-reprod extra); backends using it must be skipped when it is absent."""


@squin.kernel
def small_physical_kernel():
    reg = squin.qalloc(1)
    squin.x(reg[0])
    return squin.broadcast.measure(reg)


@squin.kernel
def physical_kernel_with_moves_and_gates():
    reg = squin.qalloc(2)
    squin.h(reg[0])
    squin.cz(reg[0], reg[1])
    return squin.broadcast.measure(reg)


def _plain_callable():
    return None


def test_physical_result_uses_post_processing():
    raw_measurements = [[True, False, True]]
    post_processing = MagicMock()
    post_processing.emit_return.return_value = ["return-value"]
    post_processing.emit_detectors.return_value = [[True, False]]
    post_processing.emit_observables.return_value = [[False]]

    result = PhysicalResult(
        _raw_measurements=raw_measurements,
        _detector_error_model=MagicMock(),
        _post_processing=post_processing,
        _fidelity_min=0.5,
        _fidelity_max=0.9,
    )

    assert result.measurements == raw_measurements
    assert result.return_values == ["return-value"]
    assert result.detectors == [[True, False]]
    assert result.observables == [[False]]
    post_processing.emit_return.assert_called_once_with(raw_measurements)
    post_processing.emit_detectors.assert_called_once_with(raw_measurements)
    post_processing.emit_observables.assert_called_once_with(raw_measurements)


def _mock_physical_task() -> Any:
    task = object.__new__(PhysicalSimulatorTask)
    backend = MagicMock()
    backend._detector_error_model.return_value = "dem"
    object.__setattr__(task, "physical_squin_kernel", "noisy-kernel")
    object.__setattr__(task, "noiseless_physical_squin_kernel", "noiseless-kernel")
    object.__setattr__(task, "_simulator_backend", backend)
    object.__setattr__(task, "fidelity_bounds", MagicMock(return_value=(0.5, 0.9)))
    object.__setattr__(task, "_post_processing", MagicMock())
    return task


def test_physical_task_run_routes_through_backend():
    task = _mock_physical_task()
    raw_measurements = np.array([[True, False]])
    task._backend.sample.return_value = BackendSample(measurements=raw_measurements)
    task._backend._convert_loss_to_measurement.side_effect = (
        lambda shots_arr, **_kwargs: shots_arr
    )

    result = PhysicalSimulatorTask.run(task, shots=1, with_noise=True)

    task._backend._detector_error_model.assert_called_once_with("noisy-kernel")
    task._backend.sample.assert_called_once_with("noisy-kernel", shots=1)
    task._backend._convert_loss_to_measurement.assert_called_once_with(
        raw_measurements, loss_replace=True, loss=None
    )
    assert isinstance(result, PhysicalResult)
    assert result.measurements == [[True, False]]
    assert result.detector_error_model == "dem"


def test_physical_simulator_exports_from_gemini_namespace():
    from bloqade.gemini import PhysicalSimulator, GeminiPhysicalSimulator
    from bloqade.gemini.device import PhysicalSimulator as DevicePhysicalSimulator

    assert PhysicalSimulator is DevicePhysicalSimulator
    assert PhysicalSimulator is GeminiPhysicalSimulator


def test_physical_simulator_result_aliases_are_public():
    assert PhysicalResult is Result
    assert PhysicalDetectorResult is DetectorResult


def test_physical_simulator_defaults_to_ppvm_backend():
    simulator = PhysicalSimulator()

    assert isinstance(simulator.backend, PPVMSimulatorBackend)
    assert simulator.task(small_physical_kernel)._backend is simulator.backend


def test_logical_and_physical_tasks_share_non_dataclass_runtime():
    assert not is_dataclass(_SimulatorTaskBase)
    assert issubclass(GeminiLogicalSimulatorTask, _SimulatorTaskBase)
    assert issubclass(PhysicalSimulatorTask, _SimulatorTaskBase)


def test_compiled_task_exposes_move_program_metrics():
    task = PhysicalSimulator().task(physical_kernel_with_moves_and_gates)
    statements = tuple(task.physical_move_kernel.callable_region.walk())
    move_statements = tuple(
        statement for statement in statements if isinstance(statement, move.Move)
    )
    single_qubit_gate_types = (
        move.LocalR,
        move.GlobalR,
        move.LocalRz,
        move.GlobalRz,
        move.StarRz,
    )
    move_metric_calculator = MoveMetricCalculator(task.physical_arch_spec)
    state_frame, _ = atom.AtomInterpreter(
        task.physical_move_kernel.dialects,
        arch_spec=task.physical_arch_spec,
    ).run(task.physical_move_kernel)

    assert task.move_depth == len(move_statements)
    assert task.cz_pulse_count == sum(
        isinstance(statement, move.CZ) for statement in statements
    )
    assert task.single_qubit_gate_count == sum(
        isinstance(statement, single_qubit_gate_types) for statement in statements
    )
    assert task.total_gate_count == (task.cz_pulse_count + task.single_qubit_gate_count)
    assert task.total_distance_moved_um == pytest.approx(
        sum(
            move_metric_calculator.lane_distance_um(lane)
            for statement in move_statements
            for state in (state_frame.get(statement.current_state),)
            if isinstance(state, atom.AtomState)
            for lane in statement.lanes
            if (
                state.data.get_qubit(task.physical_arch_spec.get_endpoints(lane)[0])
                is not None
            )
        )
    )
    assert task._move_program_metrics is task._move_program_metrics


def test_move_distance_ignores_empty_filler_lanes():
    @move_kernel
    def move_kernel_with_empty_lane():
        state0 = move.load()
        state1 = move.fill(state0, location_addresses=(move.LocationAddress(1, 0),))
        state2 = move.move(
            state1,
            lanes=(
                SiteLaneAddress(1, 0, 0),
                SiteLaneAddress(1, 2, 0),
            ),
        )
        move.store(state2)

    task = object.__new__(PhysicalSimulatorTask)
    object.__setattr__(task, "physical_move_kernel", move_kernel_with_empty_lane)
    object.__setattr__(task, "physical_arch_spec", get_arch_spec())

    calculator = MoveMetricCalculator(task.physical_arch_spec)
    occupied_lane = SiteLaneAddress(1, 0, 0)
    assert task.total_distance_moved_um == pytest.approx(
        calculator.lane_distance_um(occupied_lane)
    )


def test_move_distance_counts_backward_return_lanes():
    @move_kernel
    def move_kernel_with_return():
        state0 = move.load()
        state1 = move.fill(state0, location_addresses=(move.LocationAddress(1, 0),))
        state2 = move.move(state1, lanes=(SiteLaneAddress(1, 0, 0),))
        state3 = move.move(
            state2,
            lanes=(SiteLaneAddress(1, 0, 0, Direction.BACKWARD),),
        )
        move.store(state3)

    task = object.__new__(PhysicalSimulatorTask)
    object.__setattr__(task, "physical_move_kernel", move_kernel_with_return)
    object.__setattr__(task, "physical_arch_spec", get_arch_spec())

    lane = SiteLaneAddress(1, 0, 0)
    calculator = MoveMetricCalculator(task.physical_arch_spec)
    assert task.total_distance_moved_um == pytest.approx(
        2 * calculator.lane_distance_um(lane)
    )


@pytest.mark.parametrize("simulator_type", [GeminiLogicalSimulator, PhysicalSimulator])
@pytest.mark.parametrize(
    "method",
    [
        "run",
        "run_async",
        "visualize",
        "physical_squin_kernel",
        "physical_move_kernel",
        "tsim_circuit",
        "fidelity_bounds",
        "move_depth",
        "cz_pulse_count",
        "single_qubit_gate_count",
        "total_gate_count",
        "total_distance_moved_um",
    ],
)
def test_simulators_expose_task_only_execution_api(simulator_type, method):
    assert method not in simulator_type.__dict__


@pytest.mark.parametrize(
    "task_type", [GeminiLogicalSimulatorTask, PhysicalSimulatorTask]
)
def test_simulator_tasks_store_backend_privately(task_type):
    task = object.__new__(task_type)
    backend = MagicMock()
    object.__setattr__(task, "_simulator_backend", backend)

    assert task._backend is backend
    assert not hasattr(task, "backend")
    assert not hasattr(task, "simulator_backend")


@pytest.mark.parametrize(
    "attribute",
    [
        "measurement_sampler",
        "noiseless_measurement_sampler",
        "detector_sampler",
        "noiseless_detector_sampler",
    ],
)
def test_task_base_does_not_expose_cached_samplers(attribute):
    assert attribute not in _SimulatorTaskBase.__dict__


def test_physical_simulator_task_passes_placement_strategy(monkeypatch):
    import bloqade.gemini.device.physical_simulator as pipeline_module

    captured = {}
    move_kernel = MagicMock()

    class FakePhysicalPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def emit(self, kernel, no_raise=True):
            captured["kernel"] = kernel
            captured["no_raise"] = no_raise
            return move_kernel

    monkeypatch.setattr(pipeline_module, "PhysicalPipeline", FakePhysicalPipeline)

    @squin.kernel
    def kernel():
        reg = squin.qalloc(1)
        return squin.broadcast.measure(reg)

    source_ir = kernel.print_str()
    placement_strategy = MagicMock()
    place_opt_type: Any = MagicMock()
    backend = TsimSimulatorBackend(run_detectors=True)

    simulator = PhysicalSimulator(
        backend=backend,
        place_opt_type=place_opt_type,
        placement_strategy=placement_strategy,
    )
    task = simulator.task(kernel)

    assert captured["place_opt_type"] is place_opt_type
    assert captured["placement_strategy"] is placement_strategy
    assert captured["kernel"] is task.source_squin_kernel
    assert task.source_squin_kernel is not kernel
    assert task.source_squin_kernel.print_str() == source_ir
    assert kernel.print_str() == source_ir
    assert captured["no_raise"] is False
    assert task.physical_move_kernel is move_kernel
    assert task._post_processing is not None
    assert task._backend is backend
    assert not hasattr(task, "backend")
    assert not hasattr(task, "simulator_backend")
    assert not hasattr(task, "simulator")


def test_physical_task_preserves_source_kernel_across_repeated_compilation():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        return squin.broadcast.measure(reg)

    prepared_kernel = kernel.similar()
    append_measurements_and_annotations_physical(
        prepared_kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )
    prepared_ir = prepared_kernel.print_str()
    simulator = PhysicalSimulator()
    first = simulator.task(prepared_kernel)
    second = simulator.task(prepared_kernel)

    assert prepared_kernel.print_str() == prepared_ir
    assert first.source_squin_kernel is not prepared_kernel
    assert second.source_squin_kernel is not prepared_kernel
    assert first.source_squin_kernel is not second.source_squin_kernel
    for task in (first, second):
        assert (
            sum(
                isinstance(stmt, SetDetector)
                for stmt in task.source_squin_kernel.callable_region.walk()
            )
            == 1
        )
        assert (
            sum(
                isinstance(stmt, SetObservable)
                for stmt in task.source_squin_kernel.callable_region.walk()
            )
            == 1
        )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TsimSimulatorBackend(), id="tsim"),
        pytest.param(
            CliffTSimulatorBackend(seed=29),
            id="clifft",
            marks=pytest.mark.skipif(
                not _HAS_CLIFFT, reason="clifft requires Python >= 3.12"
            ),
        ),
        pytest.param(_PyQrackSimulatorBackend(seed=29), id="pyqrack"),
        pytest.param(PPVMSimulatorBackend(seed=29), id="ppvm"),
    ],
)
@pytest.mark.parametrize("with_noise", [False, True], ids=["noiseless", "noisy"])
def test_builtin_backends_run_physical_workflow_with_guaranteed_dem(
    backend, with_noise
):
    result = (
        PhysicalSimulator(backend=backend)
        .task(small_physical_kernel)
        .run(shots=2, with_noise=with_noise)
    )

    assert isinstance(result, Result)
    assert len(result.measurements) == 2
    assert all(len(measurements) == 1 for measurements in result.measurements)
    assert result.detector_error_model is not None
    if isinstance(backend, CliffTSimulatorBackend):
        backend._programs.clear()


def test_tsim_physical_detector_backend_returns_detector_result():
    prepared_kernel = small_physical_kernel.similar()
    append_measurements_and_annotations_physical(
        prepared_kernel,
        m2dets=[[1]],
        m2obs=[[1]],
    )
    simulator = PhysicalSimulator(backend=TsimSimulatorBackend(run_detectors=True))

    result = simulator.task(prepared_kernel).run(shots=2, with_noise=False)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 2
    assert all(len(detectors) == 1 for detectors in result.detectors)
    assert len(result.observables) == 2
    assert all(len(observables) == 1 for observables in result.observables)
    assert result.detector_error_model is not None


def test_pyqrack_physical_returns_measurement_backed_result():
    prepared_kernel = small_physical_kernel.similar()
    append_measurements_and_annotations_physical(
        prepared_kernel,
        m2dets=[[1]],
        m2obs=[[1]],
    )
    simulator = PhysicalSimulator(backend=_PyQrackSimulatorBackend(seed=30))

    result = simulator.task(prepared_kernel).run(shots=2, with_noise=False)

    assert isinstance(result, Result)
    assert result.detectors == [[True], [True]]
    assert result.observables == [[True], [True]]
    assert result.detector_error_model is not None


@pytest.mark.parametrize(
    "method",
    [
        PhysicalSimulatorTask.run,
        PhysicalSimulatorTask.run_async,
    ],
)
def test_physical_task_run_methods_do_not_expose_runtime_configuration(method):
    assert "run_detectors" not in inspect.signature(method).parameters
    assert "seed" not in inspect.signature(method).parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"m2dets": [[1]]}, id="m2dets"),
        pytest.param({"m2obs": [[1]]}, id="m2obs"),
    ],
)
def test_physical_simulator_constructor_rejects_measurement_matrices(kwargs):
    simulator_type: Any = PhysicalSimulator

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulator_type(**kwargs)


def test_physical_simulator_task_rejects_non_squin_before_pipeline(monkeypatch):
    import bloqade.gemini.device.physical_simulator as pipeline_module

    physical_pipeline = MagicMock()
    monkeypatch.setattr(pipeline_module, "PhysicalPipeline", physical_pipeline)
    invalid_kernel: Any = _plain_callable

    with pytest.raises(TypeError, match="Squin ir.Method"):
        PhysicalSimulator().task(invalid_kernel)

    physical_pipeline.assert_not_called()


def test_physical_simulator_task_uses_source_kernel_post_processing(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("post-processing must not use the lowered move kernel")

    monkeypatch.setattr(
        atom.AtomInterpreter,
        "get_post_processing",
        fail_if_called,
        raising=False,
    )

    task = PhysicalSimulator().task(small_physical_kernel)

    assert task._post_processing is not None


def test_append_measurements_and_annotations_physical_preserves_kernel_return():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(4)
        measurements = squin.broadcast.measure(reg)
        return measurements

    append_measurements_and_annotations_physical(
        kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )

    assert sum(isinstance(s, SetDetector) for s in kernel.callable_region.walk()) == 2
    assert sum(isinstance(s, SetObservable) for s in kernel.callable_region.walk()) == 2

    post_processing = build_post_processing(kernel)

    raw_shots = [[True, False, False, True]]

    return_values = list(post_processing.emit_return(raw_shots))
    assert len(return_values) == 1
    assert list(return_values[0]) == raw_shots[0]
    assert list(post_processing.emit_detectors(raw_shots)) == [[True, True]]
    assert list(post_processing.emit_observables(raw_shots)) == [[True, False]]


def test_append_measurements_and_annotations_physical_uses_logical_mvp_helpers(
    monkeypatch,
):
    import bloqade.gemini.compile.task as compile_task_module

    find_qubit_ssas = MagicMock(wraps=compile_task_module._find_qubit_ssas)
    find_return_stmt = MagicMock(wraps=compile_task_module._find_return_stmt)
    insert_before = MagicMock(wraps=compile_task_module._insert_before)
    monkeypatch.setattr(compile_task_module, "_find_qubit_ssas", find_qubit_ssas)
    monkeypatch.setattr(compile_task_module, "_find_return_stmt", find_return_stmt)
    monkeypatch.setattr(compile_task_module, "_insert_before", insert_before)

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.broadcast.measure(reg)

    append_measurements_and_annotations_physical(
        kernel,
        m2dets=[[1], [1]],
        m2obs=None,
    )

    find_qubit_ssas.assert_called_once_with(kernel)
    find_return_stmt.assert_called_once_with(kernel)
    assert insert_before.call_count > 0


def test_append_measurements_and_annotations_physical_validates_block_size():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.broadcast.measure(reg)

    with pytest.raises(
        ValueError,
        match="physical qubits must be divisible",
    ):
        append_measurements_and_annotations_physical(
            kernel,
            m2dets=[[1], [1]],
            m2obs=None,
        )


def test_append_measurements_and_annotations_physical_accepts_new_at_allocations():
    kernel = squin.kernel.add(gemini_qubit)
    kernel.run_pass = squin.kernel.run_pass

    @kernel(typeinfer=True)
    def pinned_kernel():
        q0 = gemini_qubit.new_at(0, 0, 0)
        q1 = gemini_qubit.new_at(0, 1, 0)
        squin.broadcast.measure(ilist.IList([q0, q1]))

    append_measurements_and_annotations_physical(
        pinned_kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )

    assert (
        sum(isinstance(s, SetDetector) for s in pinned_kernel.callable_region.walk())
        == 1
    )
    assert (
        sum(isinstance(s, SetObservable) for s in pinned_kernel.callable_region.walk())
        == 1
    )


def test_find_qubit_ssas_ignores_qubit_typed_results_without_addresses():
    test_dialect = ir.Dialect("test.qubit_alloc")

    @statement(dialect=test_dialect)
    class CustomQubitAlloc(ir.Statement):
        result: ir.ResultValue = info.result(bloqade_types.QubitType)

    block = ir.Block(argtypes=(types.MethodType,))
    alloc = CustomQubitAlloc()
    none_stmt = func.ConstantNone()
    for stmt in (alloc, none_stmt, func.Return(none_stmt.result)):
        block.stmts.append(stmt)

    function = func.Function(
        sym_name="custom_alloc",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=ir.Region(blocks=block),
    )
    method = ir.Method(
        dialects=ir.DialectGroup([func.dialect, ssacfg.dialect, test_dialect]),
        code=function,
        sym_name="custom_alloc",
        arg_names=[],
    )

    assert _find_qubit_ssas(method) == []


if TYPE_CHECKING:

    def _check_physical_run_results(
        task: PhysicalSimulatorTask[Any],
    ) -> None:
        assert_type(task.run(), SimulatorResult[Any])
        assert_type(task.run_async(), Future[SimulatorResult[Any]])
