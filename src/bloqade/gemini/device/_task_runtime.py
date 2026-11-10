from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Protocol,
    cast,
)
from functools import cached_property
from dataclasses import dataclass
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from stim import DetectorErrorModel
from kirin import ir
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.analysis.fidelity import FidelityAnalysis
from bloqade.lanes.arch.metrics import MoveMetricCalculator

from bloqade.gemini.device.simulator_backend import (
    AbstractSimulatorBackend,
    _get_tsim_circuit,
)

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]
    from bloqade.lanes.arch.spec import ArchSpec

RetType = TypeVar("RetType")
ResultRetType_co = TypeVar("ResultRetType_co", covariant=True)


class SimulatorResult(Protocol[ResultRetType_co]):
    """Common interface for simulator results."""

    def fidelity_bounds(self) -> tuple[float, float]: ...

    @property
    def detector_error_model(self) -> DetectorErrorModel: ...

    @property
    def return_values(self) -> Sequence[ResultRetType_co]: ...

    @property
    def detectors(self) -> Sequence[Sequence[bool]]: ...

    @property
    def measurements(self) -> Sequence[Sequence[bool]]: ...

    @property
    def observables(self) -> Sequence[Sequence[bool]]: ...


@dataclass(frozen=True)
class DetectorResult(Generic[ResultRetType_co]):
    """Detector and observable outcomes from simulator sampling."""

    _detector_error_model: DetectorErrorModel
    _fidelity_min: float
    _fidelity_max: float
    _detectors: list[list[bool]]
    _observables: list[list[bool]]

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit.

        Returns:
            DetectorErrorModel: The STIM detector error model.

        """
        return self._detector_error_model

    @property
    def return_values(self) -> Sequence[ResultRetType_co]:
        raise ValueError(
            "kernel return values are unavailable for detector-only results"
        )

    @property
    def detectors(self) -> tuple[tuple[bool, ...], ...]:
        """The detector outcomes from the simulation.

        Returns:
            tuple[tuple[bool, ...], ...]: The detector outcomes, one tuple per shot.

        """
        return tuple(tuple(shot) for shot in self._detectors)

    @property
    def measurements(self) -> Sequence[Sequence[bool]]:
        raise ValueError("Raw measurements are unavailable for detector-only results")

    @property
    def observables(self) -> tuple[tuple[bool, ...], ...]:
        """The observable outcomes from the simulation.

        Returns:
            tuple[tuple[bool, ...], ...]: The observable outcomes, one tuple per shot.

        """
        return tuple(tuple(shot) for shot in self._observables)


@dataclass(frozen=True)
class Result(Generic[RetType]):
    """Measurements, post-processed values, fidelity, and a guaranteed DEM."""

    _raw_measurements: list[list[bool]]
    _detector_error_model: DetectorErrorModel
    _post_processing: atom.PostProcessing[RetType]
    _fidelity_min: float
    _fidelity_max: float

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Note: The upper and lower bounds are related to branching logic in the kernel.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit.

        Returns:
            DetectorErrorModel: The STIM detector error model.

        """
        return self._detector_error_model

    @property
    def return_values(self) -> list[RetType]:
        """The return values of the kernel.

        Returns:
            list[RetType]: The return values, one per shot.

        """
        return self._return_values

    @cached_property
    def _return_values(self) -> list[RetType]:
        return list(self._post_processing.emit_return(self._raw_measurements))

    @property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes from the simulation.

        Returns:
            list[list[bool]]: The detector outcomes, one list per shot.

        """
        return self._detectors

    @cached_property
    def _detectors(self) -> list[list[bool]]:
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes used to compute detectors and observables.

        Returns:
            list[list[bool]]: The raw measurement outcomes, one list per shot.

        """
        return self._measurements

    @cached_property
    def _measurements(self) -> list[list[bool]]:
        return list(map(list, self._raw_measurements))

    @property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes from the simulation.

        Returns:
            list[list[bool]]: The observable outcomes, one list per shot.

        """
        return self._observables

    @cached_property
    def _observables(self) -> list[list[bool]]:
        return list(self._post_processing.emit_observables(self._raw_measurements))


@dataclass(frozen=True)
class _MoveProgramMetrics:
    move_depth: int
    cz_pulse_count: int
    single_qubit_gate_count: int
    total_distance_moved_um: float

    @property
    def total_gate_count(self) -> int:
        return self.cz_pulse_count + self.single_qubit_gate_count


class _SimulatorTaskBase(Generic[RetType]):
    """Shared execution runtime for compiled logical and physical tasks."""

    physical_arch_spec: ArchSpec
    physical_move_kernel: ir.Method[[], RetType]
    _post_processing: atom.PostProcessing[RetType]

    @property
    def _backend(self) -> AbstractSimulatorBackend:
        return cast(Any, self)._simulator_backend

    @property
    def _physical_kernel(self) -> ir.Method[[], RetType]:
        return cast(Any, self).physical_squin_kernel

    @property
    def _noiseless_physical_kernel(self) -> ir.Method[[], RetType]:
        return cast(Any, self).noiseless_physical_squin_kernel

    @cached_property
    def _thread_pool_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor()

    @cached_property
    def tsim_circuit(self) -> tsim_backend.Circuit:
        """Tsim-compatible circuit for the noisy physical kernel."""
        return _get_tsim_circuit(self._backend, self._physical_kernel)

    @cached_property
    def noiseless_tsim_circuit(self) -> tsim_backend.Circuit:
        """Tsim-compatible circuit for the noiseless physical kernel."""
        return _get_tsim_circuit(self._backend, self._noiseless_physical_kernel)

    @cached_property
    def detector_error_model(self) -> DetectorErrorModel:
        """Guaranteed detector error model for the noisy physical kernel."""
        return self._backend._detector_error_model(self._physical_kernel)

    @cached_property
    def _move_program_metrics(self) -> _MoveProgramMetrics:
        """Compute metrics shared by the public move-program properties."""

        move_depth = 0
        cz_pulse_count = 0
        single_qubit_gate_count = 0
        total_distance_moved_um = 0.0
        move_metric_calculator = MoveMetricCalculator(self.physical_arch_spec)
        single_qubit_gate_types = (
            move.LocalR,
            move.GlobalR,
            move.LocalRz,
            move.GlobalRz,
            move.StarRz,
        )
        state_frame, _ = atom.AtomInterpreter(
            self.physical_move_kernel.dialects,
            arch_spec=self.physical_arch_spec,
        ).run(self.physical_move_kernel)

        for stmt in self.physical_move_kernel.callable_region.walk():
            if isinstance(stmt, move.Move):
                move_depth += 1
                consumes_state = stmt.get_trait(move.ConsumesState)
                assert consumes_state is not None
                state = state_frame.get(consumes_state.get_state_argument(stmt))
                assert isinstance(state, atom.AtomState)
                total_distance_moved_um += sum(
                    move_metric_calculator.lane_distance_um(lane)
                    for lane in stmt.lanes
                    if state.data.get_qubit(
                        self.physical_arch_spec.get_endpoints(lane)[0]
                    )
                    is not None
                )
            elif isinstance(stmt, move.CZ):
                cz_pulse_count += 1
            elif isinstance(stmt, single_qubit_gate_types):
                single_qubit_gate_count += 1

        return _MoveProgramMetrics(
            move_depth=move_depth,
            cz_pulse_count=cz_pulse_count,
            single_qubit_gate_count=single_qubit_gate_count,
            total_distance_moved_um=total_distance_moved_um,
        )

    @property
    def move_depth(self) -> int:
        """Number of parallel move instructions in the compiled move program."""
        return self._move_program_metrics.move_depth

    @property
    def cz_pulse_count(self) -> int:
        """Number of CZ pulse instructions in the compiled move program."""
        return self._move_program_metrics.cz_pulse_count

    @property
    def single_qubit_gate_count(self) -> int:
        """Number of single-qubit gate instructions in the move program.

        A local or global pulse counts once, independent of how many qubits it
        addresses.
        """
        return self._move_program_metrics.single_qubit_gate_count

    @property
    def total_gate_count(self) -> int:
        """Total number of single-qubit gate and CZ pulse instructions."""
        return self._move_program_metrics.total_gate_count

    @property
    def total_distance_moved_um(self) -> float:
        """Distance travelled by all moved atoms, in micrometres."""
        return self._move_program_metrics.total_distance_moved_um

    @property
    def _loss_replace_vals(self) -> tuple[Any, Any]:
        """Return ``(loss_replace, loss)`` for backend sample conversion."""
        return (None, None)

    def visualize(
        self, animated: bool = False, interactive: bool = True, arch_vis=False
    ):
        """Visualize the physical move kernel using the built-in debugger.

        Args:
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.
            arch_vis (bool): Whether to use the interactive plotly debugger to visualize the architecture. Defaults to False.

        """
        # Visualization dependencies are optional; load them only on demand.
        from bloqade.lanes.visualize import debugger, plotly_debugger, animated_debugger

        if arch_vis:
            plotly_debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )
        else:
            if animated:
                animated_debugger(
                    self.physical_move_kernel,
                    self.physical_arch_spec,
                    interactive=interactive,
                )
            else:
                debugger(
                    self.physical_move_kernel,
                    self.physical_arch_spec,
                    interactive=interactive,
                )

    def fidelity_bounds(self) -> tuple[float, float]:
        """Compute the fidelity bounds for the physical squin kernel.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        analysis = FidelityAnalysis(self._physical_kernel.dialects)
        analysis.run(self._physical_kernel)

        min_fidelity = 1.0
        max_fidelity = 1.0
        for gate_fidelity in analysis.gate_fidelities:
            min_fidelity *= gate_fidelity.min
            max_fidelity *= gate_fidelity.max
        return min_fidelity, max_fidelity

    @staticmethod
    def _normalize_matrix(payload: Any, *, name: str, shots: int) -> list[list[bool]]:
        try:
            array = np.asarray(payload, dtype=bool)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Backend returned invalid {name} samples") from exc
        if array.ndim != 2:
            raise ValueError(f"Backend {name} samples must be a two-dimensional array")
        if array.shape[0] != shots:
            raise ValueError(
                f"Backend returned {array.shape[0]} {name} rows for {shots} shots"
            )
        return array.tolist()

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
    ) -> SimulatorResult[RetType]:
        """Run the kernel and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            SimulatorResult[RetType]: The simulation result containing measurements, detectors, and observables.

        """
        # Build the guaranteed DEM before beginning a potentially expensive
        # sampling request. This also fails early when Tsim is unavailable.
        detector_error_model = self.detector_error_model
        physical_kernel = (
            self._physical_kernel if with_noise else self._noiseless_physical_kernel
        )
        sample = self._backend.sample(physical_kernel, shots=shots)
        fidelity_min, fidelity_max = self.fidelity_bounds()

        measurements_payload = sample.measurements
        has_detectors = sample.detectors is not None
        has_observables = sample.observables is not None

        if (
            measurements_payload is not None
            and not has_detectors
            and not has_observables
        ):
            loss_replace, loss = self._loss_replace_vals
            loss_converted_measurements = self._backend._convert_loss_to_measurement(
                measurements_payload, loss_replace=loss_replace, loss=loss
            )
            measurements = self._normalize_matrix(
                loss_converted_measurements, name="measurement", shots=shots
            )
            return cast(
                SimulatorResult[RetType],
                Result(
                    measurements,
                    detector_error_model,
                    self._post_processing,
                    fidelity_min,
                    fidelity_max,
                ),
            )

        if measurements_payload is None and has_detectors and has_observables:
            detectors = self._normalize_matrix(
                sample.detectors, name="detector", shots=shots
            )
            observables = self._normalize_matrix(
                sample.observables, name="observable", shots=shots
            )
            return DetectorResult(
                _detector_error_model=detector_error_model,
                _fidelity_min=fidelity_min,
                _fidelity_max=fidelity_max,
                _detectors=detectors,
                _observables=observables,
            )

        raise ValueError(
            "Backend samples must be measurement-only or detector+observable-only"
        )

    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
    ) -> Future[SimulatorResult[RetType]]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            Future[SimulatorResult[RetType]]: A future resolving to the full simulation result.

        """
        return cast(
            Future[SimulatorResult[RetType]],
            self._thread_pool_executor.submit(self.run, shots, with_noise),
        )
