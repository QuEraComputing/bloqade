from __future__ import annotations

import re
import abc
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
from _thread import LockType
from weakref import WeakKeyDictionary
from threading import Lock
from dataclasses import field, dataclass

import numpy as np
from kirin import ir, rewrite
from kirin.rewrite.abc import RewriteRule, RewriteResult
from bloqade.lanes.rewrite.squin2stim import RemoveReturn

if TYPE_CHECKING:
    from stim import Circuit as StimCircuit, DetectorErrorModel
    from tsim import Circuit as TsimCircuit  # type: ignore[reportMissingImports]


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be a non-bool int in the range [0, 2**63).")


def _next_child_seed(
    rng_state: np.random.Generator | None, rng_lock: LockType
) -> int | None:
    """Reserve the next native-sampler seed from a backend's root stream."""
    if rng_state is None:
        return None
    with rng_lock:
        return int(rng_state.integers(0, 2**63))


@dataclass(frozen=True)
class BackendSample:
    """Raw samples returned by a simulator backend.

    A backend may return measurements, or it may return detector and observable
    arrays directly. Task execution validates and normalizes the selected form.
    """

    measurements: np.ndarray | None = None
    detectors: np.ndarray | None = None
    observables: np.ndarray | None = None


class AbstractSimulatorBackend(abc.ABC):
    """Sampling and detector-model contract for Gemini simulators."""

    @abc.abstractmethod
    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
    ) -> BackendSample:
        """Sample a compiled physical SQuIn kernel."""

    @abc.abstractmethod
    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        """Build the detector error model for a physical SQuIn kernel."""

    @staticmethod
    def _convert_loss_to_measurement(
        shots_arr: np.ndarray,
        loss_replace: Any = None,
        loss: Any = None,
    ) -> np.ndarray:
        """Convert loss values to measurement values when the backend supports it."""
        return shots_arr


@runtime_checkable
class _TsimCircuitCapability(Protocol):
    """Structural capability for backends that can provide a Tsim circuit."""

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit: ...


def _get_tsim_circuit(
    backend: AbstractSimulatorBackend, physical_squin_kernel: ir.Method
) -> TsimCircuit:
    if not isinstance(backend, _TsimCircuitCapability):
        raise TypeError(
            f"{type(backend).__name__} does not provide Tsim circuit compatibility"
        )
    return backend._tsim_circuit(physical_squin_kernel)


def _tsim() -> Any:
    try:
        from bloqade import tsim
    except ImportError as exc:
        raise ImportError(
            "Tsim simulation could not load the Tsim integration. "
            "`bloqade-tsim` is a required dependency of `bloqade`. "
            "Reinstall `bloqade-tsim` in your environment."
        ) from exc

    return tsim


@dataclass
class TsimSimulatorBackend(AbstractSimulatorBackend):
    """Backend that uses Tsim for both sampling and DEM generation.

    ``seed`` is a construction-time root seed. Each seeded sampling request
    receives the next derived child seed from this backend's private stream.
    """

    seed: int | None = None
    run_detectors: bool = False
    _circuits: WeakKeyDictionary[ir.Method, TsimCircuit] = field(
        default_factory=WeakKeyDictionary, init=False, repr=False
    )
    _rng_state: np.random.Generator | None = field(init=False, repr=False)
    _rng_lock: LockType = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        self._rng_state = (
            np.random.default_rng(self.seed) if self.seed is not None else None
        )

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        """Convert a physical SQuIn kernel to a cached Tsim circuit."""

        try:
            return self._circuits[physical_squin_kernel]
        except KeyError:
            pass

        owned_kernel = physical_squin_kernel.similar()
        rewrite.Walk(RemoveReturn()).rewrite(owned_kernel.code)
        circuit = _tsim().Circuit(owned_kernel)
        self._circuits[physical_squin_kernel] = circuit
        return circuit

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
    ) -> BackendSample:
        seed = _next_child_seed(self._rng_state, self._rng_lock)
        circuit = self._tsim_circuit(physical_squin_kernel)
        if self.run_detectors:
            return self._sample_detectors(circuit, shots, seed=seed)

        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler(seed=seed)
        else:
            sampler = circuit.compile_sampler(seed=seed)

        return BackendSample(
            measurements=np.asarray(sampler.sample(shots=shots), dtype=bool)
        )

    @staticmethod
    def _sample_detectors(
        circuit: TsimCircuit, shots: int, seed: int | None = None
    ) -> BackendSample:
        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler(seed=seed)
            converter = circuit.compile_m2d_converter(skip_reference_sample=True)
            measurements = sampler.sample(shots=shots)
            detectors, observables = converter.convert(
                measurements=measurements, separate_observables=True
            )
        else:
            sampler = circuit.compile_detector_sampler(seed=seed)
            detectors, observables = sampler.sample(
                shots=shots, separate_observables=True
            )

        return BackendSample(
            detectors=np.asarray(detectors, dtype=bool),
            observables=np.asarray(observables, dtype=bool),
        )

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        return self._tsim_circuit(physical_squin_kernel).detector_error_model(
            approximate_disjoint_errors=True
        )


def _clifft_compatible_stim_text(circuit: Any) -> str:
    """Strip Stim instruction tags that CliffT does not currently parse."""

    return "\n".join(
        re.sub(r"^([A-Z][A-Z0-9_]*)(\[[^\]\n]+\])", r"\1", line)
        for line in str(circuit).splitlines()
    )


_PPVM_CORRELATED_LOSS_EVENT = re.compile(
    r"^(?P<instruction>\s*I_ERROR)"
    r"\[correlated_loss:(?P<event_id>\d+)\]"
    r"(?P<arguments>\([^)\n]*\))"
    r"(?P<targets>\s+[^#\n]+)"
    r"(?P<comment>#.*)?$"
)


def _ppvm_compatible_stim_text(circuit: str | StimCircuit) -> str:
    """Convert paired Bloqade correlated-loss tags to PPVM syntax."""

    converted_lines: list[str] = []
    for line in str(circuit).splitlines():
        match = _PPVM_CORRELATED_LOSS_EVENT.match(line)
        if match is None:
            converted_lines.append(line)
            continue

        targets = match.group("targets").split()
        if len(targets) != 2:
            event_id = match.group("event_id")
            raise ValueError(
                "PPVM requires each Bloqade correlated-loss event to target "
                f"exactly two qubits, but correlated_loss:{event_id} targets "
                f"{len(targets)}."
            )

        converted_lines.append(
            f"{match.group('instruction')}[correlated_loss]"
            f"{match.group('arguments')}{match.group('targets')}"
            f"{match.group('comment') or ''}"
        )

    return "\n".join(converted_lines)


def _clifft() -> Any:
    try:
        import clifft  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "CliffT simulation requires the optional `clifft` dependency. "
            "Install it with `bloqade[sim]` or include `clifft` "
            "in your environment."
        ) from exc

    return clifft


def _ppvm() -> Any:
    try:
        import ppvm  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "PPVM simulation could not import `ppvm` from `bloqade-ppvm`, "
            "a required dependency of `bloqade`. "
            "Reinstall `bloqade-ppvm` in your environment."
        ) from exc

    return ppvm


def _clifft_tsim_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "CliffT simulation requires Tsim for physical SQuIn conversion and "
        "detector error model generation, even when CliffT is selected for "
        "sampling. `bloqade-tsim` is a required dependency of `bloqade`. "
        "Reinstall `bloqade-tsim` in your environment."
    )


def _ppvm_tsim_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "PPVM simulation requires Tsim for physical SQuIn conversion and "
        "detector error model generation, even when PPVM is selected for "
        "sampling. `bloqade-tsim` is a required dependency of `bloqade`. "
        "Reinstall `bloqade-tsim` in your environment."
    )


@dataclass
class CliffTSimulatorBackend(AbstractSimulatorBackend):
    """Backend using Tsim for conversion/DEM generation and CliffT for sampling.

    ``seed`` is a construction-time root seed. Each seeded sampling request
    receives the next derived child seed from this backend's private stream.
    """

    seed: int | None = None
    _tsim_backend: TsimSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    _programs: WeakKeyDictionary[ir.Method, Any] = field(
        default_factory=WeakKeyDictionary, init=False, repr=False
    )
    _rng_state: np.random.Generator | None = field(init=False, repr=False)
    _rng_lock: LockType = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        self._rng_state = (
            np.random.default_rng(self.seed) if self.seed is not None else None
        )

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        try:
            return self._tsim_backend._tsim_circuit(physical_squin_kernel)
        except ImportError as exc:
            raise _clifft_tsim_import_error(exc) from exc

    def _clifft_program(self, physical_squin_kernel: ir.Method) -> Any:
        try:
            return self._programs[physical_squin_kernel]
        except KeyError:
            pass

        program = _clifft().compile(
            _clifft_compatible_stim_text(self._tsim_circuit(physical_squin_kernel))
        )
        self._programs[physical_squin_kernel] = program
        return program

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
    ) -> BackendSample:
        sample_kwargs: dict[str, int] = {"shots": int(shots)}
        effective_seed = _next_child_seed(self._rng_state, self._rng_lock)
        if effective_seed is not None:
            sample_kwargs["seed"] = effective_seed

        sample_result = _clifft().sample(
            self._clifft_program(physical_squin_kernel), **sample_kwargs
        )
        return BackendSample(
            measurements=np.asarray(sample_result.measurements, dtype=bool)
        )

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        try:
            return self._tsim_backend._detector_error_model(physical_squin_kernel)
        except ImportError as exc:
            raise _clifft_tsim_import_error(exc) from exc


def _pyqrack_tsim_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "PyQrack simulation requires Tsim for detector error model generation, "
        "even when PyQrack is selected for sampling. `bloqade-tsim` is a "
        "required dependency of `bloqade`. Reinstall `bloqade-tsim` in your "
        "environment."
    )


@dataclass
class PPVMSimulatorBackend(AbstractSimulatorBackend):
    """
    Backend using Tsim for conversion/DEM generation and PPVM for sampling.

    ``seed`` is a construction-time root seed. Each seeded sampling request
    receives the next derived child seed from this backend's private stream.
    """

    seed: int | None = None
    _tsim_backend: TsimSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    _programs: WeakKeyDictionary[ir.Method, Any] = field(
        default_factory=WeakKeyDictionary, init=False, repr=False
    )
    _rng_state: np.random.Generator | None = field(init=False, repr=False)
    _rng_lock: LockType = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        _validate_seed(self.seed)
        self._rng_state = (
            np.random.default_rng(self.seed) if self.seed is not None else None
        )

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        try:
            return self._tsim_backend._tsim_circuit(physical_squin_kernel)
        except ImportError as exc:
            raise _ppvm_tsim_import_error(exc) from exc

    def _ppvm_program(self, physical_squin_kernel: ir.Method) -> Any:
        try:
            return self._programs[physical_squin_kernel]
        except KeyError:
            pass

        ppvm = _ppvm()
        tsim_circuit = self._tsim_circuit(physical_squin_kernel)
        stim_text = _ppvm_compatible_stim_text(tsim_circuit.stim_circuit)

        program = ppvm.StimProgram.parse(
            stim_text
        )  # parses and validates compatibility
        self._programs[physical_squin_kernel] = program
        return program

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
    ) -> BackendSample:
        effective_seed = _next_child_seed(self._rng_state, self._rng_lock)

        program = self._ppvm_program(physical_squin_kernel)

        ppvm = _ppvm()
        samples = ppvm.sample_stim(
            program,
            n_qubits=program.num_qubits,
            num_shots=int(shots),
            seed=effective_seed,
        )

        raw_measurements = np.asarray(samples)

        measurements = np.empty(raw_measurements.shape, dtype=object)
        measurements[raw_measurements == ppvm.MeasurementResult.ZERO] = False
        measurements[raw_measurements == ppvm.MeasurementResult.ONE] = True
        measurements[raw_measurements == ppvm.MeasurementResult.LOST] = None

        return BackendSample(measurements=measurements)

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        try:
            return self._tsim_backend._detector_error_model(physical_squin_kernel)
        except ImportError as exc:
            raise _ppvm_tsim_import_error(exc) from exc

    @staticmethod
    def _convert_loss_to_measurement(
        shots_arr: np.ndarray,
        loss_replace: Any = None,
        loss: Any = None,
    ) -> np.ndarray:
        """Applies the conversion of `loss` values to `loss_replace`."""
        shots_arr[shots_arr == loss] = loss_replace
        return shots_arr.astype(bool)


class _RemovePyQrackAnnotations(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        from bloqade.decoders.dialects.annotate.stmts import (
            SetDetector,
            SetObservable,
        )

        if isinstance(node, (SetDetector, SetObservable)):
            node.delete()
            return RewriteResult(has_done_something=True)
        return RewriteResult()


@dataclass
class _PyQrackSimulatorBackend(AbstractSimulatorBackend):
    """Backend using PyQrack for sampling and Tsim for guaranteed DEM generation.

    The backend seed deterministically derives the seed for each native Qrack
    simulator. Exact native measurement outcomes remain dependent on Qrack's
    platform-specific random-number implementation.
    """

    seed: int | None = None
    options: dict[str, Any] | None = None
    min_qubits: int = 0
    _tsim_backend: TsimSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    _rng_state: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        self._rng_state = np.random.default_rng(self.seed)

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        try:
            return self._tsim_backend._tsim_circuit(physical_squin_kernel)
        except ImportError as exc:
            raise _pyqrack_tsim_import_error(exc) from exc

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
    ) -> BackendSample:
        from bloqade.pyqrack.reg import MeasurementResultValue
        from bloqade.pyqrack.base import PyQrackInterpreter
        from bloqade.pyqrack.task import PyQrackSimulatorTask
        from bloqade.pyqrack.device import StackMemorySimulator

        rng_state = self._rng_state

        class _RecordingPyQrackInterpreter(PyQrackInterpreter):
            measurements: list[MeasurementResultValue]

            def initialize(self):
                self.measurements = []
                # This resets memory and constructs the fresh QrackSimulator.
                initialized = super().initialize()

                # Give each one-shot Qrack simulator a seed from this backend's
                # persistent random-number stream.
                qrack_seed = int(initialized.rng_state.integers(0, 2**63))
                initialized.memory.sim_reg.seed(qrack_seed)

                return initialized

            def set_global_measurement_id(self, m):
                super().set_global_measurement_id(m)
                self.measurements.append(m)

        class _RecordingStackMemorySimulator(StackMemorySimulator):
            def new_task(self, mt, args, kwargs, memory):
                interpreter = _RecordingPyQrackInterpreter(
                    mt.dialects,
                    memory=memory,
                    rng_state=self.rng_state,
                    loss_m_result=self.loss_m_result,
                )
                return PyQrackSimulatorTask(
                    kernel=mt,
                    args=args,
                    kwargs=kwargs,
                    pyqrack_interp=interpreter,
                )

        # Annotation removal mutates its input, so prepare one owned kernel for
        # this sampling request and reuse it only to construct fresh shot tasks.
        owned_kernel = physical_squin_kernel.similar()
        rewrite.Walk(_RemovePyQrackAnnotations()).rewrite(owned_kernel.code)
        simulator = _RecordingStackMemorySimulator(
            options=cast(Any, self.options or {}),
            rng_state=rng_state,
            min_qubits=self.min_qubits,
        )
        recorded_measurements: list[list[bool]] = []
        for _ in range(shots):
            task = simulator.task(owned_kernel)
            task.run()
            interpreter = cast(_RecordingPyQrackInterpreter, task.pyqrack_interp)
            recorded_measurements.append(
                [
                    self._measurement_to_bool(measurement, MeasurementResultValue)
                    for measurement in interpreter.measurements
                ]
            )

        return BackendSample(measurements=np.asarray(recorded_measurements, dtype=bool))

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        try:
            return self._tsim_backend._detector_error_model(physical_squin_kernel)
        except ImportError as exc:
            raise _pyqrack_tsim_import_error(exc) from exc

    @staticmethod
    def _measurement_to_bool(measurement: Any, measurement_result_value: Any) -> bool:
        if measurement == measurement_result_value.Zero:
            return False
        if measurement == measurement_result_value.One:
            return True
        raise ValueError(f"Unsupported PyQrack measurement result: {measurement!r}")
