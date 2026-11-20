from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)
from functools import cache, cached_property
from dataclasses import field, dataclass

from kirin import ir, passes
from kirin.dialects import py, ilist
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.lanes.transform import PhysicalPipeline, MoveToSquinPhysical
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable

from bloqade import qubit
from bloqade.gemini.post_processing import build_post_processing
from bloqade.gemini.device._task_runtime import (
    Result as Result,
    DetectorResult as DetectorResult,
    _SimulatorTaskBase,
)
from bloqade.gemini.device.simulator_backend import (
    PPVMSimulatorBackend,
    TsimSimulatorBackend,
    AbstractSimulatorBackend,
)

if TYPE_CHECKING:
    from bloqade.lanes.analysis import atom
    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.analysis.placement import PlacementStrategyABC
    from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC

RetType = TypeVar("RetType")
PhysicalResult = Result


def _default_noise_model() -> NoiseModelABC:

    return generate_simple_noise_model()


def _default_arch_spec() -> ArchSpec:

    return get_arch_spec()


def _validate_m2_matrix(matrix: list[list[int]], name: str) -> int:
    if len(matrix) == 0:
        raise ValueError(f"{name} must have at least one row")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} rows must all have the same length")
    return len(matrix)


def append_measurements_and_annotations_physical(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> None:
    """Append physical detector/observable annotations from per-block matrices.

    The method is mutated in-place. The supplied matrices are interpreted as
    measurement-to-detector/observable maps for one logical block laid out in a
    flat physical measurement list. If the kernel measures multiple such blocks,
    the matrices are repeated block-by-block. The original return value is
    preserved.
    """
    from bloqade.gemini.compile.task import (
        _insert_before,
        _find_qubit_ssas,
        _find_return_stmt,
    )

    if m2dets is None and m2obs is None:
        raise ValueError("At least one of m2dets or m2obs must be provided")

    AggressiveUnroll(mt.dialects, no_raise=True).fixpoint(mt)

    physical_qubits_per_logical_qubit: int | None = None
    if m2dets is not None:
        physical_qubits_per_logical_qubit = _validate_m2_matrix(m2dets, "m2dets")
    if m2obs is not None:
        m2obs_rows = _validate_m2_matrix(m2obs, "m2obs")
        if (
            physical_qubits_per_logical_qubit is not None
            and physical_qubits_per_logical_qubit != m2obs_rows
        ):
            raise ValueError("m2dets and m2obs must have the same number of rows")
        physical_qubits_per_logical_qubit = m2obs_rows
    assert physical_qubits_per_logical_qubit is not None

    num_physical_qubits = len(_find_qubit_ssas(mt))
    if num_physical_qubits == 0:
        raise ValueError("No physical qubit allocations found in the kernel")

    num_logical_qubits, remainder = divmod(
        num_physical_qubits, physical_qubits_per_logical_qubit
    )
    if remainder != 0:
        raise ValueError(
            "The number of physical qubits must be divisible by the number of "
            "physical qubits per logical qubit"
        )

    measure_stmts = [
        stmt
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, qubit.stmts.Measure)
    ]
    if len(measure_stmts) != 1:
        raise ValueError("Expected exactly one physical qubit.Measure statement")
    measure_stmt = measure_stmts[0]
    return_stmt = _find_return_stmt(mt)

    @cache
    def _get_physical_measurement(q_idx: int, m_idx: int) -> ir.SSAValue:
        flat_idx = q_idx * physical_qubits_per_logical_qubit + m_idx
        (idx := py.Constant(flat_idx)).insert_before(return_stmt)
        (getitem := py.GetItem(measure_stmt.result, idx.result)).insert_before(
            return_stmt
        )
        return getitem.result

    if m2dets is not None:
        for q_idx in range(num_logical_qubits):
            for j in range(len(m2dets[0])):
                meas_ssas = [
                    _get_physical_measurement(q_idx, m_idx)
                    for m_idx, row in enumerate(m2dets)
                    if row[j]
                ]
                meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
                coord_0 = _insert_before(py.Constant(float(q_idx)), return_stmt)
                coord_1 = _insert_before(py.Constant(float(j)), return_stmt)
                coords = _insert_before(
                    ilist.New([coord_0.result, coord_1.result]), return_stmt
                )
                _insert_before(
                    SetDetector(meas_list.result, coords.result), return_stmt
                )

    if m2obs is not None:
        for q_idx in range(num_logical_qubits):
            for j in range(len(m2obs[0])):
                meas_ssas = [
                    _get_physical_measurement(q_idx, m_idx)
                    for m_idx, row in enumerate(m2obs)
                    if row[j]
                ]
                meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
                _insert_before(SetObservable(meas_list.result), return_stmt)


@dataclass(frozen=True)
class PhysicalSimulatorTask(_SimulatorTaskBase[RetType], Generic[RetType]):
    """A compiled simulation task for physical SQuIn programs."""

    source_squin_kernel: ir.Method[[], RetType]
    """The input physical SQuIn kernel."""
    noise_model: NoiseModelABC
    """The physical noise model to insert into the SQuIn kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel compiled from the source SQuIn kernel."""
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    """The post-processing object for extracting detectors, observables, and return values."""
    _simulator_backend: AbstractSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    """Sampling and detector-model backend used to perform sampling."""

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel with noise channels."""

        return MoveToSquinPhysical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
        ).emit(self.physical_move_kernel)

    @cached_property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel without noise channels."""

        return MoveToSquinPhysical(
            arch_spec=self.physical_arch_spec,
        ).emit(self.physical_move_kernel)

    @property
    def _loss_replace_vals(self) -> tuple[Any, Any]:
        """Replaces `loss` values (None) with `loss_replace` (True)."""
        return (True, None)


@dataclass
class GeminiPhysicalSimulator:
    """Simulator for physical-pipeline-compatible ``ir.Method`` kernels.

    Callers supply terminal physical measurement and any desired annotations. The
    simulator performs no conversion or insertion.
    """

    noise_model: NoiseModelABC = field(default_factory=_default_noise_model)
    """The physical noise model used for simulation."""
    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""
    backend: AbstractSimulatorBackend = field(default_factory=PPVMSimulatorBackend)
    """Backend for created tasks; defaults to PPVM sampling with Tsim DEMs."""
    place_opt_type: type[passes.Pass] | None = None
    """Optional placement pass used for compilation."""
    placement_strategy: PlacementStrategyABC | None = None
    """Optional physical placement strategy."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
    ) -> PhysicalSimulatorTask[RetType]:
        """Compile a physical-pipeline-compatible ``ir.Method`` into a task.

        The method must already contain terminal physical measurement and any
        desired annotations; no conversion or insertion is performed.
        """
        if not isinstance(physical_kernel, ir.Method):
            raise TypeError("GeminiPhysicalSimulator.task() requires a Squin ir.Method")

        # Physical compilation mutates its input. Keep the method supplied by
        # the caller reusable.
        source_squin_kernel = physical_kernel.similar()

        place_opt_type = self.place_opt_type or SequentialPlacePass

        physical_pipeline = PhysicalPipeline(
            arch_spec=self.arch_spec,
            place_opt_type=place_opt_type,
            placement_strategy=self.placement_strategy,
        )
        physical_move_kernel = physical_pipeline.emit(
            source_squin_kernel, no_raise=False
        )
        post_processing = build_post_processing(source_squin_kernel)

        return PhysicalSimulatorTask(
            source_squin_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
            post_processing,
            self.backend,
        )


PhysicalSimulator = GeminiPhysicalSimulator
