from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)
from functools import cached_property
from dataclasses import field, dataclass

from kirin import ir, passes
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform import PhysicalPipeline, MoveToSquinPhysical
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.arch.gemini.physical import get_arch_spec

from bloqade.gemini.utils import (
    append_measurements_and_annotations_physical as append_measurements_and_annotations_physical,
)
from bloqade.simulation.backends import (
    PPVMSimulatorBackend,
    TsimSimulatorBackend,
    AbstractSimulatorBackend,
)
from bloqade.gemini._task_runtime import (
    Result as Result,
    DetectorResult as DetectorResult,
    _SimulatorTaskBase,
)
from bloqade.gemini.post_processing import build_post_processing

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
