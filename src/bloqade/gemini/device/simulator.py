from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)
from functools import cached_property
from dataclasses import field, dataclass

from kirin import ir
from bloqade.lanes.transform import MoveToSquinLogical
from bloqade.lanes.noise_model import generate_logical_noise_model

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
    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

RetType = TypeVar("RetType")


def _default_noise_model() -> LogicalNoiseModelABC:

    return generate_logical_noise_model()


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(_SimulatorTaskBase[RetType], Generic[RetType]):
    """A compiled simulation task for the Gemini logical simulator.

    Created by :meth:`GeminiLogicalSimulator.task`. The squin-to-move compilation
    and post-processing extraction are performed eagerly at construction time.
    Simulation artifacts (physical squin kernel, stim circuits, samplers, detector
    error model) are computed lazily on first access since they depend on the
    noise model.
    """

    logical_squin_kernel: ir.Method[[], RetType]
    """The input logical squin kernel to be executed on the Gemini architecture."""
    noise_model: LogicalNoiseModelABC
    """The noise model to be inserted into the physical squin kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel that executes the logical squin kernel on the physical architecture."""
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    """The post-processing object for extracting detectors, observables, and return values."""
    _simulator_backend: AbstractSimulatorBackend = field(
        default_factory=TsimSimulatorBackend
    )

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel with noise channels."""

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=True,
        ).emit(self.physical_move_kernel)

    @cached_property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel without noise channels."""

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=False,
        ).emit(self.physical_move_kernel)

    @property
    def _loss_replace_vals(self) -> tuple[Any, Any]:
        """Replaces `loss` values (None) with `loss_replace` (True)."""
        return (True, None)


@dataclass
class GeminiLogicalSimulator:
    """This is the primary entry point for compiling and simulating logical quantum
    circuits on the Gemini architecture. Use :meth:`task` to compile a kernel into
    a reusable :class:`GeminiLogicalSimulatorTask`.

    CUDA-Q users must convert kernels and add any desired detector/observable
    annotations externally.
    """

    noise_model: LogicalNoiseModelABC = field(default_factory=_default_noise_model)
    """The noise model used for simulation. Defaults to :func:`generate_logical_noise_model`."""
    backend: AbstractSimulatorBackend = field(default_factory=PPVMSimulatorBackend)
    """Sampling backend for created tasks; defaults to PPVM."""

    def task(
        self,
        logical_kernel: ir.Method[[], RetType],
    ) -> GeminiLogicalSimulatorTask[RetType]:
        """Create a simulation task for the given kernel.

        Eagerly compiles the kernel through squin-to-move and extracts post-processing.
        Kernels are expected to be SQuIN kernels (for example, CUDA-Q kernels are expected
        to be first converted to SQuIN).

        CUDA-Q conversion and any desired detector/observable annotations must be
        completed externally.

        Args:
            logical_kernel (ir.Method[[], RetType]): The logical
                squin kernel to compile and run.

        Returns:
            GeminiLogicalSimulatorTask[RetType]: The compiled simulation task.
        """
        if not isinstance(logical_kernel, ir.Method):
            raise TypeError("GeminiLogicalSimulator.task() requires a Squin ir.Method")

        from bloqade.gemini.compile import compile_task

        (
            logical_squin_kernel,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
        ) = compile_task(logical_kernel)
        return GeminiLogicalSimulatorTask(
            logical_squin_kernel,
            self.noise_model,
            physical_arch_spec,
            physical_move_kernel,
            post_processing,
            self.backend,
        )
