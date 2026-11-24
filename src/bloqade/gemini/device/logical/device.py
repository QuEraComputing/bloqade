from dataclasses import field, dataclass
from importlib.metadata import version

from kirin import ir
from kirin.validation import ValidationSuite
from bloqade.core.device import Device
from bloqade.core.device.task import (
    KernelBatchTask,
    SingleKernelTask,
    ParameterScanTask,
)
from bloqade.lanes.validation.steane import SteaneLogicalValidation
from bloqade.lanes.validation.const_address import ConstAddressValidation
from bloqade.lanes.validation.duplicate_address import DuplicateAddressValidation
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
)

from bloqade.gemini.logical.group import kernel as logical_kernel
from bloqade.gemini.device.logical.task import (
    GeminiKernelBatchTask,
    GeminiSingleKernelTask,
    GeminiParameterScanTask,
)
from bloqade.gemini.device.logical.future import GeminiLogicalFuture
from bloqade.gemini.logical.validation.arguments import GeminiLogicalArgumentValidation
from bloqade.gemini.logical.validation.extensions.analysis import (
    NoLogicalExtensionsValidation,
)

_bloqade_version = version("bloqade-circuit")


def _default_validation_suite() -> ValidationSuite:
    """Build an independent validation suite for Gemini logical submission."""
    return ValidationSuite(
        [
            SteaneLogicalValidation,
            SteaneTerminalMeasurementValidation,
            FlatKernelNoCloningValidation,
            ConstAddressValidation,
            DuplicateAddressValidation,
            # NOTE: not redundant with the `@logical.kernel` decorator. A
            # method can be built without it and handed straight to the
            # device, so submission is its own enforcement point.
            GeminiLogicalArgumentValidation,
            # The experimental extensions leave the logical subspace and
            # are only meaningful with post-selection the user applies
            # themselves. They remain available through GeminiLogicalSimulator,
            # which does not run this suite, but are rejected by devices using
            # the hardware submission policy.
            NoLogicalExtensionsValidation,
        ]
    )


@dataclass(kw_only=True)
class GeminiLogicalDevice(Device[GeminiLogicalFuture]):
    """Device that builds tasks for the Gemini logical backend.

    Wires the Gemini-specific task and future classes into the generic
    `Device` factory and defaults `context_name` to "gemini-logical".
    """

    program_language: str = "squin"
    language_version: str = _bloqade_version
    validation_suite: ValidationSuite | None = field(
        default_factory=_default_validation_suite
    )
    # Annotated ``| None`` to match ``Device.dialect_group`` as of
    # bloqade-core 0.6.8, which widened it. The default_factory keeps this
    # concrete in practice; the union is here only so the override stays
    # type-compatible with the base.
    dialect_group: ir.DialectGroup | None = field(
        default_factory=lambda: logical_kernel
    )

    future_cls: type[GeminiLogicalFuture] = GeminiLogicalFuture
    single_kernel_task_cls: type[SingleKernelTask[GeminiLogicalFuture]] = field(
        default=GeminiSingleKernelTask,
        init=False,
    )
    parameter_scan_task_cls: type[ParameterScanTask[GeminiLogicalFuture]] = field(
        default=GeminiParameterScanTask,
        init=False,
    )
    kernel_batch_task_cls: type[KernelBatchTask[GeminiLogicalFuture]] = field(
        default=GeminiKernelBatchTask,
        init=False,
    )

    context_name: str = "gemini-logical"
