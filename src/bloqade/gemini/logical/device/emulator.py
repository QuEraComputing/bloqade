from typing import Any
from dataclasses import field, dataclass

from kirin import ir
from kirin.validation import ValidationSuite
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.transform import LogicalPipeline, MoveToSquinLogical
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC as NoiseModel

from bloqade.gemini.logical.group import kernel
from bloqade.gemini._device_emulator import GeminiDeviceEmulator, _Prepare
from bloqade.gemini.logical.device.device import (
    _default_validation_suite as validation_suite,
)
from bloqade.gemini.logical.device.result import GeminiLogicalResult


@dataclass(kw_only=True)
class GeminiLogicalDeviceEmulator(GeminiDeviceEmulator[GeminiLogicalResult[Any]]):
    """Execute Gemini logical submissions locally and return hardware-frame results."""

    _result_type = GeminiLogicalResult
    _thread_prefix = "gemini-logical-simulator"
    dialect_group: ir.DialectGroup = field(default_factory=lambda: kernel)
    validation_suite: ValidationSuite | None = field(default_factory=validation_suite)
    noise_model: NoiseModel | None = None

    def _compiler(self) -> _Prepare:
        with_noise = self.with_noise
        model = self.noise_model

        def prepare(kernel: ir.Method, arch: ArchSpec) -> tuple[ir.Method, ir.Method]:
            move = LogicalPipeline(transversal_rewrite=True, simulation=True).emit(
                kernel
            )
            noise = generate_logical_noise_model() if model is None else model
            lower = MoveToSquinLogical(
                arch_spec=arch, noise_model=noise, add_noise=with_noise
            )
            return move, lower.emit(move)

        return prepare
