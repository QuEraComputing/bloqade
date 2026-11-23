from typing import Any
from dataclasses import field, dataclass

from kirin import ir
from kirin.validation import ValidationSuite
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.transform import PhysicalPipeline, MoveToSquinPhysical
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC as NoiseModel

from bloqade.gemini.physical.group import (
    kernel,
    physical_validation_suite as validation_suite,
)
from bloqade.gemini._device_emulator import GeminiDeviceEmulator, _Prepare
from bloqade.gemini.physical.device.result import GeminiPhysicalResult


@dataclass(kw_only=True)
class GeminiPhysicalDeviceEmulator(GeminiDeviceEmulator[GeminiPhysicalResult[Any]]):
    """Execute Gemini physical submissions locally and return hardware-frame results."""

    _result_type = GeminiPhysicalResult
    _thread_prefix = "gemini-physical-simulator"
    dialect_group: ir.DialectGroup = field(default_factory=lambda: kernel)
    validation_suite: ValidationSuite | None = field(default_factory=validation_suite)
    noise_model: NoiseModel | None = None

    def _compiler(self) -> _Prepare:
        with_noise = self.with_noise
        model = self.noise_model

        def prepare(kernel: ir.Method, arch: ArchSpec) -> tuple[ir.Method, ir.Method]:
            move = PhysicalPipeline(arch_spec=arch).emit(kernel, no_raise=False)
            lower = MoveToSquinPhysical(arch_spec=arch)
            if with_noise:
                lower.noise_model = (
                    generate_simple_noise_model() if model is None else model
                )
            return move, lower.emit(move)

        return prepare
