from bloqade.cirq_registry import register_cirq_loader as _register_cirq_loader

from bloqade.gemini.physical.group import kernel as kernel
from bloqade.gemini.physical.device.result import (
    GeminiPhysicalResult as GeminiPhysicalResult,
)
from bloqade.gemini.physical.device.emulator import (
    GeminiPhysicalDeviceEmulator as GeminiPhysicalDeviceEmulator,
)


def _get_cirq_loader():
    from bloqade.gemini.common.cirq_conversion import GeminiCirqLowerer

    return GeminiCirqLowerer


# Cirq is optional; loading this integration also installs the shared
# ``new_at`` emitter for physical kernels.
_register_cirq_loader(dialects=kernel, factory=_get_cirq_loader)

__all__ = [
    "GeminiPhysicalResult",
    "GeminiPhysicalDeviceEmulator",
    "_register_cirq_loader",
    "kernel",
]
