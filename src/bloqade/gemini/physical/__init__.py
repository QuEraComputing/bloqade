from bloqade.squin import (
    h as h,
    s as s,
    t as t,
    x as x,
    y as y,
    z as z,
    cx as cx,
    cy as cy,
    cz as cz,
    rx as rx,
    ry as ry,
    rz as rz,
    u3 as u3,
    ccz as ccz,
    cnot as cnot,
    swap as swap,
    reset as reset,
    s_adj as s_adj,
    shift as shift,
    t_adj as t_adj,
    is_one as is_one,
    qalloc as qalloc,
    sqrt_x as sqrt_x,
    sqrt_y as sqrt_y,
    sqrt_z as sqrt_z,
    is_lost as is_lost,
    is_zero as is_zero,
    measure as measure,
    broadcast as broadcast,
    phased_xz as phased_xz,
    sqrt_x_adj as sqrt_x_adj,
    sqrt_y_adj as sqrt_y_adj,
    sqrt_z_adj as sqrt_z_adj,
    set_detector as set_detector,
    set_observable as set_observable,
)
from bloqade.cirq_registry import register_cirq_loader as _register_cirq_loader

from bloqade.gemini.physical.group import kernel as kernel
from bloqade.gemini.physical.simulation import (
    GeminiPhysicalSimulator as GeminiPhysicalSimulator,
)
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
    "GeminiPhysicalDeviceEmulator",
    "GeminiPhysicalResult",
    "GeminiPhysicalSimulator",
    "_register_cirq_loader",
    "broadcast",
    "ccz",
    "cnot",
    "cx",
    "cy",
    "cz",
    "h",
    "is_lost",
    "is_one",
    "is_zero",
    "kernel",
    "measure",
    "phased_xz",
    "qalloc",
    "reset",
    "rx",
    "ry",
    "rz",
    "s",
    "s_adj",
    "set_detector",
    "set_observable",
    "shift",
    "sqrt_x",
    "sqrt_x_adj",
    "sqrt_y",
    "sqrt_y_adj",
    "sqrt_z",
    "sqrt_z_adj",
    "swap",
    "t",
    "t_adj",
    "u3",
    "x",
    "y",
    "z",
]
