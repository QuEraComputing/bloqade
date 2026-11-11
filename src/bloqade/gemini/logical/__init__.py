from bloqade.qubit import qalloc as qalloc
from kirin.dialects import ilist as ilist
from bloqade.cirq_registry import register_cirq_loader as _register_cirq_loader
from bloqade.lanes.dialects.arch import loc as loc
from bloqade.squin.stdlib.simple import (
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
    cnot as cnot,
    swap as swap,
    s_adj as s_adj,
    shift as shift,
    t_adj as t_adj,
    sqrt_x as sqrt_x,
    sqrt_y as sqrt_y,
    sqrt_z as sqrt_z,
    sqrt_x_adj as sqrt_x_adj,
    sqrt_y_adj as sqrt_y_adj,
    sqrt_z_adj as sqrt_z_adj,
)
from bloqade.lanes.dialects.arrange import move_to as move_to
from bloqade.lanes.dialects.steane713 import terminal_measure as terminal_measure

from bloqade.gemini.logical import (
    impl as impl,
    extensions as extensions,
    validation as validation,
)
from bloqade.gemini.logical.group import kernel as kernel
from bloqade.gemini.logical.extensions import star_rz as star_rz
from bloqade.gemini.logical.stdlib import (
    broadcast as broadcast,
    qalloc_at as qalloc_at,
    default_post_processing as default_post_processing,
)
from bloqade.gemini.logical.studio import to_studio_url as to_studio_url


def _get_cirq_loader():
    from bloqade.gemini.logical.cirq_conversion import GeminiLogicalCirqLowerer

    return GeminiLogicalCirqLowerer


# Keep Cirq optional: load its Gemini integration only when conversion is used.
_register_cirq_loader(dialects=kernel, factory=_get_cirq_loader)

__all__ = [
    "_register_cirq_loader",
    "broadcast",
    "cnot",
    "cx",
    "cy",
    "cz",
    "default_post_processing",
    "extensions",
    "h",
    "ilist",
    "impl",
    "kernel",
    "loc",
    "move_to",
    "qalloc",
    "qalloc_at",
    "rx",
    "ry",
    "rz",
    "s",
    "s_adj",
    "shift",
    "sqrt_x",
    "sqrt_x_adj",
    "sqrt_y",
    "sqrt_y_adj",
    "sqrt_z",
    "sqrt_z_adj",
    "star_rz",
    "swap",
    "t",
    "t_adj",
    "terminal_measure",
    "to_studio_url",
    "u3",
    "validation",
    "x",
    "y",
    "z",
]
