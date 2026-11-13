from __future__ import annotations

from typing import cast
from collections.abc import Mapping

from kirin import ir
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini.decoding.kernels import (
    _DecoderPrimitiveSet,
    _LogicalTomographyReturn,
    _produce_tomography_kernels,
    _build_tomography_primitives,
)


def _default_post_processing() -> ir.Method[..., _LogicalTomographyReturn]:
    from bloqade.gemini.logical.stdlib import default_post_processing

    return cast(ir.Method[..., _LogicalTomographyReturn], default_post_processing)


# NOTE: this is basically what the user would "instantiate" for the specific
# MSD experiment in the "Experimental Demonstration of MSD" paper
def _build_msd_primitives(
    theta: float,
    phi: float,
    lam: float,
    log: bool = True,
) -> _DecoderPrimitiveSet:
    """Build the state-preparation and logical-circuit primitives for MSD.

    Args:
        theta: U3 ``theta`` angle for input magic-state preparation.
        phi: U3 ``phi`` angle for input magic-state preparation.
        lam: U3 ``lambda`` angle for input magic-state preparation.
        log: If true, print a progress message.

    Returns:
        Primitive Squin kernels for the MSD state injection and logical circuit.
    """

    if log:
        print("Building MSD primitives...")

    @squin.kernel
    def msd_magic_prep(reg):
        squin.broadcast.u3(theta, phi, lam, reg)

    @squin.kernel
    def msd_forward(reg):
        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_x_adj(reg)

    return _DecoderPrimitiveSet(
        state_injection_circuit=msd_magic_prep,
        logical_circuit=msd_forward,
    )


def _build_decoder_kernel_bundle(
    primitive_set: _DecoderPrimitiveSet,
    num_logical_qubits: int = 5,
    tomography_kernels: Mapping[str, ir.Method[..., None]] | None = None,
) -> dict[str, ir.Method[..., _LogicalTomographyReturn]]:
    """Build basis-labeled MSD tomography kernels.

    Args:
        primitive_set: Primitive state-injection and logical-circuit kernels.
        num_logical_qubits: Number of logical qubits in the MSD logical circuit.
        tomography_kernels: Optional tomography operations keyed by basis.

    Returns:
        Basis-labeled tomography kernels.
    """

    tomography_primitives = (
        dict(tomography_kernels)
        if tomography_kernels is not None
        else _build_tomography_primitives(output_qubit=0)
    )
    state_injection_circuit = primitive_set.state_injection_circuit
    logical_circuit = primitive_set.logical_circuit

    @squin.kernel
    def actual_logical_kernel(reg):
        state_injection_circuit(reg)
        logical_circuit(reg)

    default_post_processing = _default_post_processing()
    actual_kernels = _produce_tomography_kernels(
        num_logical_qubits,
        actual_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "msd_actual",
    )

    return dict(actual_kernels)
