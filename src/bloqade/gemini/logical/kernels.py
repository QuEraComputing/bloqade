"""Kernel builders for logical tomography and magic-state-distillation experiments."""

from __future__ import annotations

import math
from typing import Any, TypeVar, cast
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from kirin import ir
from kirin.dialects import ilist
from bloqade.decoders.dialects.annotate.types import Detector, Observable

from bloqade import qubit, squin

_LogicalTomographyReturn = tuple[
    ilist.IList[Detector, Any],
    ilist.IList[Observable, Any],
]
_ReturnT = TypeVar("_ReturnT")


# TODO: In principle, these tomography kernels aren't gemini-specific.
# BUT, right now, the kernels are written in the gemini logical dialect.
# Can maybe move this code to bloqade-core.
@dataclass(frozen=True)
class PostselectionExperimentKernels:
    """Primitive Squin kernels needed to build MSD decoder tasks.

    Attributes:
        state_injection_circuit: Squin kernel that prepares the input logical
            magic states.
        logical_circuit: Squin kernel for the logical circuit under test.
    """

    state_injection_circuit: ir.Method[..., None]
    logical_circuit: ir.Method[..., None]


def build_tomography_kernels(*, output_qubit: int) -> dict[str, ir.Method[..., None]]:
    """Build X/Y/Z tomography-basis Squin kernels for one output qubit."""

    @squin.kernel
    def tomography_x(reg):
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_y(reg):
        squin.sqrt_z_adj(reg[output_qubit])
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_z(reg):
        return

    # TODO: stick to X, Y, Z for now. Work with pauli strings by default.
    return {"X": tomography_x, "Y": tomography_y, "Z": tomography_z}


# TODO: validate that the tomography kernels ONLY contain clifford operations (and no non-clifford
# operations). It's possible to construct a "tomography circuit" with non-clifford operations
# and attempt to add nonclifford gates at the end of the circuit
def _produce_tomography_kernels(
    num_qubits: int,
    logical_kernel: ir.Method[..., None],
    tomography_kernels: Mapping[str, ir.Method[..., None]],
    return_val_fn: ir.Method[..., _ReturnT] | Callable[[object], _ReturnT],
    kernel_name: str,
    *,
    supply_reg: bool = True,
) -> Mapping[str, ir.Method[..., _ReturnT]]:
    """Compose logical and tomography kernels into labeled tomography kernels.

    Args:
        num_qubits: Number of logical qubits to allocate when ``supply_reg`` is
            true.
        logical_kernel: Kernel applied before the tomography rotation.
        tomography_kernels: Mapping from tomography-kernel name to tomography
            rotation kernel.
        return_val_fn: Callable applied to the logical register to produce the
            returned value for allocated Gemini logical kernels.
        kernel_name: Prefix for generated kernel names.
        supply_reg: If true, create Gemini logical kernels that allocate their
            own register. If false, return Squin kernels that accept ``reg``.

    Returns:
        Mapping from generated kernel names to generated Kirin kernels.
    """

    # TODO: remove the current customization of allowing the user to supply a custom return_kernel.
    # TODO: see if we can get rid of changing __name__ and __qualname__; see standard library functions in Kirin?
    def make_kernel(
        tomog_kernel: ir.Method[..., None],
        generated_name: str,
    ) -> ir.Method[..., _ReturnT]:
        def inner_tomog_kernel(reg):
            logical_kernel(reg)
            tomog_kernel(reg)

        inner_tomog_kernel.__name__ = generated_name
        inner_tomog_kernel.__qualname__ = generated_name
        inner_kernel = squin.kernel(inner_tomog_kernel)

        if not supply_reg:
            return cast(ir.Method[..., _ReturnT], inner_kernel)

        def alloc_kernel():
            reg = qubit.qalloc(num_qubits)
            inner_kernel(reg)
            return return_val_fn(reg)

        alloc_kernel.__name__ = generated_name
        alloc_kernel.__qualname__ = generated_name
        from bloqade.gemini import logical as gemini_logical

        return cast(
            ir.Method[..., _ReturnT],
            gemini_logical.kernel(aggressive_unroll=True)(alloc_kernel),
        )

    return {
        tomog_kernel_key: make_kernel(
            tomog_kernel,
            f"{kernel_name}_{tomog_kernel_key.lower()}",
        )
        for tomog_kernel_key, tomog_kernel in tomography_kernels.items()
    }


def _default_post_processing() -> ir.Method[..., _LogicalTomographyReturn]:
    from bloqade.gemini.logical.stdlib import default_post_processing

    return cast(ir.Method[..., _LogicalTomographyReturn], default_post_processing)


# NOTE: this is basically what the user would "instantiate" for the specific
# MSD experiment in the "Experimental Demonstration of MSD" paper
def build_msd_kernels(
    theta: float,
    phi: float,
    lam: float,
    log: bool = True,
) -> PostselectionExperimentKernels:
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

    return PostselectionExperimentKernels(
        state_injection_circuit=msd_magic_prep,
        logical_circuit=msd_forward,
    )


def build_bell_state_kernels(log: bool = True) -> PostselectionExperimentKernels:
    """Build the primitives for a 10-qubit GHZ-state preparation.

    Args:
        log: If true, print a progress message.

    Returns:
        Primitive Squin kernels for the plus-state preparation and the CX
        fan-out. Needs ``num_logical_qubits=10``.
    """

    if log:
        print("Building GHZ State primitives...")

    @squin.kernel
    def plus_state_prep(reg):
        squin.broadcast.sqrt_y(reg)

    @squin.kernel
    def bell_state_body(reg):
        squin.cx(reg[0], reg[1])
        squin.broadcast.cx(reg[:2], reg[2:4])
        squin.broadcast.cx(reg[:4], reg[4:8])
        squin.broadcast.cx(reg[6:8], reg[8:])

    return PostselectionExperimentKernels(
        state_injection_circuit=plus_state_prep, logical_circuit=bell_state_body
    )


def build_logical_toffoli(
    thetas: list[float], phis: list[float], lams: list[float], log: bool = True
) -> PostselectionExperimentKernels:
    """Build the primitives for a logical Toffoli on three input states.

    The inputs are prepared on ``reg[0]``, ``reg[2]`` and ``reg[5]`` with U3
    angles ``(thetas[i], phis[i], lams[i])``; the other seven qubits are
    T-state ancillae.

    Args:
        thetas: U3 ``theta`` angles of the three input states.
        phis: U3 ``phi`` angles of the three input states.
        lams: U3 ``lambda`` angles of the three input states.
        log: If true, print a progress message.

    Returns:
        Primitive Squin kernels for the input-state preparation and the Toffoli
        circuit. Needs ``num_logical_qubits=10``.
    """

    if log:
        print("Building Toffoli primitives...")

    @squin.kernel
    def input_states_prep(reg):
        squin.u3(thetas[0], phis[0], lams[0], reg[0])
        squin.u3(thetas[1], phis[1], lams[1], reg[2])
        squin.u3(thetas[2], phis[2], lams[2], reg[5])
        squin.broadcast.t(
            ilist.IList([reg[1], reg[3], reg[4], reg[6], reg[7], reg[8], reg[9]])
        )

    @squin.kernel
    def toffoli_body(reg):
        squin.h(reg[5])
        squin.cx(reg[2], reg[5])
        squin.s_adj(reg[5])
        squin.cx(reg[5], reg[6])
        squin.cx(reg[0], reg[5])
        squin.cx(reg[5], reg[7])
        squin.cx(reg[2], reg[5])
        squin.s_adj(reg[5])
        squin.cx(reg[5], reg[8])
        squin.cx(reg[0], reg[5])
        squin.s_adj(reg[2])
        squin.broadcast.cx(ilist.IList([reg[2], reg[5]]), ilist.IList([reg[3], reg[9]]))
        squin.cx(reg[0], reg[2])
        squin.h(reg[5])
        squin.s_adj(reg[2])
        squin.cx(reg[2], reg[4])
        squin.cx(reg[0], reg[2])
        squin.cx(reg[0], reg[1])
        squin.s(reg[2])

    return PostselectionExperimentKernels(
        state_injection_circuit=input_states_prep, logical_circuit=toffoli_body
    )


def build_concatenated_msd_kernels(
    theta: float,
    phi: float,
    lam: float,
    log: bool = True,
) -> PostselectionExperimentKernels:
    """Build the primitives for two-level (concatenated) magic-state distillation.

    Five independent level-1 distillation blocks feed a single level-2 block,
    so the 25 injected magic states reduce to one output. Level-1 block ``b``
    occupies ``reg[5b : 5b + 5]`` and leaves its distilled state on ``reg[5b]``;
    the level-2 round then runs on those five outputs
    ``reg[0], reg[5], reg[10], reg[15], reg[20]`` and leaves the final state on
    ``reg[0]`` -- the tomography output qubit.

    Each round is the same 5-qubit Clifford circuit
    :func:`build_msd_kernels` uses, so error suppression composes: a round
    taking ``p -> O(p^3)`` gives ``O(p^9)`` overall.

    .. warning::

        **This does not currently run on the Gemini logical target.** It needs 25
        logical qubits and ``SteaneLogicalValidation.max_qubits`` is 10, so
        :func:`compose_tomography_kernels` rejects it with "kernel allocates 25
        qubits, exceeding the maximum of 10". That cap is a Gemini capacity
        policy rather than an intrinsic Steane limit (see the comment on the
        field), and it is a ``ClassVar``, so a caller who wants to run this can
        bind a subclass with a larger ``max_qubits``. No 5-to-1 two-level scheme
        fits in 10 qubits, so this is a capacity gap, not something a different
        circuit layout can work around. The kernels themselves are valid Squin
        and compile fine.

    .. warning::

        True concatenated distillation measures and post-selects each level-1
        block's four check qubits *before* level 2 consumes its output. A
        :class:`PostselectionExperimentKernels` carries unitary circuits only, and
        :func:`compose_tomography_kernels` measures once at the end, so the
        intermediate rejection is not applied here. The emitted circuit is the
        unitary skeleton; a caller wanting the real protocol must post-select on
        the level-1 check qubits (indices ``5b + 1 .. 5b + 4``) itself.

    Args:
        theta: U3 ``theta`` angle for input magic-state preparation.
        phi: U3 ``phi`` angle for input magic-state preparation.
        lam: U3 ``lambda`` angle for input magic-state preparation.
        log: If true, print a progress message.

    Returns:
        Primitive Squin kernels for the concatenated-MSD state injection and
        logical circuit. Needs ``num_logical_qubits=25``.
    """

    if log:
        print("Building concatenated MSD primitives...")

    @squin.kernel
    def concatenated_magic_prep(reg):
        squin.broadcast.u3(theta, phi, lam, reg)

    @squin.kernel
    def msd_round(block):
        """One 5-to-1 distillation round; distilled state lands on ``block[0]``."""
        squin.broadcast.sqrt_x(ilist.IList([block[0], block[1], block[4]]))
        squin.broadcast.cz(
            ilist.IList([block[0], block[2]]), ilist.IList([block[1], block[3]])
        )
        squin.broadcast.sqrt_y(ilist.IList([block[0], block[3]]))
        squin.broadcast.cz(
            ilist.IList([block[0], block[3]]), ilist.IList([block[2], block[4]])
        )
        squin.sqrt_x_adj(block[0])
        squin.broadcast.cz(
            ilist.IList([block[0], block[1]]), ilist.IList([block[4], block[3]])
        )
        squin.broadcast.sqrt_x_adj(block)

    @squin.kernel
    def concatenated_msd_forward(reg):
        # Level 1: five disjoint blocks, each distilling onto its first qubit.
        msd_round(ilist.IList([reg[0], reg[1], reg[2], reg[3], reg[4]]))
        msd_round(ilist.IList([reg[5], reg[6], reg[7], reg[8], reg[9]]))
        msd_round(ilist.IList([reg[10], reg[11], reg[12], reg[13], reg[14]]))
        msd_round(ilist.IList([reg[15], reg[16], reg[17], reg[18], reg[19]]))
        msd_round(ilist.IList([reg[20], reg[21], reg[22], reg[23], reg[24]]))
        # Level 2: distil the five level-1 outputs onto reg[0].
        msd_round(ilist.IList([reg[0], reg[5], reg[10], reg[15], reg[20]]))

    return PostselectionExperimentKernels(
        state_injection_circuit=concatenated_magic_prep,
        logical_circuit=concatenated_msd_forward,
    )


def compose_tomography_kernels(
    primitive_set: PostselectionExperimentKernels,
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
        else build_tomography_kernels(output_qubit=0)
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


def build_msd_steane(
    *,
    theta_offset: float = 0.0,
    phi_offset: float = 0.0,
    lam_offset: float = 0.0,
) -> tuple[ir.Method[..., None], ir.Method[..., None]]:
    """
    Returns the nonclifford prefix circuit and clifford logical circuit
    used for magic state distillation. Note that the kernels take in the qubit register 'reg' as input.
    This means that the kernels are technically agnostic to the number of input qubits.

    Args:
        theta_offset (float): An optional offset to theta to manually introduce rotation error in the nonclifford prefix. Defaults to 0.0.
        phi_offset (float): An optional offset to phi to manually introduce rotation error in the nonclifford prefix. Defaults to 0.0.
        lam_offset (float): An optional offset to lam to manually introduce rotation error in the nonclifford prefix. Defaults to 0.0.

    Returns:
        tuple[ir.Method[..., None], ir.Method[..., None]]: The (nonclifford_prefix, logical_circuit) pair.
    """
    ideal_theta = math.acos(1 / math.sqrt(3))
    ideal_phi = 0.25 * math.pi
    ideal_lam = 0.0

    theta = ideal_theta + theta_offset
    phi = ideal_phi + phi_offset
    lam = ideal_lam + lam_offset

    primitive_set = build_msd_kernels(theta, phi, lam)

    return primitive_set.state_injection_circuit, primitive_set.logical_circuit


def build_singleq_tomo() -> dict[str, ir.Method[..., None]]:
    """
    Returns X, Y, and Z basis kernels used to perform single-qubit tomography. Note that the kernels take in the register as the input.

    Returns:
        dict[str, ir.Method[..., None]]: A dictionary mapping "X", "Y", and "Z" to the respective tomography kernels.
    """
    # return a list?
    # should return (X, Y, Z) in order, but can check this.
    return build_tomography_kernels(output_qubit=0)


def build_empty_logical_circuit() -> ir.Method[..., None]:
    """Return a no-op logical Squin kernel for injected-state tomography."""

    @squin.kernel
    def empty_logical(reg):
        return

    return empty_logical


__all__ = [
    "PostselectionExperimentKernels",
    "build_bell_state_kernels",
    "build_concatenated_msd_kernels",
    "build_empty_logical_circuit",
    "build_logical_toffoli",
    "build_msd_kernels",
    "build_msd_steane",
    "build_singleq_tomo",
    "build_tomography_kernels",
    "compose_tomography_kernels",
]
