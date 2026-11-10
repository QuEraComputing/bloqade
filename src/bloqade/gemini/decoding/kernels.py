from __future__ import annotations

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
class _DecoderPrimitiveSet:
    """Primitive Squin kernels needed to build MSD decoder tasks.

    Attributes:
        state_injection_circuit: Squin kernel that prepares the input logical
            magic states.
        logical_circuit: Squin kernel for the logical circuit under test.
    """

    state_injection_circuit: ir.Method[..., None]
    logical_circuit: ir.Method[..., None]


def _build_tomography_primitives(
    *, output_qubit: int
) -> dict[str, ir.Method[..., None]]:
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
