"""Read the gates of ``move.PhysicalInitialize`` off its initialization kernel.

``move.PhysicalInitialize`` is a placeholder. ``move2squin`` replaces it with a
call to an initialization kernel (for the Steane [[7,1,3]] code,
``steane7_initialize_broadcast``), so the move program itself says nothing
about the gates and atom moves that prepare the state.

:func:`initialize_operations` recovers them. It applies the kernel to the
blocks of a ``PhysicalInitialize``, lowers the result to the neutral-atom gate
set the way the compiler lowers any circuit, and returns the flat list of
``R``, ``Rz`` and ``CZ`` operations. The move debuggers turn that list into
steps.
"""

from __future__ import annotations

import math
from dataclasses import replace, dataclass
from collections.abc import Sequence

from kirin import ir
from kirin.dialects import scf, func, ilist
from bloqade.analysis import address
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.native.dialects import gate as native_gate
from bloqade.native.upstream.squin2native import SquinToNative

from bloqade import squin

# A constant angle goes turns -> radians -> turns on its way through the
# kernel. Rounding drops the float noise that round trip leaves behind.
_ANGLE_DIGITS = 12


class InitializeKernelError(ValueError):
    """An initialization kernel cannot be shown step by step."""


@dataclass(frozen=True)
class InitializeR:
    """Rotation about an equatorial axis. Angles are in turns."""

    axis_angle: float
    rotation_angle: float
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class InitializeRz:
    """Rotation about Z. The angle is in turns."""

    rotation_angle: float
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class InitializeCZ:
    """CZ between ``controls[i]`` and ``targets[i]``.

    The kernel only pairs the qubits. By convention the debuggers treat the
    controls as the atoms that move: each travels to the CZ partner site of its
    target and back.
    """

    controls: tuple[int, ...]
    targets: tuple[int, ...]


InitializeOperation = InitializeR | InitializeRz | InitializeCZ


def initialize_operations(
    kernel: ir.Method,
    thetas: Sequence[float],
    phis: Sequence[float],
    lams: Sequence[float],
    block_size: int,
) -> list[InitializeOperation]:
    """List the native operations ``kernel`` applies to initialize some blocks.

    Qubits are numbered block by block: qubit ``block * block_size + site`` is
    site ``site`` of block ``block``, matching the nesting of
    ``move.PhysicalInitialize.location_addresses``.

    Rotations by zero turns are dropped, and equal rotations on different
    qubits are merged into one operation when nothing between them acts on
    those qubits, so blocks prepared alike are shown together.

    Args:
        kernel: Initialization kernel, called as
            ``kernel(thetas, phis, lams, blocks)`` with angles in radians.
        thetas: Per-block ``theta`` of ``move.PhysicalInitialize``, in turns.
        phis: Per-block ``phi``, in turns.
        lams: Per-block ``lam``, in turns.
        block_size: Number of physical qubits in a block.

    Raises:
        InitializeKernelError: If ``kernel`` does not reduce to a fixed
            sequence of native gates on known qubits.
    """
    num_blocks = len(thetas)
    theta_radians = ilist.IList([math.tau * theta for theta in thetas])
    phi_radians = ilist.IList([math.tau * phi for phi in phis])
    lam_radians = ilist.IList([math.tau * lam for lam in lams])

    @squin.kernel
    def apply_kernel():
        qubits = squin.qalloc(num_blocks * block_size)

        def block(index: int):
            return qubits[index * block_size : (index + 1) * block_size]

        kernel(
            theta_radians,
            phi_radians,
            lam_radians,
            ilist.map(block, ilist.range(num_blocks)),
        )

    native = SquinToNative().emit(apply_kernel)
    AggressiveUnroll(native.dialects).fixpoint(native)
    frame, _ = address.AddressAnalysis(native.dialects).run(native)

    def qubits_of(value: ir.SSAValue) -> tuple[int, ...]:
        register = frame.get(value)
        if not isinstance(register, address.AddressReg):
            raise InitializeKernelError(
                f"the kernel {kernel.sym_name!r} applies a gate to qubits that "
                "cannot be resolved"
            )
        return tuple(register.data)

    def angle_of(value: ir.SSAValue) -> float:
        constant = value.hints.get("const")
        angle = getattr(constant, "data", None)
        if not isinstance(angle, (float, int)):
            raise InitializeKernelError(
                f"the kernel {kernel.sym_name!r} applies a gate whose angle is "
                "not a constant"
            )
        return round(float(angle), _ANGLE_DIGITS)

    operations: list[InitializeOperation] = []
    for stmt in native.callable_region.walk():
        if isinstance(stmt, native_gate.stmts.R):
            operations.append(
                InitializeR(
                    angle_of(stmt.axis_angle),
                    angle_of(stmt.rotation_angle),
                    qubits_of(stmt.qubits),
                )
            )
        elif isinstance(stmt, native_gate.stmts.Rz):
            operations.append(
                InitializeRz(angle_of(stmt.rotation_angle), qubits_of(stmt.qubits))
            )
        elif isinstance(stmt, native_gate.stmts.CZ):
            operations.append(
                InitializeCZ(qubits_of(stmt.controls), qubits_of(stmt.targets))
            )
        elif isinstance(stmt, (func.Invoke, func.Call, scf.IfElse, scf.For)):
            # Gates behind a call or a branch that survived unrolling would be
            # missed, or listed as if they always ran.
            raise InitializeKernelError(
                f"the kernel {kernel.sym_name!r} does not unroll to a fixed "
                "sequence of gates"
            )

    return _merge_rotations(operations)


def _merge_rotations(
    operations: Sequence[InitializeOperation],
) -> list[InitializeOperation]:
    """Drop identity rotations and merge equal ones on independent qubits."""
    merged: list[InitializeOperation] = []
    for operation in operations:
        if isinstance(operation, InitializeCZ):
            merged.append(operation)
        elif operation.rotation_angle == 0.0:
            continue
        elif (index := _joinable(merged, operation)) is None:
            merged.append(operation)
        else:
            earlier = merged[index]
            assert not isinstance(earlier, InitializeCZ)
            merged[index] = replace(earlier, qubits=earlier.qubits + operation.qubits)
    return merged


def _joinable(
    merged: Sequence[InitializeOperation], operation: InitializeR | InitializeRz
) -> int | None:
    """Index of an earlier, equal rotation that ``operation`` can join.

    Operations on other qubits commute with ``operation``, so it may slide back
    past them. A CZ, or anything on its own qubits, stops the slide.
    """
    for index in reversed(range(len(merged))):
        earlier = merged[index]
        if isinstance(earlier, InitializeCZ):
            return None
        if set(earlier.qubits) & set(operation.qubits):
            return None
        if replace(earlier, qubits=()) == replace(operation, qubits=()):
            return index
    return None
