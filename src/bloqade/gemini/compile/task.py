from typing import Any
from collections.abc import Callable

import numpy as np
from kirin import ir
from kirin.validation import ValidationSuite
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.validation.steane import SteaneLogicalValidation
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
)

from bloqade.gemini.cudaq import cudaq_to_squin, is_cudaq_kernel
from bloqade.gemini.post_processing import build_post_processing
from bloqade.gemini.steane_defaults import (
    STEANE7_PHYSICAL_QUBITS,
    steane7_m2obs,
    steane7_m2dets,
)
from bloqade.gemini.utils.annotations import (
    _insert_before,
    _find_qubit_ssas,
    _find_return_stmt,
    _append_measurements_and_annotations,
)
from bloqade.gemini.logical.validation.arguments import GeminiLogicalArgumentValidation

__all__ = [
    "_find_qubit_ssas",
    "_find_return_stmt",
    "_insert_before",
    "append_measurements_and_annotations",
    "compile_task",
    "run_squin_kernel_validation",
]


def run_squin_kernel_validation(mt: ir.Method):
    """
    Run validation checks on a Squin kernel method.

    Args:
        mt (ir.Method): The Squin kernel method to validate.

    Returns:
        ValidationResult: A validation result object containing the
            validation errors, if they exist

    Note: To trigger an error run `run_squin_kernel_validation(mt).raise_if_invalid()`.

    """
    validator = ValidationSuite(
        [
            SteaneLogicalValidation,
            SteaneTerminalMeasurementValidation,
            FlatKernelNoCloningValidation,
            # NOTE: not redundant with the `@logical.kernel` decorator. A method
            # reaching `compile_task` need never have been decorated -- the
            # CUDA-Q path builds one by conversion -- so this is the only place
            # the no-arguments contract is checked on that route.
            GeminiLogicalArgumentValidation,
        ]
    )
    return validator.validate(mt)


def append_measurements_and_annotations(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> None:
    """Append terminal measurement, detector, and observable IR statements to a squin kernel.

    This legacy compiler entry point mutates the method in place. For
    submission kernels use ``bloqade.gemini.utils.append_measurements_and_annotations``.

    The annotations are Steane [[7,1,3]], so both matrices must be rectangular
    with one row per physical qubit: ``num_qubits * 7``. A shape that disagrees
    is a ``ValueError`` rather than a silently mis-indexed annotation.

    Args:
        mt: A squin ``ir.Method`` whose body returns ``None``.
        m2dets: Binary matrix of shape ``(num_qubits * 7, num_detectors)``.
            Each column defines a detector by its non-zero row indices.
        m2obs: Binary matrix of shape ``(num_qubits * 7, num_observables)``.
            Each column defines an observable by its non-zero row indices.

    Raises:
        ValueError: If neither matrix is given; if ``mt`` allocates no qubits;
            if either matrix has the wrong number of rows or is ragged; or if
            ``mt``'s terminal measurement already declares a width other than
            Steane [[7,1,3]]'s seven.
    """

    if m2dets is None and m2obs is None:
        raise ValueError("At least one of m2dets or m2obs must be provided")

    qubit_ssas = _find_qubit_ssas(mt)
    num_qubits = len(qubit_ssas)
    if num_qubits == 0:
        raise ValueError("No qubit allocations found in the kernel")

    expected_rows = num_qubits * STEANE7_PHYSICAL_QUBITS
    for name, matrix in (("m2dets", m2dets), ("m2obs", m2obs)):
        if matrix is None:
            continue
        if len(matrix) != expected_rows:
            raise ValueError(
                f"{name} has {len(matrix)} rows, expected {expected_rows}: "
                f"{num_qubits} logical qubit(s) x {STEANE7_PHYSICAL_QUBITS} "
                "physical qubits per Steane [[7,1,3]] qubit"
            )
        # Columns are read as ``row[j]`` for every j in the first row's range,
        # so a ragged matrix either raises IndexError from inside the
        # annotation loop or silently ignores a longer row's extra columns.
        column_counts = {len(row) for row in matrix}
        if len(column_counts) > 1:
            raise ValueError(
                f"{name} is ragged: rows have {sorted(column_counts)} columns. "
                "Every row must define the same number of annotations"
            )

    _append_measurements_and_annotations(
        mt,
        qubit_ssas,
        _find_return_stmt(mt),
        None if m2dets is None else np.asarray(m2dets),
        None if m2obs is None else np.asarray(m2obs),
        logical=True,
        validate_terminal=False,
    )


def compile_task(
    logical_kernel: ir.Method | Callable[..., Any],
    m2dets: list[list[int]] | None = None,
    m2obs: list[list[int]] | None = None,
):
    """Compile a logical kernel into physical move artifacts.

    Handles CUDAQ kernel detection/conversion, squin kernel validation,
    squin-to-move compilation, architecture spec generation, and
    post-processing extraction.

    Args:
        logical_kernel: A squin ``ir.Method`` or a CUDA-Q kernel to compile.
        m2dets: Binary measurement-to-detector matrix. For CUDA-Q kernels,
            defaults to Steane [[7,1,3]] detectors if ``None``.
        m2obs: Binary measurement-to-observable matrix. For CUDA-Q kernels,
            defaults to Steane [[7,1,3]] observables if ``None``.

    Returns:
        A tuple of ``(logical_squin_kernel, physical_arch_spec,
        physical_move_kernel, post_processing)``.

    """
    if is_cudaq_kernel(logical_kernel):
        logical_squin_kernel: ir.Method = cudaq_to_squin(logical_kernel)

        if m2dets is None and m2obs is None:
            num_qubits = len(_find_qubit_ssas(logical_squin_kernel))
            m2dets = steane7_m2dets(num_qubits)
            m2obs = steane7_m2obs(num_qubits)

        append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
    elif isinstance(logical_kernel, ir.Method):
        # Compilation and annotation rewrites are in-place. Work on an owned
        # copy so creating a task never changes a caller-owned kernel.
        logical_squin_kernel = logical_kernel.similar()
        if m2dets is not None or m2obs is not None:
            append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
    else:
        raise ValueError(f"Unknown kernel type {type(logical_kernel)}")

    run_squin_kernel_validation(logical_squin_kernel).raise_if_invalid()

    physical_arch_spec = physical.get_arch_spec()
    physical_move_kernel = LogicalPipeline(transversal_rewrite=True).emit(
        logical_squin_kernel
    )
    post_processing = build_post_processing(logical_squin_kernel)

    return (
        logical_squin_kernel,
        physical_arch_spec,
        physical_move_kernel,
        post_processing,
    )
