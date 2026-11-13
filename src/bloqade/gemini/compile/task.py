from typing import Any, TypeVar
from functools import cache
from collections.abc import Callable

from kirin import ir
from kirin.dialects import py, func, ilist
from kirin.validation import ValidationSuite
from bloqade.lanes.transform import LogicalPipeline
from bloqade.analysis.address import AddressQubit, AddressAnalysis
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.validation.steane import SteaneLogicalValidation
from bloqade.lanes.dialects.steane713.stmts import TerminalLogicalMeasurement
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
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


_S = TypeVar("_S", bound=ir.Statement)


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Return one qubit SSA value per concrete qubit address.

    ``qalloc`` calls must be aggressively unrolled so each allocation has a
    corresponding SSA value in ``mt``.
    """
    address_analysis = AddressAnalysis(mt.dialects)
    frame, _ = address_analysis.run(mt)
    qubits_by_address: dict[int, ir.SSAValue] = {}

    for stmt in mt.callable_region.walk():
        for result in stmt.results:
            address = frame.get(result)
            if isinstance(address, AddressQubit):
                qubits_by_address.setdefault(address.data, result)

    return [
        qubits_by_address[address] for address in range(address_analysis.qubit_count)
    ]


def _find_return_stmt(mt: ir.Method) -> func.Return:
    """Find the func.Return statement at the end of the function body."""
    block = mt.callable_region.blocks[0]
    last = block.last_stmt
    assert isinstance(last, func.Return), f"Expected func.Return, got {type(last)}"
    return last


def _insert_before(stmt: _S, anchor: ir.Statement) -> _S:
    """Insert stmt before anchor and return stmt for chaining."""
    stmt.insert_before(anchor)
    return stmt


def append_measurements_and_annotations(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> None:
    """Append terminal measurement, detector, and observable IR statements to a squin kernel.

    The method is mutated in-place.

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

    existing_terminal = next(
        (
            s
            for s in mt.callable_region.walk()
            if isinstance(s, TerminalLogicalMeasurement)
        ),
        None,
    )
    # Everything below addresses a measurement as ``divmod(row,
    # STEANE7_PHYSICAL_QUBITS)``, so a statement that already declares some
    # other width would have its records read at the wrong stride -- wrong
    # detectors, or an out-of-range index into a narrower logical measurement.
    # Nothing here honours a non-Steane width, so say so instead.
    if (
        existing_terminal is not None
        and existing_terminal.num_physical_qubits is not None
        and existing_terminal.num_physical_qubits != STEANE7_PHYSICAL_QUBITS
    ):
        raise ValueError(
            "This function inserts Steane [[7,1,3]] annotations, but the "
            "kernel's terminal measurement declares num_physical_qubits="
            f"{existing_terminal.num_physical_qubits}"
        )

    # The annotations below address a measurement as ``divmod(row,
    # STEANE7_PHYSICAL_QUBITS)``, and the terminal measurement is stamped with
    # that same width. Deriving the stride from ``len(m2) // num_qubits``
    # instead would make the matrices a second, unchecked source of the width:
    # a matrix disagreeing with the stamp yields detectors that reference the
    # wrong records, with no error. Validate against the one source instead.
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

    return_stmt = _find_return_stmt(mt)

    # insert TerminalLogicalMeasurement if not present
    if existing_terminal is not None:
        term_meas = existing_terminal
    else:
        qlist_stmt = _insert_before(ilist.New(qubit_ssas), return_stmt)
        term_meas = _insert_before(
            TerminalLogicalMeasurement(qlist_stmt.result), return_stmt
        )

    # ``InsertQubitCount`` stamps this attribute during ``@logical.kernel``
    # decoration, but a statement inserted here is created afterwards and so is
    # never walked -- it stays ``None``, and ``MeasurementIDAnalysis`` then
    # cannot expand a logical measurement into per-physical-qubit records
    # (every logical qubit degrades to ``AnyMeasureId``, taking the detectors
    # with it). The annotations this function inserts are Steane [[7,1,3]], so
    # the width is known here. A statement that already carries a count keeps
    # it.
    if term_meas.num_physical_qubits is None:
        term_meas.num_physical_qubits = STEANE7_PHYSICAL_QUBITS

    @cache
    def _get_logical_measurement(q_idx: int) -> ir.SSAValue:
        (idx := py.Constant(q_idx)).insert_before(return_stmt)
        (getitem := py.GetItem(term_meas.result, idx.result)).insert_before(return_stmt)
        return getitem.result

    @cache
    def _get_physical_measurement(q_idx: int, m_idx: int) -> ir.SSAValue:
        (idx := py.Constant(m_idx)).insert_before(return_stmt)
        (
            getitem := py.GetItem(_get_logical_measurement(q_idx), idx.result)
        ).insert_before(return_stmt)
        return getitem.result

    # insert detectors
    if m2dets is not None:
        for j in range(len(m2dets[0])):
            indices = [i for i, row in enumerate(m2dets) if row[j]]
            meas_ssas = [
                _get_physical_measurement(*divmod(idx, STEANE7_PHYSICAL_QUBITS))
                for idx in indices
            ]
            meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)

            coord_0 = _insert_before(py.Constant(0.0), return_stmt)
            coord_1 = _insert_before(py.Constant(float(j)), return_stmt)
            coords = _insert_before(
                ilist.New([coord_0.result, coord_1.result]), return_stmt
            )

            _insert_before(SetDetector(meas_list.result, coords.result), return_stmt)

    # insert observables
    if m2obs is not None:
        for j in range(len(m2obs[0])):
            indices = [i for i, row in enumerate(m2obs) if row[j]]
            meas_ssas = [
                _get_physical_measurement(*divmod(idx, STEANE7_PHYSICAL_QUBITS))
                for idx in indices
            ]
            meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
            _insert_before(SetObservable(meas_list.result), return_stmt)


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
