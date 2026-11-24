"""Measurement and parity annotations for Gemini submission kernels."""

from typing import TypeVar
from functools import cache

import numpy as np
from kirin import ir, types
from kirin.passes import TypeInfer
from numpy.typing import ArrayLike
from kirin.dialects import py, func, ilist
from kirin.validation import ValidationSuite
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.analysis.address import AddressQubit, AddressAnalysis
from bloqade.lanes.dialects.steane713.stmts import TerminalLogicalMeasurement
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.lanes.validation.terminal_measure import (
    PhysicalTerminalMeasurementValidation,
)
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
)

from bloqade import qubit
from bloqade.gemini.steane_defaults import STEANE7_PHYSICAL_QUBITS

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
    exits = [
        block.last_stmt
        for block in mt.callable_region.blocks
        if isinstance(block.last_stmt, func.Return)
    ]
    if len(exits) != 1:
        raise ValueError("Expected a single return in the unrolled Gemini kernel")
    last = exits[0]
    assert isinstance(last, func.Return)
    return last


def _insert_before(stmt: _S, anchor: ir.Statement) -> _S:
    """Insert stmt before anchor and return stmt for chaining."""
    stmt.insert_before(anchor)
    return stmt


def _parity_matrix(
    matrix: ArrayLike | None, name: str, num_measurements: int
) -> np.ndarray | None:
    if matrix is None:
        return None
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != num_measurements:
        raise ValueError(
            f"{name} must have {num_measurements} rows (one per measurement); got shape {values.shape}"
        )
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} must contain only zeros and ones")
    return values


def _emit_annotations(
    measure: TerminalLogicalMeasurement | qubit.stmts.Measure,
    ret: func.Return,
    dets: np.ndarray | None,
    obs: np.ndarray | None,
) -> tuple[list[ir.SSAValue], list[ir.SSAValue]]:
    """Emit one parity per matrix column in terminal measurement order."""

    @cache
    def index(value: ir.SSAValue, i: int) -> ir.SSAValue:
        pos = _insert_before(py.Constant(i), ret)
        return _insert_before(py.GetItem(value, pos.result), ret).result

    def bit(i: int) -> ir.SSAValue:
        if isinstance(measure, TerminalLogicalMeasurement):
            q, offset = divmod(i, STEANE7_PHYSICAL_QUBITS)
            return index(index(measure.result, q), offset)
        return index(measure.result, i)

    detectors: list[ir.SSAValue] = []
    observables: list[ir.SSAValue] = []
    for matrix, is_detector in ((dets, True), (obs, False)):
        if matrix is None:
            continue
        for column, parity in enumerate(matrix.T):
            bits = [bit(int(i)) for i in np.flatnonzero(parity)]
            selected = _insert_before(ilist.New(bits), ret)
            if is_detector:
                coords = [
                    _insert_before(py.Constant(float(x)), ret).result
                    for x in (0, column)
                ]
                coordinate_list = _insert_before(ilist.New(coords), ret)
                annotation = _insert_before(
                    SetDetector(selected.result, coordinate_list.result), ret
                )
                detectors.append(annotation.result)
            else:
                observable = _insert_before(SetObservable(selected.result), ret)
                observables.append(observable.result)

    return detectors, observables


def _append_measurements_and_annotations(
    mt: ir.Method,
    qubits: list[ir.SSAValue],
    ret: func.Return,
    dets: np.ndarray | None,
    obs: np.ndarray | None,
    *,
    logical: bool,
    validate_terminal: bool,
) -> tuple[list[ir.SSAValue], list[ir.SSAValue]]:
    """Reuse or insert the terminal measurement and emit the requested parities in place."""
    measure_type = TerminalLogicalMeasurement if logical else qubit.stmts.Measure
    existing = [
        s
        for s in mt.callable_region.walk()
        if isinstance(s, (TerminalLogicalMeasurement, qubit.stmts.Measure))
    ]
    if existing:
        if len(existing) != 1 or not isinstance(existing[0], measure_type):
            raise ValueError(
                "Expected a single terminal measurement for this kernel dialect"
            )
        measure = existing[0]
    else:
        register = _insert_before(ilist.New(qubits), ret)
        measure = _insert_before(measure_type(register.result), ret)
    if isinstance(measure, TerminalLogicalMeasurement):
        if measure.num_physical_qubits not in (None, STEANE7_PHYSICAL_QUBITS):
            raise ValueError(
                f"Expected a Steane terminal measurement, got num_physical_qubits={measure.num_physical_qubits}"
            )
        measure.num_physical_qubits = STEANE7_PHYSICAL_QUBITS

    if validate_terminal:
        validation = (
            SteaneTerminalMeasurementValidation
            if logical
            else PhysicalTerminalMeasurementValidation
        )
        ValidationSuite([validation]).validate(mt).raise_if_invalid()
    return _emit_annotations(measure, ret, dets, obs)


def _annotated_kernel(
    kernel: ir.Method,
    m2dets: ArrayLike | None,
    m2obs: ArrayLike | None,
    *,
    logical: bool,
) -> ir.Method:
    if m2dets is None and m2obs is None:
        raise ValueError("Supply at least one of m2dets or m2obs")

    mt = kernel.similar()
    AggressiveUnroll(mt.dialects, no_raise=False).fixpoint(mt)
    ret = _find_return_stmt(mt)
    if not ret.value.type.is_subseteq(types.NoneType) or ret.value.type == types.Bottom:
        raise ValueError("The input kernel must return None before adding annotations")

    qubits = _find_qubit_ssas(mt)
    if not qubits:
        raise ValueError("No qubit allocations found in the kernel")
    num_measurements = len(qubits) * STEANE7_PHYSICAL_QUBITS if logical else len(qubits)
    dets = _parity_matrix(m2dets, "m2dets", num_measurements)
    obs = _parity_matrix(m2obs, "m2obs", num_measurements)
    detectors, observables = _append_measurements_and_annotations(
        mt,
        qubits,
        ret,
        dets,
        obs,
        logical=logical,
        validate_terminal=True,
    )
    detector_list = _insert_before(ilist.New(detectors), ret)
    observable_list = _insert_before(ilist.New(observables), ret)
    output = _insert_before(
        py.tuple.New((detector_list.result, observable_list.result)), ret
    )
    ret.replace_by(func.Return(output.result))
    TypeInfer(mt.dialects, no_raise=False)(mt)
    mt.verify()
    return mt


def append_measurements_and_annotations(
    kernel: ir.Method, m2dets: ArrayLike | None, m2obs: ArrayLike | None
) -> ir.Method:
    """Return an annotated copy of a None-returning logical kernel.

    Each binary matrix has one row per physical measurement (seven consecutive
    rows per measured Steane block). Each column defines one parity and may
    span blocks. The kernel returns (detectors, observables) in column order.
    An existing terminal measurement determines row order. A new measurement
    uses allocation order.
    """
    return _annotated_kernel(kernel, m2dets, m2obs, logical=True)


def append_measurements_and_annotations_physical(
    kernel: ir.Method, m2dets: ArrayLike | None, m2obs: ArrayLike | None
) -> ir.Method:
    """Return an annotated copy of a None-returning physical kernel.

    Each binary matrix has one row per entry in the flat terminal measurement.
    Each column selects the measurements for one parity across the full kernel.
    The kernel returns (detectors, observables) in column order. When inserting
    a measurement, measure all qubits in allocation order.
    """
    return _annotated_kernel(kernel, m2dets, m2obs, logical=False)
