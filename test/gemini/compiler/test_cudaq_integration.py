from __future__ import annotations

import pytest
from kirin.dialects import func
from bloqade.lanes.dialects.arch import new_at
from bloqade.lanes.dialects.steane713.stmts import TerminalLogicalMeasurement
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.compile.task import (
    compile_task,
    _find_qubit_ssas,
    _find_return_stmt,
    append_measurements_and_annotations,
)
from bloqade.gemini.steane_defaults import steane7_m2obs, steane7_m2dets

# Every kernel annotated below allocates a single logical qubit, and
# ``append_measurements_and_annotations`` inserts Steane [[7,1,3]] annotations,
# so the matrices must have 7 rows. Build them from the shipped defaults rather
# than by hand so the shape cannot drift from what the function expects.
DETS = steane7_m2dets(1)  # 7 measurements, 3 detectors
OBS = steane7_m2obs(1)  # 7 measurements, 1 observable


def _make_kernel(num_qubits: int, *, with_measure: bool = False):
    # These fixtures deliberately omit terminal measurement so
    # append_measurements_and_annotations can add it below.
    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = qubit.qalloc(num_qubits)
        squin.h(reg[0])
        if with_measure:
            gemini_logical.terminal_measure(reg)

    return kernel


def _count_stmts(mt, stmt_type) -> int:
    return sum(1 for s in mt.callable_region.walk() if isinstance(s, stmt_type))


@pytest.mark.parametrize("n", [1, 3])
def test_find_qubit_ssas(n: int):
    assert len(_find_qubit_ssas(_make_kernel(n))) == n


def test_find_qubit_ssas_no_qubits():
    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        return 42

    assert len(_find_qubit_ssas(kernel)) == 0


def test_find_qubit_ssas_includes_new_at_allocations():
    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        new_at(0, 0, 0)
        qubit.qalloc(2)

    assert len(_find_qubit_ssas(kernel)) == 3


def test_find_return_stmt():
    mt = _make_kernel(1)
    ret = _find_return_stmt(mt)
    assert isinstance(ret, func.Return)
    assert ret is mt.callable_region.blocks[0].last_stmt


def test_raises_when_both_matrices_none():
    with pytest.raises(ValueError, match="At least one"):
        append_measurements_and_annotations(_make_kernel(1), None, None)


def test_raises_when_no_qubits():
    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        return 42

    with pytest.raises(ValueError, match="No qubit allocations"):
        append_measurements_and_annotations(kernel, DETS, OBS)


@pytest.mark.parametrize(
    "dets, obs, offender",
    [
        ([[1, 0], [0, 1]], OBS, "m2dets"),
        (DETS, [[1], [1]], "m2obs"),
        (steane7_m2dets(2), steane7_m2obs(2), "m2dets"),
    ],
    ids=["short_dets", "short_obs", "wrong_qubit_count"],
)
def test_rejects_matrix_whose_rows_disagree_with_the_steane_width(dets, obs, offender):
    """A row count other than ``num_qubits * 7`` used to be accepted, and the
    stride was silently re-derived from it -- annotations then referenced
    records that the terminal measurement never produced."""
    with pytest.raises(ValueError, match=f"{offender} has .* rows, expected 7"):
        append_measurements_and_annotations(_make_kernel(1), dets, obs)


@pytest.mark.parametrize(
    "dets, obs, offender",
    [
        ([row[:-1] if i else row for i, row in enumerate(DETS)], OBS, "m2dets"),
        (DETS, [row + [1] if i else row for i, row in enumerate(OBS)], "m2obs"),
    ],
    ids=["ragged_dets", "ragged_obs"],
)
def test_rejects_ragged_matrix(dets, obs, offender):
    """Columns are read as ``row[j]`` over the first row's range, so a short
    row raised IndexError from inside the annotation loop and a long row had
    its extra columns silently dropped."""
    with pytest.raises(ValueError, match=f"{offender} is ragged"):
        append_measurements_and_annotations(_make_kernel(1), dets, obs)


def test_rejects_terminal_measurement_with_non_steane_width():
    """Annotations are indexed at a stride of 7 regardless, so a statement
    declaring another width would have its records read at the wrong stride."""
    mt = _make_kernel(1, with_measure=True)
    terminal = next(
        s
        for s in mt.callable_region.walk()
        if isinstance(s, TerminalLogicalMeasurement)
    )
    terminal.num_physical_qubits = 3

    with pytest.raises(ValueError, match="num_physical_qubits=3"):
        append_measurements_and_annotations(mt, DETS, OBS)


@pytest.mark.parametrize(
    "with_measure", [False, True], ids=["no_terminal", "has_terminal"]
)
def test_terminal_measurement_count(with_measure: bool):
    mt = _make_kernel(1, with_measure=with_measure)
    append_measurements_and_annotations(mt, DETS, OBS)
    assert _count_stmts(mt, TerminalLogicalMeasurement) == 1


@pytest.mark.parametrize(
    "dets, obs, expected_dets, expected_obs",
    [
        (DETS, OBS, 3, 1),
        (DETS, None, 3, 0),
        (None, OBS, 0, 1),
    ],
    ids=["both", "dets_only", "obs_only"],
)
def test_inserted_annotation_counts(dets, obs, expected_dets, expected_obs):
    mt = _make_kernel(1)
    append_measurements_and_annotations(mt, dets, obs)
    assert _count_stmts(mt, SetDetector) == expected_dets
    assert _count_stmts(mt, SetObservable) == expected_obs


def test_rewrites_return_when_no_existing_terminal():
    mt = _make_kernel(1)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)
    append_measurements_and_annotations(mt, DETS, OBS)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)


def test_rewrites_return_when_terminal_exists():
    mt = _make_kernel(1, with_measure=True)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)
    append_measurements_and_annotations(mt, DETS, OBS)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)


def test_compile_task_preserves_source_kernel_across_repeated_compilation():
    source = _make_kernel(1)
    source_ir = source.print_str()

    first, _, _, _ = compile_task(source, DETS, OBS)
    second, _, _, _ = compile_task(source, DETS, OBS)

    assert first is not source
    assert second is not source
    assert first is not second
    assert source.print_str() == source_ir
    assert _count_stmts(source, TerminalLogicalMeasurement) == 0
    assert _count_stmts(source, SetDetector) == 0
    assert _count_stmts(source, SetObservable) == 0
    assert _count_stmts(first, TerminalLogicalMeasurement) == 1
    assert _count_stmts(second, TerminalLogicalMeasurement) == 1
    assert _count_stmts(first, SetDetector) == len(DETS[0])
    assert _count_stmts(second, SetDetector) == len(DETS[0])
    assert _count_stmts(first, SetObservable) == len(OBS[0])
    assert _count_stmts(second, SetObservable) == len(OBS[0])
