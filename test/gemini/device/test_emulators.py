import numpy as np
import pytest
from kirin.dialects import ilist
from bloqade.core.device import TaskBuilder
from bloqade.lanes.dialects.steane713.stmts import TerminalLogicalMeasurement

from bloqade import qubit
from bloqade.gemini import GeminiResult, logical, physical
from bloqade.gemini.utils import (
    append_measurements_and_annotations,
    append_measurements_and_annotations_physical,
)
from bloqade.gemini.post_processing import build_post_processing


@physical.kernel(verify=False)
def physical_unmeasured():
    q = physical.qalloc(4)
    physical.x(q[0])


@physical.kernel
def physical_reordered():
    q = physical.qalloc(4)
    physical.x(q[0])
    physical.broadcast.measure(ilist.IList([q[3], q[2], q[1], q[0]]))


@logical.kernel(verify=False)
def logical_unmeasured():
    logical.qalloc(2)


@logical.kernel
def logical_measured():
    logical.terminal_measure(logical.qalloc(2))


@pytest.mark.parametrize("kernel", [logical_measured, logical_unmeasured])
def test_logical_annotation_parities(kernel):
    # The first detector and the observable each span two Steane blocks.
    dets = np.zeros((14, 2), dtype=int)
    dets[[0, 8], 0] = 1
    dets[[0, 1], 1] = 1
    obs = np.zeros((14, 1), dtype=int)
    obs[[1, 8], 0] = 1
    mt = append_measurements_and_annotations(kernel, dets, obs)
    assert (
        sum(
            isinstance(s, TerminalLogicalMeasurement) for s in mt.callable_region.walk()
        )
        == 1
    )
    shots = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    postprocess = build_post_processing(mt)
    d, o = next(iter(postprocess.emit_return(shots)))
    assert list(d) == [False, True]
    assert list(o) == [True]
    assert list(postprocess.emit_detectors(shots)) == [[False, True]]
    assert list(postprocess.emit_observables(shots)) == [[True]]


@pytest.mark.parametrize(
    "kernel,expected",
    [(physical_unmeasured, [1, 0, 0, 0]), (physical_reordered, [0, 0, 0, 1])],
)
def test_physical_submission_measurement_order(kernel, expected):
    mt = append_measurements_and_annotations_physical(
        kernel, [[1, 0], [0, 1], [1, 0], [0, 1]], [[1], [0], [0], [1]]
    )
    assert (
        sum(isinstance(s, qubit.stmts.Measure) for s in mt.callable_region.walk()) == 1
    )
    builder = TaskBuilder()
    builder.add_subtask(mt, 3)
    future = physical.GeminiPhysicalDeviceEmulator(
        with_noise=False, batch_shots=2
    ).run_async(builder)
    result = future.result()
    assert isinstance(result, GeminiResult)
    assert isinstance(result, physical.GeminiPhysicalResult)
    d = [expected[0] ^ expected[2], expected[1] ^ expected[3]]
    o = [expected[0] ^ expected[3]]
    np.testing.assert_array_equal(result.measurements[0], [expected] * 3)
    np.testing.assert_array_equal(result.detectors[0], [d] * 3)
    np.testing.assert_array_equal(result.observables[0], [o] * 3)
    for returned_dets, returned_obs in result.return_values[0]:
        np.testing.assert_array_equal(returned_dets, d)
        np.testing.assert_array_equal(returned_obs, o)
    selected = result.postselect_on_fully_filled()
    assert type(selected) is type(result)
    np.testing.assert_array_equal(selected.measurements, result.measurements)


def test_logical_submission_result():
    mt = append_measurements_and_annotations(
        logical_unmeasured, None, [[1, 0]] * 7 + [[0, 1]] * 7
    )
    builder = TaskBuilder()
    builder.add_subtask(mt, 2)
    device = logical.GeminiLogicalDeviceEmulator(with_noise=False)
    result = device.run_async(builder).result()
    assert type(result.postselect_on_fully_filled()) is logical.GeminiLogicalResult
    assert np.shape(result.observables) == (1, 2, 2)
    assert list(result.return_values[0][0][0]) == []


@pytest.mark.parametrize(
    "dialect,annotate",
    [
        (logical, append_measurements_and_annotations),
        (physical, append_measurements_and_annotations_physical),
    ],
)
def test_reject_existing_return(dialect, annotate):
    @dialect.kernel(verify=False)
    def returns_value():
        return 17

    with pytest.raises(ValueError, match="must return None"):
        annotate(returns_value, [[1] * 7], None)


@pytest.mark.parametrize(
    "dets,obs",
    [
        ([[1, 1, 1]], None),
        ([[1, 1]], [[1]]),
        ([[1], [0], [2], [0]], None),
        (None, None),
    ],
)
def test_reject_invalid_matrices(dets, obs):
    with pytest.raises(ValueError):
        append_measurements_and_annotations_physical(physical_unmeasured, dets, obs)


@pytest.mark.parametrize("matrix_name", ["m2dets", "m2obs"])
def test_logical_annotations_require_whole_program_rows(matrix_name):
    matrices = {"m2dets": None, "m2obs": None}
    matrices[matrix_name] = np.ones((7, 1), dtype=int)
    with pytest.raises(ValueError, match=f"{matrix_name} must have 14 rows"):
        append_measurements_and_annotations(logical_unmeasured, **matrices)


def test_annotation_rejects_multiple_physical_measurements():
    @physical.kernel(verify=False)
    def twice():
        reg = physical.qalloc(2)
        physical.broadcast.measure(reg)
        physical.broadcast.measure(reg)

    with pytest.raises(ValueError, match="single terminal measurement"):
        append_measurements_and_annotations_physical(twice, [[1], [1]], None)
