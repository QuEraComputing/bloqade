from datetime import datetime, timezone

import numpy as np
import pytest
from kirin.dialects import ilist
from bloqade.core.device import (
    ShotFilter,
    ShotResult,
    DictStorage,
)
from kirin.serialization import JSONSerializer
from bloqade.lanes.analysis import atom
from bloqade.lanes.analysis.atom.analysis import PostProcessing
from qlam_core.plugins.tasks.api.tasks_models import (
    Program,
    Subtask,
    TaskMetadata,
    TaskDefinition,
)

from bloqade import squin
from bloqade.gemini import GeminiLogicalResult, logical
from bloqade.gemini.device.logical import utils as utils_module, result as result_module
from bloqade.gemini.device.logical.utils import (
    ShotRemappingException,
    get_slm_mapping_postprocessing,
)

CREATION_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

# === real serialized kernels ===
#
# Tests that count kernel decoding and SLM post-processing compilation run
# them for real (via mocker.spy), so Program.content must be valid kernel JSON.
# Two distinct kernels let per-program assertions distinguish them.


@logical.kernel(aggressive_unroll=True)
def _kernel_a():
    q = squin.qalloc(2)
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def _kernel_b():
    q = squin.qalloc(3)
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def _kernel_with_postprocessing():
    q = squin.qalloc(2)
    measurements = logical.terminal_measure(q)
    return squin.set_detector(
        ilist.IList([measurements[0][0], measurements[1][0]]), [0, 1]
    )


_serializer = JSONSerializer()
KERNEL_A_JSON = _serializer.encode(_kernel_a.dialects.encode(_kernel_a))
KERNEL_B_JSON = _serializer.encode(_kernel_b.dialects.encode(_kernel_b))
KERNEL_WITH_POSTPROCESSING_JSON = _serializer.encode(
    _kernel_with_postprocessing.dialects.encode(_kernel_with_postprocessing)
)


def make_shot(
    *,
    task_id: str = "task-1",
    shot_index: int = 0,
    subtask_index: int = 0,
    subtask_shot_index: int = 0,
    frame_type: str = "DETECTED",
    bitstring: tuple[bool, ...] = (True, False),
):
    return ShotResult(
        task_id=task_id,
        shot_index=shot_index,
        subtask_index=subtask_index,
        subtask_shot_index=subtask_shot_index,
        frame_type=frame_type,
        bitstring=np.array(bitstring),
    )


def make_task_definition(
    *,
    programs: list[Program] | None = None,
    subtasks: list[Subtask] | None = None,
) -> TaskDefinition:
    if programs is None:
        programs = [Program(content="program-0")]
    if subtasks is None:
        subtasks = [Subtask(program_index=0, num_shots=10)]
    return TaskDefinition(
        program_language="flair.v1",
        programs=programs,
        subtasks=subtasks,
    )


@pytest.fixture
def storage():
    return DictStorage()


def add_task_definition(
    storage,
    task_id: str,
    task_definition: TaskDefinition,
    creation_time: datetime = CREATION_TIME,
):
    storage.add_task_definition(task_id, task_definition, creation_time)


def test_default_shot_filter_is_detected_frame_type(storage):
    res = GeminiLogicalResult(storage=storage)
    assert res.shot_filter.frame_type == "DETECTED"
    assert res.shot_filter.task_ids is None
    assert res.shot_filter.subtask_indices is None


def test_custom_shot_filter_is_used(storage):
    custom_filter = ShotFilter(task_ids=("task-1",), frame_type="RAW")
    res = GeminiLogicalResult(storage=storage, shot_filter=custom_filter)
    assert res.shot_filter is custom_filter


def test_task_ids_uses_shot_filter_task_ids_when_set(storage):
    add_task_definition(storage, "task-1", make_task_definition())
    add_task_definition(storage, "task-2", make_task_definition())

    res = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(task_ids=("task-1",))
    )
    assert res.task_ids() == {"task-1"}


def test_task_ids_falls_back_to_storage_when_filter_has_none(storage):
    add_task_definition(storage, "task-1", make_task_definition())
    add_task_definition(storage, "task-2", make_task_definition())

    res = GeminiLogicalResult(storage=storage)
    assert res.task_ids() == {"task-1", "task-2"}


def test_task_ids_empty_storage(storage):
    res = GeminiLogicalResult(storage=storage)
    assert res.task_ids() == set()


def test_arguments_returns_subtask_arguments(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1, arguments={"a": 1}),
                Subtask(program_index=0, num_shots=2, arguments={"b": 2}),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    args = sorted(res.arguments(), key=lambda d: next(iter(d.keys())))  # type: ignore
    assert args == [{"a": 1}, {"b": 2}]


def test_arguments_filtered_by_task(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=2, arguments={"b": 2})],
        ),
    )

    res = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(task_ids=("task-1",))
    )
    assert res.arguments() == [{"a": 1}]


def test_subtasks_filtered_by_task_returns_merged_view(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=2),
            ],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=3)],
        ),
    )

    res = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(task_ids=("task-1",))
    )
    subtasks = res.subtasks()
    # Merged view drops task_id; only task-1's two subtask_indices remain.
    assert all("task_id" not in s for s in subtasks)
    assert sorted(s["subtask_index"] for s in subtasks) == [0, 1]


def test_subtasks_merges_across_task_ids_and_aggregates_num_shots(storage):
    """Two task_ids with the same subtask structure should collapse to one
    bucket per subtask_index, with num_shots summed."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=10, arguments={"a": 1}),
                Subtask(program_index=0, num_shots=20, arguments={"a": 2}),
            ],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=5, arguments={"a": 1}),
                Subtask(program_index=0, num_shots=7, arguments={"a": 2}),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    subtasks = res.subtasks()

    assert [s["subtask_index"] for s in subtasks] == [0, 1]
    assert all("task_id" not in s for s in subtasks)
    assert subtasks[0]["num_shots"] == 15  # 10 + 5
    assert subtasks[1]["num_shots"] == 27  # 20 + 7
    assert subtasks[0]["arguments"] == {"a": 1}
    assert subtasks[1]["arguments"] == {"a": 2}


def test_full_subtasks_returns_per_task_records(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=2)]),
    )

    res = GeminiLogicalResult(storage=storage)
    full = res.full_subtasks()

    assert sorted((s["task_id"], s["subtask_index"]) for s in full) == [
        ("task-1", 0),
        ("task-2", 0),
    ]
    # Per-task num_shots should NOT be aggregated in the full view.
    assert sorted(s["num_shots"] for s in full) == [1, 2]


def test_subtasks_does_not_mutate_storage(storage):
    """Calling subtasks() must not strip task_id from the storage's own dicts.
    Verifies the copy-on-read contract of DictStorage.get_subtasks."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=3)]),
    )

    res = GeminiLogicalResult(storage=storage)
    res.subtasks()
    res.subtasks()  # must not raise KeyError on the second call

    # full_subtasks() must still expose task_id afterward.
    full = res.full_subtasks()
    assert full[0]["task_id"] == "task-1"


def test_shot_results_returns_arrays_per_subtask(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=2),
                Subtask(program_index=0, num_shots=2),
            ],
        ),
    )
    s00 = make_shot(shot_index=0, subtask_index=0, bitstring=(True, False))
    s01 = make_shot(shot_index=1, subtask_index=0, bitstring=(False, True))
    s10 = make_shot(shot_index=2, subtask_index=1, bitstring=(True, True))
    s11 = make_shot(shot_index=3, subtask_index=1, bitstring=(False, False))
    storage.add_shots([s00, s01, s10, s11])

    res = GeminiLogicalResult(storage=storage)
    shots_per_subtask = res.shot_results()

    assert len(shots_per_subtask) == 2
    np.testing.assert_array_equal(
        np.sort(shots_per_subtask[0], axis=0),
        np.sort(np.array([[True, False], [False, True]]), axis=0),
    )
    np.testing.assert_array_equal(
        np.sort(shots_per_subtask[1], axis=0),
        np.sort(np.array([[True, True], [False, False]]), axis=0),
    )


def test_shot_results_sorts_subtasks_by_index(storage):
    """Even if storage returns subtasks out of order, results should be ordered by subtask_index."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=1),
            ],
        ),
    )
    storage.add_shots(
        [
            make_shot(shot_index=0, subtask_index=0, bitstring=(True, False)),
            make_shot(shot_index=1, subtask_index=1, bitstring=(False, True)),
        ]
    )

    res = GeminiLogicalResult(storage=storage)

    # Override get_subtasks to return them in reverse order to test sorting.
    original_get_subtasks = storage.get_subtasks

    def reversed_get_subtasks(*args, **kwargs):
        return list(reversed(original_get_subtasks(*args, **kwargs)))

    storage.get_subtasks = reversed_get_subtasks

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 2
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, False]]))
    np.testing.assert_array_equal(shots_per_subtask[1], np.array([[False, True]]))


def test_shot_results_default_filter_excludes_non_detected_frames(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=2)]),
    )
    detected = make_shot(shot_index=0, frame_type="DETECTED", bitstring=(True, False))
    raw = make_shot(shot_index=1, frame_type="RAW", bitstring=(False, True))
    storage.add_shots([detected, raw])

    res = GeminiLogicalResult(storage=storage)
    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, False]]))


# tests for the postprocessing utility
def test_postproc_termmeasure():
    @logical.kernel(aggressive_unroll=True)
    def test_qalloc_4_5():
        qubits = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(qubits[0])
        squin.cx(qubits[0], qubits[1])
        return logical.terminal_measure(qubits)

    expected_indices = [64, 65, 66, 67, 68, 69, 70, 80, 81, 82, 83, 84, 85, 86]
    meas_postproc, _ = get_slm_mapping_postprocessing(test_qalloc_4_5)
    mock_shots = np.asarray([np.zeros(160, dtype=bool)])
    mock_shots[:, expected_indices] = True
    ret_shots = meas_postproc(mock_shots)
    assert ret_shots == [[True for _ in range(len(expected_indices))]]


def test_postproc_termmeasure_one():
    @logical.kernel(aggressive_unroll=True)
    def test_qalloc_4_5():
        qubits = logical.qalloc_at(ilist.IList([4]))
        squin.h(qubits[0])
        # squin.cx(qubits[0], qubits[1])
        return logical.terminal_measure(qubits)

    expected_indices = [64, 65, 66, 67, 68, 69, 70]
    meas_postproc, _ = get_slm_mapping_postprocessing(test_qalloc_4_5)
    mock_shots = np.asarray([np.zeros(160, dtype=bool)])
    mock_shots[:, expected_indices] = True
    ret_shots = meas_postproc(mock_shots)
    assert ret_shots == [[True for _ in range(len(expected_indices))]]


def test_postproc_termmeasure_none():
    # An empty program no longer passes validation; this pins post-processing.
    @logical.kernel(aggressive_unroll=True, verify=False)
    def test_qalloc_4_5():
        _ = logical.qalloc_at(ilist.IList([]))
        # squin.h(qubits[0])
        # squin.cx(qubits[0], qubits[1])
        # return logical.terminal_measure(qubits)

    with pytest.raises(
        ShotRemappingException, match="no measurement statement to project"
    ):
        _ = get_slm_mapping_postprocessing(test_qalloc_4_5)


def test_postproc_termmeasure_all():
    @logical.kernel(aggressive_unroll=True)
    def test_qalloc_4_5():
        qubits = logical.qalloc_at(ilist.IList([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        squin.h(qubits[0])
        squin.cx(qubits[0], qubits[1])
        return logical.terminal_measure(qubits)

    expected_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
    ]
    meas_postproc, _ = get_slm_mapping_postprocessing(test_qalloc_4_5)
    mock_shots = np.asarray([np.zeros(160, dtype=bool)])
    mock_shots[:, expected_indices] = True
    ret_shots = meas_postproc(mock_shots)
    assert ret_shots == [[True for _ in range(len(expected_indices))]]


def test_default_postproc_remapping():
    """A kernel returning a post-processing tuple rather than a nested
    ilist of measurements. The mapping comes from the analysis's
    measurement records, not the return-value shape, so the return type
    no longer constrains it (issue #967)."""

    @logical.kernel(aggressive_unroll=True)
    def test_qalloc_4_5():
        qubits = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(qubits[0])
        squin.cx(qubits[0], qubits[1])
        return logical.default_post_processing(qubits)

    _ = get_slm_mapping_postprocessing(test_qalloc_4_5)


def test_get_slm_mapping_postprocessing_maps_and_validates_frames(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("SLM post-processing must come from the source kernel")

    # Patch the class itself, not ``utils_module.atom``: utils imports the
    # atom package lazily inside the function (see #985's import-cycle fix), so
    # it is not a module attribute there.
    monkeypatch.setattr(
        atom.AtomInterpreter,
        "get_post_processing",
        fail_if_called,
        raising=False,
    )
    map_shots, _ = utils_module.get_slm_mapping_postprocessing(_kernel_a)

    empty_slm_frame = np.zeros((1, 160), dtype=bool)
    assert map_shots(empty_slm_frame, invert=False) == [[False] * 14]
    assert map_shots(empty_slm_frame, invert=True) == [[True] * 14]

    with pytest.raises(ValueError, match="Expected a 2-D array"):
        map_shots(np.zeros(160, dtype=bool), invert=False)

    with pytest.raises(ValueError, match="Expected 160 Zone-0 columns"):
        map_shots(np.zeros((1, 159), dtype=bool), invert=False)


def test_postprocessing_functions_compile_once_independent_of_bit_convention(
    storage, monkeypatch, mocker
):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )

    post_processing = PostProcessing(
        emit_return=lambda rows: (list(row) for row in rows),
        emit_detectors=lambda rows: (list(row) for row in rows),
        emit_observables=lambda rows: (list(row) for row in rows),
    )
    mapping_calls = []

    def get_mapping(_kernel):
        mapping_calls.append(_kernel)
        return lambda shots, *, invert: shots.tolist(), post_processing

    monkeypatch.setattr(utils_module, "get_slm_mapping_postprocessing", get_mapping)
    decode_spy = mocker.spy(result_module.logical.kernel, "decode_json")

    result = GeminiLogicalResult(storage=storage)
    mappings = result.postprocessing_functions()
    assert result.postprocessing_functions() is mappings

    assert len(mapping_calls) == 1
    # The duplicate program index across task IDs is decoded once and served
    # from the convention-independent cache.
    assert decode_spy.call_count == 1


@pytest.mark.parametrize("frame_type", ["DETECTED", "SORTED", "RAW", None])
def test_slm_result_views_use_detected_and_sorted_frames(
    storage, monkeypatch, frame_type
):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=1)],
        ),
    )
    detected = (True, False) + (False,) * 158
    sorted_ = (False, True) + (False,) * 158
    storage.add_shots(
        [
            make_shot(shot_index=0, frame_type="DETECTED", bitstring=detected),
            make_shot(shot_index=0, frame_type="SORTED", bitstring=sorted_),
        ]
    )

    post_processing = PostProcessing(
        emit_return=lambda rows: (list(row) for row in rows),
        emit_detectors=lambda rows: ([row[0]] for row in rows),
        emit_observables=lambda rows: ([row[1]] for row in rows),
    )
    mapping_calls = []
    conventions = []

    def get_mapping(_kernel):
        mapping_calls.append(_kernel)

        def postprocess(shots, *, invert):
            conventions.append(invert)
            measurements = shots[:, :2]
            return (~measurements if invert else measurements).tolist()

        return postprocess, post_processing

    monkeypatch.setattr(
        utils_module,
        "get_slm_mapping_postprocessing",
        get_mapping,
    )

    result = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(frame_type=frame_type)
    )
    assert result.measurements == [[[False, True]]]
    assert result.return_values == [[[False, True]]]
    assert result.detectors == [[[False]]]
    assert result.observables == [[[True]]]
    assert result.filling_at_start == [[[False, True]]]
    assert result.shot_filter.frame_type == frame_type
    assert len(mapping_calls) == 1
    assert conventions.count(True) == 4
    assert conventions.count(False) == 1


def test_measurements_re_reads_shared_storage(storage, monkeypatch):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=2)],
        ),
    )
    storage.add_shots(
        [make_shot(shot_index=0, bitstring=(False, True) + (False,) * 158)]
    )

    post_processing = PostProcessing(
        emit_return=lambda rows: (list(row) for row in rows),
        emit_detectors=lambda rows: (list(row) for row in rows),
        emit_observables=lambda rows: (list(row) for row in rows),
    )

    def get_mapping(_kernel):
        return lambda shots, *, invert: shots[:, :2].tolist(), post_processing

    monkeypatch.setattr(utils_module, "get_slm_mapping_postprocessing", get_mapping)

    result = GeminiLogicalResult(storage=storage)
    assert result.measurements == [[[False, True]]]

    storage.add_shots(
        [make_shot(shot_index=1, bitstring=(True, False) + (False,) * 158)]
    )

    assert result.measurements == [[[False, True], [True, False]]]


def test_slm_result_views_return_empty_lists_after_shot_filter(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=1)],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                shot_index=0,
                frame_type="DETECTED",
                bitstring=(False,) * 160,
            ),
            make_shot(
                shot_index=0,
                frame_type="SORTED",
                bitstring=(True,) * 160,
            ),
        ]
    )

    result = GeminiLogicalResult(storage=storage).where_shots(lambda shot: False)

    assert result.measurements == [[]]
    assert result.return_values == [[]]
    assert result.detectors == [[]]
    assert result.observables == [[]]
    assert result.filling_at_start == [[]]


def test_postselect_on_fully_filled_returns_result_filtered_by_raw_slm_bits(
    storage, monkeypatch
):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content="program-0"), Program(content="program-1")],
            subtasks=[
                Subtask(program_index=0, num_shots=2),
                Subtask(program_index=1, num_shots=2),
            ],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                shot_index=0,
                subtask_index=0,
                frame_type="SORTED",
                bitstring=(True, True) + (False,) * 158,
            ),
            make_shot(
                shot_index=0,
                subtask_index=0,
                frame_type="DETECTED",
                bitstring=(False, True) + (False,) * 158,
            ),
            make_shot(
                shot_index=1,
                subtask_shot_index=1,
                subtask_index=0,
                frame_type="SORTED",
                bitstring=(True, False) + (False,) * 158,
            ),
            make_shot(
                shot_index=1,
                subtask_shot_index=1,
                subtask_index=0,
                frame_type="DETECTED",
                bitstring=(True, True) + (False,) * 158,
            ),
            make_shot(
                shot_index=2,
                subtask_index=1,
                frame_type="SORTED",
                bitstring=(False, False, True, True) + (False,) * 156,
            ),
            make_shot(
                shot_index=2,
                subtask_index=1,
                frame_type="DETECTED",
                bitstring=(False, False, False, True) + (False,) * 156,
            ),
            make_shot(
                shot_index=3,
                subtask_shot_index=1,
                subtask_index=1,
                frame_type="SORTED",
                bitstring=(True, True, True, False) + (False,) * 156,
            ),
            make_shot(
                shot_index=3,
                subtask_shot_index=1,
                subtask_index=1,
                frame_type="DETECTED",
                bitstring=(True, True, True, True) + (False,) * 156,
            ),
        ]
    )

    post_processing = PostProcessing(
        emit_return=lambda rows: (list(row) for row in rows),
        emit_detectors=lambda rows: (list(row) for row in rows),
        emit_observables=lambda rows: (list(row) for row in rows),
    )
    mapper_calls = []

    def mapper(indices):
        def slm_to_raw(shots, *, invert):
            mapper_calls.append((tuple(indices), invert, shots.copy()))
            return shots[:, indices].tolist()

        return slm_to_raw

    monkeypatch.setattr(
        GeminiLogicalResult,
        "_slm_postprocessing_functions",
        lambda _self: {
            0: (mapper([0, 1]), post_processing),
            1: (mapper([2, 3]), post_processing),
        },
    )

    result = GeminiLogicalResult(storage=storage).postselect_on_fully_filled()

    assert isinstance(result, GeminiLogicalResult)
    assert [(indices, invert) for indices, invert, _ in mapper_calls] == [
        ((0, 1), False),
        ((0, 1), False),
        ((2, 3), False),
        ((2, 3), False),
    ]
    selected_shots = result.shot_results()
    np.testing.assert_array_equal(
        selected_shots[0], np.array([[False, True] + [False] * 158])
    )
    np.testing.assert_array_equal(
        selected_shots[1], np.array([[False, False, False, True] + [False] * 156])
    )


def test_return_values_rejects_compact_frames(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=1),
            ],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                shot_index=0,
                subtask_index=0,
                bitstring=(False,) * 160,
            ),
            make_shot(
                shot_index=1,
                subtask_index=1,
                bitstring=(False, True),
            ),
        ]
    )

    with pytest.raises(ValueError, match="Expected 160 Zone-0 columns, got 2"):
        _ = GeminiLogicalResult(storage=storage).return_values


def test_shot_results_empty_when_no_subtasks(storage):
    res = GeminiLogicalResult(storage=storage)
    assert res.shot_results() == []


def test_postprocessing_functions_decodes_each_kernel(storage, mocker):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON), Program(content=KERNEL_B_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=1)],
        ),
    )

    decode_spy = mocker.spy(result_module.logical.kernel, "decode_json")
    mapping_spy = mocker.spy(utils_module, "get_slm_mapping_postprocessing")

    res = GeminiLogicalResult(storage=storage)
    funcs = res.postprocessing_functions()

    # Each program in storage gets decoded and compiled into an SLM mapper once.
    assert decode_spy.call_count == 2
    assert [c.args[0] for c in decode_spy.call_args_list] == [
        KERNEL_A_JSON,
        KERNEL_B_JSON,
    ]
    assert mapping_spy.call_count == 2
    assert set(funcs.keys()) == {0, 1}


def test_postprocessing_functions_filters_by_shot_filter_task_ids(storage, mocker):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(programs=[Program(content=KERNEL_B_JSON)]),
    )

    decode_spy = mocker.spy(result_module.logical.kernel, "decode_json")

    res = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(task_ids=("task-1",))
    )
    res.postprocessing_functions()

    # Only task-1's program should be decoded.
    assert decode_spy.call_count == 1
    assert decode_spy.call_args.args[0] == KERNEL_A_JSON


def test_postprocessing_rejects_different_kernels_at_same_program_index(storage):
    add_task_definition(
        storage,
        "task-a",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )
    add_task_definition(
        storage,
        "task-b",
        make_task_definition(programs=[Program(content=KERNEL_B_JSON)]),
    )

    result = GeminiLogicalResult(storage=storage)

    with pytest.raises(
        ValueError,
        match="different kernels for program_index=0: 'task-a' and 'task-b'",
    ):
        result.postprocessing_functions()


def test_postprocessing_program_validation_respects_narrowed_task_scope(storage):
    add_task_definition(
        storage,
        "task-a",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )
    add_task_definition(
        storage,
        "task-b",
        make_task_definition(programs=[Program(content=KERNEL_B_JSON)]),
    )

    result = GeminiLogicalResult(storage=storage).where_subtasks(
        lambda subtask: subtask["task_id"] == "task-a"
    )

    assert set(result.postprocessing_functions()) == {0}


def test_postprocessing_functions_uses_cached_compilation(storage, mocker):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )

    decode_spy = mocker.spy(result_module.logical.kernel, "decode_json")
    mapping_spy = mocker.spy(utils_module, "get_slm_mapping_postprocessing")

    res = GeminiLogicalResult(storage=storage)
    res.postprocessing_functions()
    res.postprocessing_functions()

    assert decode_spy.call_count == 1
    assert mapping_spy.call_count == 1


def test_return_values_rebuilds_const_hints_after_serialization(storage):
    """Stored logical kernels retain their post-processing behavior.

    Kirin JSON does not preserve SSA ``const`` hints. The detector indexes its
    terminal-measurement result with literal indices, so this covers the
    serialization path used when retrieving logical task results.
    """
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_WITH_POSTPROCESSING_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=2)],
        ),
    )
    storage.add_shots(
        [
            make_shot(shot_index=0, bitstring=(False,) * 160),
            make_shot(shot_index=1, bitstring=(True,) * 160),
        ]
    )

    result = GeminiLogicalResult(storage=storage).return_values

    assert result == [[False, False]]


def test_return_values_applies_postprocessing_per_subtask(storage, monkeypatch):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[
                Subtask(program_index=0, num_shots=1),
                Subtask(program_index=0, num_shots=1),
            ],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                shot_index=0,
                subtask_index=0,
                bitstring=(True, False) + (False,) * 158,
            ),
            make_shot(
                shot_index=1,
                subtask_index=1,
                bitstring=(False, True) + (False,) * 158,
            ),
        ]
    )

    post_processing = PostProcessing(
        emit_return=lambda rows: (tuple(row) for row in rows),
        emit_detectors=lambda rows: ([] for _ in rows),
        emit_observables=lambda rows: ([] for _ in rows),
    )

    def get_mapping(_kernel):
        return lambda shots, *, invert: shots[:, :2].tolist(), post_processing

    monkeypatch.setattr(utils_module, "get_slm_mapping_postprocessing", get_mapping)

    res = GeminiLogicalResult(storage=storage)
    out = res.return_values

    assert out == [[(True, False)], [(False, True)]]


def test_return_values_merges_shots_across_task_ids(storage, monkeypatch):
    """Two task_ids with the same program/subtask structure should merge
    their shots into one bucket per subtask_index, and run postprocessing
    once on the combined array."""
    add_task_definition(
        storage,
        "task-A",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=1)],
        ),
    )
    add_task_definition(
        storage,
        "task-B",
        make_task_definition(
            programs=[Program(content=KERNEL_A_JSON)],
            subtasks=[Subtask(program_index=0, num_shots=1)],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                task_id="task-A",
                shot_index=0,
                subtask_index=0,
                bitstring=(True, False) + (False,) * 158,
            ),
            make_shot(
                task_id="task-B",
                shot_index=0,
                subtask_index=0,
                bitstring=(False, True) + (False,) * 158,
            ),
        ]
    )

    post_processing = PostProcessing(
        emit_return=lambda rows: (tuple(row) for row in rows),
        emit_detectors=lambda rows: ([] for _ in rows),
        emit_observables=lambda rows: ([] for _ in rows),
    )

    def get_mapping(_kernel):
        return lambda shots, *, invert: shots[:, :2].tolist(), post_processing

    monkeypatch.setattr(utils_module, "get_slm_mapping_postprocessing", get_mapping)

    res = GeminiLogicalResult(storage=storage)
    out = res.return_values

    assert len(out) == 1
    assert sorted(out[0]) == sorted([(True, False), (False, True)])


def test_postprocessing_functions_dedupes_across_task_ids(storage, mocker):
    """When multiple task_ids share an identical program at the same
    program_index, decoding and SLM post-processing compilation run once."""
    add_task_definition(
        storage,
        "task-A",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )
    add_task_definition(
        storage,
        "task-B",
        make_task_definition(programs=[Program(content=KERNEL_A_JSON)]),
    )

    decode_spy = mocker.spy(result_module.logical.kernel, "decode_json")
    mapping_spy = mocker.spy(utils_module, "get_slm_mapping_postprocessing")

    res = GeminiLogicalResult(storage=storage)
    funcs = res.postprocessing_functions()

    assert list(funcs.keys()) == [0]
    assert decode_spy.call_count == 1
    assert mapping_spy.call_count == 1


def test_arguments_returns_one_entry_per_merged_subtask(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1, arguments={"a": 1}),
                Subtask(program_index=0, num_shots=2, arguments={"b": 2}),
            ],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=3, arguments={"a": 1}),
                Subtask(program_index=0, num_shots=4, arguments={"b": 2}),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    args = res.arguments()
    # One entry per merged subtask, not per (task_id, subtask_index).
    assert args == [{"a": 1}, {"b": 2}]


def test_full_arguments_returns_one_entry_per_storage_row(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=2, arguments={"a": 1})],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    full = res.full_arguments()
    assert full == [{"a": 1}, {"a": 1}]


def test_construction_does_not_validate_eagerly(storage):
    """A GeminiLogicalResult should be constructible over storage holding mismatched
    task definitions — only methods that depend on the merge assumption
    should trigger validation."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 999})],
        ),
    )

    # Should not raise — construction is now lazy with respect to validation.
    GeminiLogicalResult(storage=storage)


def test_validate_passes_for_consistent_task_ids(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=5, arguments={"a": 1})],
        ),
    )

    # No raise — different num_shots is fine, same program_index/arguments.
    GeminiLogicalResult(storage=storage).validate()


def test_validate_passes_with_single_task_in_filter(storage):
    """When the filter narrows to a single task_id, the rest of storage is irrelevant."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=1, num_shots=5, arguments={"a": 999})],
        ),
    )

    GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(task_ids=("task-1",))
    ).validate()


def test_validate_passes_with_single_task_in_storage(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    GeminiLogicalResult(storage=storage).validate()


def test_validate_raises_on_program_index_mismatch(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content="a"), Program(content="b")],
            subtasks=[Subtask(program_index=0, num_shots=1)],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            programs=[Program(content="a"), Program(content="b")],
            subtasks=[Subtask(program_index=1, num_shots=1)],
        ),
    )

    with pytest.raises(ValueError, match="program_index"):
        GeminiLogicalResult(storage=storage).validate()


def test_validate_raises_on_arguments_mismatch(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 2})],
        ),
    )

    with pytest.raises(ValueError, match="arguments"):
        GeminiLogicalResult(storage=storage).validate()


def test_validate_error_message_mentions_verify_false(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 2})],
        ),
    )

    with pytest.raises(ValueError, match="verify=False"):
        GeminiLogicalResult(storage=storage).validate()


def test_validate_treats_none_and_empty_dict_arguments_as_equal(storage):
    """A subtask with arguments=None and another with arguments={} should not
    trip the validator — both encode "no parameters"."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments=None)],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={})],
        ),
    )

    GeminiLogicalResult(storage=storage).validate()


def test_validate_caches_after_first_successful_run(storage):
    """A successful validate() flips _is_valid; subsequent calls should
    short-circuit without re-reading storage."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )

    res = GeminiLogicalResult(storage=storage)
    res.validate()
    assert res._is_valid is True

    call_count = {"n": 0}
    original = storage.get_subtasks

    def counting_get_subtasks(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    storage.get_subtasks = counting_get_subtasks
    res.validate()  # second call must not hit storage
    assert call_count["n"] == 0


def test_subtasks_validates_by_default(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 2})],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    with pytest.raises(ValueError, match="arguments"):
        res.subtasks()


def test_subtasks_skips_validation_with_verify_false(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 2})],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    # Should not raise — first-encountered task_id's args silently win.
    merged = res.subtasks(verify=False)
    assert len(merged) == 1


@pytest.mark.parametrize(
    "method",
    ["arguments", "shot_results"],
)
def test_merge_methods_skip_validation_with_verify_false(storage, method):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            programs=[Program(content="k")],
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            programs=[Program(content="k")],
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 2})],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    # With verify=True default, each of these would raise.
    getattr(res, method)(verify=False)


def test_inspection_methods_work_even_with_inconsistent_storage(storage):
    """full_subtasks, full_arguments, and task_ids do not depend on the merge
    assumption and must work regardless of consistency."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=1, arguments={"a": 1})],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=99, num_shots=1, arguments={"a": 999})],
        ),
    )

    res = GeminiLogicalResult(storage=storage)
    assert res.task_ids() == {"task-1", "task-2"}
    assert len(res.full_subtasks()) == 2
    assert len(res.full_arguments()) == 2


# === where_subtasks ===


def test_where_subtasks_keeps_only_matching_subtasks(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=10),
                Subtask(program_index=0, num_shots=20),
                Subtask(program_index=0, num_shots=30),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage).where_subtasks(
        predicate=lambda s: s["num_shots"] > 15
    )

    full = res.full_subtasks()
    assert sorted(s["subtask_index"] for s in full) == [1, 2]


def test_where_subtasks_inherits_default_frame_type(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )

    res = GeminiLogicalResult(storage=storage).where_subtasks(predicate=lambda s: True)
    assert res.shot_filter.frame_type == "DETECTED"


def test_where_subtasks_inherits_custom_frame_type(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )

    base = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(frame_type="RAW")
    )
    res = base.where_subtasks(predicate=lambda s: True)
    assert res.shot_filter.frame_type == "RAW"


def test_where_subtasks_inherits_unset_frame_type(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )

    base = GeminiLogicalResult(storage=storage, shot_filter=ShotFilter(frame_type=None))
    res = base.where_subtasks(predicate=lambda s: True)
    assert res.shot_filter.frame_type is None


def test_where_subtasks_predicate_scoped_by_self_shot_filter(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )

    seen_task_ids = []

    def predicate(subtask):
        seen_task_ids.append(subtask["task_id"])
        return True

    base = GeminiLogicalResult(
        storage=storage, shot_filter=ShotFilter(task_ids=("task-1",))
    )
    base.where_subtasks(predicate=predicate)
    assert seen_task_ids == ["task-1"]


# === where_arguments ===


def test_where_arguments_keeps_only_matching_subtasks(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1, arguments={"theta": 0.1}),
                Subtask(program_index=0, num_shots=1, arguments={"theta": 0.5}),
                Subtask(program_index=0, num_shots=1, arguments={"theta": 0.9}),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage).where_arguments(
        predicate=lambda args: args["theta"] > 0.3  # type: ignore
    )

    full = res.full_subtasks()
    assert sorted(s["subtask_index"] for s in full) == [1, 2]


def test_where_arguments_passes_none_for_missing_arguments(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1, arguments=None),
                Subtask(program_index=0, num_shots=1, arguments={"x": 1}),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage).where_arguments(
        predicate=lambda args: args is None
    )

    full = res.full_subtasks()
    assert [s["subtask_index"] for s in full] == [0]


# === where_metadata ===


def test_where_metadata_decodes_json_dict_for_predicate(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(
                    program_index=0,
                    num_shots=1,
                    subtask_metadata=TaskMetadata(user_metadata='{"task_type": "ghz"}'),
                ),
                Subtask(
                    program_index=0,
                    num_shots=1,
                    subtask_metadata=TaskMetadata(
                        user_metadata='{"task_type": "bell"}'
                    ),
                ),
            ],
        ),
    )

    res = GeminiLogicalResult(storage=storage).where_metadata(
        predicate=lambda m: m is not None and m.get("task_type") == "ghz"
    )

    full = res.full_subtasks()
    assert [s["subtask_index"] for s in full] == [0]


# === where_shots ===


def test_where_shots_keeps_only_matching_shots(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=2)]),
    )
    storage.add_shots(
        [
            make_shot(shot_index=0, frame_type="DETECTED", bitstring=(True, True)),
            make_shot(shot_index=1, frame_type="DETECTED", bitstring=(False, True)),
        ]
    )

    res = GeminiLogicalResult(storage=storage).where_shots(
        predicate=lambda shot: bool(shot.bitstring.all())
    )

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, True]]))


def test_where_shots_cross_frame_precondition(storage):
    """SORTED-was-all-1 → DETECTED is returned for the qualifying pair."""
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=2)]),
    )
    storage.add_shots(
        [
            # Shot 0: SORTED all-1 → qualifies
            make_shot(shot_index=0, frame_type="SORTED", bitstring=(True, True)),
            make_shot(shot_index=0, frame_type="DETECTED", bitstring=(False, True)),
            # Shot 1: SORTED not all-1 → drops
            make_shot(shot_index=1, frame_type="SORTED", bitstring=(True, False)),
            make_shot(shot_index=1, frame_type="DETECTED", bitstring=(True, True)),
        ]
    )

    # Default GeminiLogicalResult has frame_type="DETECTED"; predicate_filter overrides
    # the predicate-side frame to SORTED while the result still fetches DETECTED.
    res = GeminiLogicalResult(storage=storage).where_shots(
        predicate=lambda shot: bool(shot.bitstring.all()),
        predicate_filter=ShotFilter(frame_type="SORTED"),
    )

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    # Shot 0's DETECTED bitstring
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[False, True]]))


def test_where_shots_inherits_unset_frame_type(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )
    storage.add_shots(
        [
            make_shot(shot_index=0, frame_type="SORTED", bitstring=(True, True)),
            make_shot(shot_index=0, frame_type="DETECTED", bitstring=(False, True)),
        ]
    )

    base = GeminiLogicalResult(storage=storage, shot_filter=ShotFilter(frame_type=None))
    res = base.where_shots(
        predicate=lambda shot: bool(shot.bitstring.all()),
        predicate_filter=ShotFilter(frame_type="SORTED"),
    )

    # No frame_type filter on output → both SORTED and DETECTED rows returned
    shots = list(storage.get_shots(shot_filter=res.shot_filter))
    assert {s.frame_type for s in shots} == {"SORTED", "DETECTED"}


# === multi-task: pair filters propagate through shot_results ===


def test_where_subtasks_filters_shots_per_task_id(storage):
    """Two tasks share a subtask_index; predicate matches only task-1's subtask.
    shot_results() must return task-1's shots only — exercises both the pair
    filter's task_id component and the _shot_results_for_subtasks merge.
    """
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=10)],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[Subtask(program_index=0, num_shots=5)],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                task_id="task-1",
                shot_index=0,
                subtask_index=0,
                bitstring=(True, False),
            ),
            make_shot(
                task_id="task-2",
                shot_index=0,
                subtask_index=0,
                bitstring=(False, True),
            ),
        ]
    )

    res = GeminiLogicalResult(storage=storage).where_subtasks(
        predicate=lambda s: s["num_shots"] > 7
    )

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, False]]))


def test_where_arguments_filters_shots_per_task_id(storage):
    # Use numeric arg values to avoid pydantic's bool→float coercion in
    # the Subtask model, which would otherwise interfere with predicates.
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1, arguments={"theta": 1.0}),
            ],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[
                Subtask(program_index=0, num_shots=1, arguments={"theta": 0.0}),
            ],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                task_id="task-1",
                shot_index=0,
                subtask_index=0,
                bitstring=(True, False),
            ),
            make_shot(
                task_id="task-2",
                shot_index=0,
                subtask_index=0,
                bitstring=(False, True),
            ),
        ]
    )

    res = GeminiLogicalResult(storage=storage).where_arguments(
        predicate=lambda a: a["theta"] > 0.5  # type: ignore
    )

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, False]]))


def test_where_metadata_filters_shots_per_task_id(storage):
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(
            subtasks=[
                Subtask(
                    program_index=0,
                    num_shots=1,
                    subtask_metadata=TaskMetadata(user_metadata='{"keep": true}'),
                ),
            ],
        ),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(
            subtasks=[
                Subtask(
                    program_index=0,
                    num_shots=1,
                    subtask_metadata=TaskMetadata(user_metadata='{"keep": false}'),
                ),
            ],
        ),
    )
    storage.add_shots(
        [
            make_shot(
                task_id="task-1",
                shot_index=0,
                subtask_index=0,
                bitstring=(True, False),
            ),
            make_shot(
                task_id="task-2",
                shot_index=0,
                subtask_index=0,
                bitstring=(False, True),
            ),
        ]
    )

    res = GeminiLogicalResult(storage=storage).where_metadata(
        predicate=lambda m: m is not None and m.get("keep") is True
    )

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, False]]))


def test_where_shots_filters_pairs_per_task_id(storage):
    """Cross-frame precondition spanning two tasks: each task's qualifying
    shot is independent. task-1's SORTED is all-1, task-2's is not — only
    task-1's DETECTED row should survive.
    """
    add_task_definition(
        storage,
        "task-1",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )
    add_task_definition(
        storage,
        "task-2",
        make_task_definition(subtasks=[Subtask(program_index=0, num_shots=1)]),
    )
    storage.add_shots(
        [
            make_shot(
                task_id="task-1",
                shot_index=0,
                frame_type="SORTED",
                bitstring=(True, True),
            ),
            make_shot(
                task_id="task-1",
                shot_index=0,
                frame_type="DETECTED",
                bitstring=(True, False),
            ),
            make_shot(
                task_id="task-2",
                shot_index=0,
                frame_type="SORTED",
                bitstring=(True, False),
            ),
            make_shot(
                task_id="task-2",
                shot_index=0,
                frame_type="DETECTED",
                bitstring=(False, True),
            ),
        ]
    )

    res = GeminiLogicalResult(storage=storage).where_shots(
        predicate=lambda shot: bool(shot.bitstring.all()),
        predicate_filter=ShotFilter(frame_type="SORTED"),
    )

    shots_per_subtask = res.shot_results()
    assert len(shots_per_subtask) == 1
    # Only task-1's DETECTED row, not task-2's
    np.testing.assert_array_equal(shots_per_subtask[0], np.array([[True, False]]))
