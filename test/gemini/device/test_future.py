from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
import pytest
from bloqade.core.device import (
    ShotFilter,
    ShotResult,
    DictStorage,
    future as future_module,
)
from bloqade.core.device.future import ApiFetchOptions
from qlam_core.plugins.compilations.api import CompilationsClient
from qlam_core.plugins.tasks.api.client import TasksClient
from qlam_core.plugins.results.api.client import ResultsClient
from qlam_core.plugins.definitions.api.client import DefinitionsClient
from qlam_core.plugins.tasks.api.tasks_models import (
    Task,
    Program,
    Subtask,
    TaskStatus,
    GroupSummary,
    TaskDefinition,
)
from qlam_core.plugins.definitions.api.definitions_models import (
    Program as DefinitionProgram,
    Subtask as DefinitionSubtask,
    GroupSummary as DefinitionGroupSummary,
    TaskDefinitionResponse,
)

from bloqade.gemini import GeminiLogicalFuture

CREATION_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def make_task(status, *, task_id="task-1", error_reasons=None, **extras):
    return Task(
        id=task_id,
        task_status=status,
        created_by=uuid4(),
        created_date=datetime.now(timezone.utc),
        # ``group`` became a required field in qlam-core 0.6.0. Tests only
        # ever parse Tasks the API hands back, so any resolvable group
        # will do; nothing here reads it.
        group=GroupSummary(id=uuid4()),
        error_reasons=error_reasons or [],
        **extras,
    )


@pytest.fixture
def _auth_no_op(mocker):
    """Pulled out so multiple client fixtures in one test don't both try to
    autospec `authenticate` — the second call would see the first patch's
    Mock as the spec source and raise InvalidSpecError."""
    mocker.patch.object(future_module.AuthMixin, "authenticate", autospec=True)


@pytest.fixture
def tasks_client(mocker, _auth_no_op):
    mock_class = mocker.patch.object(future_module, "TasksClient", spec=TasksClient)
    client = mock_class.return_value
    # Self-loop so `with TasksClient(ctx) as c:` binds `c` to the same mock
    # the fixture returns. Without this, `c` is a fresh sub-mock and
    # configuration like `.get.side_effect` lands on the wrong object.
    client.__enter__.return_value = client
    return client


@pytest.fixture
def results_client(mocker, _auth_no_op):
    mock_class = mocker.patch.object(future_module, "ResultsClient", spec=ResultsClient)
    client = mock_class.return_value
    client.__enter__.return_value = client
    return client


@pytest.fixture
def compilations_client(mocker, _auth_no_op):
    mock_class = mocker.patch.object(
        future_module, "CompilationsClient", spec=CompilationsClient
    )
    client = mock_class.return_value
    client.__enter__.return_value = client
    return client


@pytest.fixture
def definitions_client(mocker, _auth_no_op):
    mock_class = mocker.patch.object(
        future_module, "DefinitionsClient", spec=DefinitionsClient
    )
    client = mock_class.return_value
    client.__enter__.return_value = client
    return client


# === _wait_for_completion ===


def test_wait_for_completion_polls_until_terminal(
    tasks_client, mocker, assert_called_with_kwargs
):
    tasks_client.get.side_effect = [
        make_task(TaskStatus.SCHEDULED),
        make_task(TaskStatus.EXECUTION_STARTED),
        make_task(TaskStatus.COMPLETED),
    ]
    mocker.patch.object(future_module.time, "sleep", autospec=True)

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut._wait_for_completion() == TaskStatus.COMPLETED
    assert tasks_client.get.call_count == 3
    assert_called_with_kwargs(tasks_client.get, id="task-1")


def test_wait_for_completion_raises_on_failure(tasks_client, mocker):
    tasks_client.get.return_value = make_task(
        TaskStatus.FAILED, error_reasons=["compile error"]
    )
    mocker.patch.object(future_module.time, "sleep", autospec=True)

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    with pytest.raises(ValueError, match="compile error"):
        fut._wait_for_completion()


def test_wait_for_completion_raises_on_payload_processing_error(tasks_client, mocker):
    tasks_client.get.return_value = make_task(
        TaskStatus.PAYLOAD_PROCESSING_ERROR, error_reasons=["bad payload"]
    )
    mocker.patch.object(future_module.time, "sleep", autospec=True)

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    with pytest.raises(ValueError, match="bad payload"):
        fut._wait_for_completion()


def test_wait_for_completion_raises_on_cancelled(tasks_client, mocker):
    tasks_client.get.return_value = make_task(TaskStatus.CANCELLED)
    mocker.patch.object(future_module.time, "sleep", autospec=True)

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    with pytest.raises(ValueError, match="cancelled"):
        fut._wait_for_completion()


def test_wait_for_completion_times_out(tasks_client, mocker):
    tasks_client.get.return_value = make_task(TaskStatus.SCHEDULED)
    mocker.patch.object(future_module.time, "sleep", autospec=True)
    mocker.patch.object(
        future_module.time, "monotonic", autospec=True, side_effect=[0.0, 0.0, 5.0]
    )

    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=DictStorage(),
        fetch_options=ApiFetchOptions(poll_interval_initial=0.0),
    )
    with pytest.raises(TimeoutError):
        fut._wait_for_completion(timeout=1.0)


def test_wait_for_completion_backoff_capped(tasks_client, mocker):
    """Backoff multiplies by `poll_interval_factor` per iteration but caps at
    `poll_interval_max`."""
    tasks_client.get.side_effect = [
        make_task(TaskStatus.SCHEDULED),
        make_task(TaskStatus.SCHEDULED),
        make_task(TaskStatus.SCHEDULED),
        make_task(TaskStatus.COMPLETED),
    ]
    sleeps = []
    mocker.patch.object(
        future_module.time, "sleep", autospec=True, side_effect=sleeps.append
    )

    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=DictStorage(),
        fetch_options=ApiFetchOptions(
            poll_interval_initial=1.0, poll_interval_factor=10.0, poll_interval_max=5.0
        ),
    )
    fut._wait_for_completion()

    # 1.0 → 5.0 (capped at max) → 5.0
    assert sleeps == [1.0, 5.0, 5.0]


# === get_task / status / done / cancelled ===


def test_get_task_returns_task(tasks_client, assert_called_with_kwargs):
    expected = make_task(TaskStatus.COMPLETED)
    tasks_client.get.return_value = expected

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut.get_task() is expected
    assert_called_with_kwargs(tasks_client.get, times=1, id="task-1")


def test_status_returns_current_status(tasks_client):
    tasks_client.get.return_value = make_task(TaskStatus.EXECUTION_STARTED)

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut.status() == TaskStatus.EXECUTION_STARTED


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
        TaskStatus.PAYLOAD_PROCESSING_ERROR,
    ],
)
def test_done_true_for_terminal_status(tasks_client, status):
    tasks_client.get.return_value = make_task(status)
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut.done() is True


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.CREATED,
        TaskStatus.PAYLOAD_PROCESSING,
        TaskStatus.SCHEDULED,
        TaskStatus.EXECUTION_STARTED,
        TaskStatus.EXECUTION_COMPLETED,
    ],
)
def test_done_false_for_non_terminal_status(tasks_client, status):
    tasks_client.get.return_value = make_task(status)
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut.done() is False


def test_cancelled_true_when_status_cancelled(tasks_client):
    tasks_client.get.return_value = make_task(TaskStatus.CANCELLED)
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut.cancelled() is True


def test_cancelled_false_when_status_not_cancelled(tasks_client):
    tasks_client.get.return_value = make_task(TaskStatus.COMPLETED)
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    assert fut.cancelled() is False


# === cancel ===


def test_cancel_calls_client_cancel_with_id(tasks_client, assert_called_with_kwargs):
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    fut.cancel()
    assert_called_with_kwargs(tasks_client.cancel, times=1, id="task-1")


def test_cancel_warns_on_exception(tasks_client, recwarn):
    tasks_client.cancel.side_effect = RuntimeError("boom")
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    fut.cancel()  # must not raise
    assert any("task-1" in str(w.message) for w in recwarn)
    assert any("boom" in str(w.message) for w in recwarn)


# === get_compilation ===


def test_get_compilation_with_explicit_id(
    compilations_client, assert_called_with_kwargs
):
    compilations_client.get.return_value = "compilation-obj"

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    result = fut.get_compilation(compilation_id="comp-explicit")

    assert result == "compilation-obj"
    assert_called_with_kwargs(compilations_client.get, times=1, id="comp-explicit")


def test_get_compilation_falls_back_to_task_compilation_id(
    tasks_client, compilations_client, assert_called_with_kwargs
):
    tasks_client.get.return_value = make_task(
        TaskStatus.COMPLETED, compilation_id="comp-from-task"
    )
    compilations_client.get.return_value = "compilation-obj"

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    result = fut.get_compilation()

    assert result == "compilation-obj"
    assert_called_with_kwargs(compilations_client.get, times=1, id="comp-from-task")


# === fetch / _fetch_subtask_page ===


def make_shot_resp(
    *,
    shot_index,
    subtask_index=0,
    subtask_shot_index=0,
    frame_type="DETECTED",
    bitstring=(True, False),
):
    return {
        "shot_index": shot_index,
        "subtask_index": subtask_index,
        "subtask_shot_index": subtask_shot_index,
        "frame_type": frame_type,
        "measurement": {"measurement_values": list(bitstring)},
    }


def make_subtask_resp(*, subtask_index=0, status="COMPLETED", shot_results):
    return {
        "subtask_index": subtask_index,
        "status": status,
        "shot_results": shot_results,
    }


def make_response(*, elements):
    return {"elements": elements}


def test_fetch_writes_shots_to_storage(results_client):
    results_client.get.side_effect = [
        # Page 0: one element with one COMPLETED subtask, two shots
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            shot_results=[
                                make_shot_resp(shot_index=0, bitstring=(True, False)),
                                make_shot_resp(shot_index=1, bitstring=(False, True)),
                            ]
                        )
                    ]
                }
            ]
        ),
        # Page 1: empty → signals done (full_subtask_page=False)
        make_response(elements=[]),
    ]

    storage = DictStorage()
    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=storage,
        fetch_options=ApiFetchOptions(subtasks_per_fetch=1, shots_per_fetch=10),
    )
    fut.fetch()

    shots = sorted(storage.get_shots(), key=lambda s: s.shot_index)
    assert len(shots) == 2
    assert shots[0].task_id == "task-1"
    assert shots[0].shot_index == 0
    np.testing.assert_array_equal(shots[0].bitstring, np.array([True, False]))
    np.testing.assert_array_equal(shots[1].bitstring, np.array([False, True]))


def test_fetch_paginates_subtasks(results_client):
    """With subtasks_per_fetch=1, each subtask fills its own page; fetch must
    keep paging until it hits an empty page."""
    results_client.get.side_effect = [
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            subtask_index=0,
                            shot_results=[make_shot_resp(shot_index=0)],
                        )
                    ]
                }
            ]
        ),
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            subtask_index=1,
                            shot_results=[
                                make_shot_resp(shot_index=10, subtask_index=1)
                            ],
                        )
                    ]
                }
            ]
        ),
        make_response(elements=[]),
    ]

    storage = DictStorage()
    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=storage,
        fetch_options=ApiFetchOptions(subtasks_per_fetch=1, shots_per_fetch=10),
    )
    fut.fetch()

    # Verify pagination: page 0, page 1, page 2.
    pages = [c.kwargs["page"] for c in results_client.get.call_args_list]
    assert pages == [0, 1, 2]
    assert {s.subtask_index for s in storage.get_shots()} == {0, 1}


def test_fetch_paginates_shots_within_subtask(results_client):
    """When a subtask has >= shots_per_fetch shots in one response, fetch
    pages shots within the same subtask_page before advancing."""
    results_client.get.side_effect = [
        # subtask_page 0, shots_page 0: full (2 shots == shots_per_fetch)
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            shot_results=[
                                make_shot_resp(shot_index=0),
                                make_shot_resp(shot_index=1),
                            ]
                        )
                    ]
                }
            ]
        ),
        # subtask_page 0, shots_page 1: not full (1 shot < shots_per_fetch)
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(shot_results=[make_shot_resp(shot_index=2)])
                    ]
                }
            ]
        ),
        # subtask_page 1: empty → done
        make_response(elements=[]),
    ]

    storage = DictStorage()
    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=storage,
        fetch_options=ApiFetchOptions(subtasks_per_fetch=1, shots_per_fetch=2),
    )
    fut.fetch()

    calls = results_client.get.call_args_list
    page_shots_pairs = [(c.kwargs["page"], c.kwargs["shots_page"]) for c in calls]
    assert page_shots_pairs == [(0, 0), (0, 1), (1, 0)]
    assert {s.shot_index for s in storage.get_shots()} == {0, 1, 2}


def test_fetch_tracks_first_incomplete_subtask_page(results_client):
    results_client.get.side_effect = [
        # Page 0: completed subtask
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            subtask_index=0,
                            status="COMPLETED",
                            shot_results=[make_shot_resp(shot_index=0)],
                        )
                    ]
                }
            ]
        ),
        # Page 1: incomplete subtask → should mark _first_incomplete_subtask_page=1
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            subtask_index=1,
                            status="EXECUTION_STARTED",
                            shot_results=[],
                        )
                    ]
                }
            ]
        ),
        # Page 2: empty → done
        make_response(elements=[]),
    ]

    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=DictStorage(),
        fetch_options=ApiFetchOptions(subtasks_per_fetch=1, shots_per_fetch=10),
    )
    fut.fetch()

    assert fut._first_incomplete_subtask_page == 1


def test_fetch_normalizes_frame_type_to_uppercase(results_client):
    results_client.get.side_effect = [
        make_response(
            elements=[
                {
                    "subtasks": [
                        make_subtask_resp(
                            shot_results=[
                                make_shot_resp(shot_index=0, frame_type="detected")
                            ]
                        )
                    ]
                }
            ]
        ),
        make_response(elements=[]),
    ]

    storage = DictStorage()
    fut = GeminiLogicalFuture(
        task_id="task-1",
        storage=storage,
        fetch_options=ApiFetchOptions(subtasks_per_fetch=1, shots_per_fetch=10),
    )
    fut.fetch()

    shots = list(storage.get_shots())
    assert shots[0].frame_type == "DETECTED"


# === results_from_storage / partial_result / result ===


def test_results_from_storage_default_filter_uses_task_id():
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    res = fut.results_from_storage()
    assert res.shot_filter.task_ids == ("task-1",)
    assert res.shot_filter.frame_type == "DETECTED"


def test_results_from_storage_uses_custom_filter():
    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    custom = ShotFilter(task_ids=("other",), frame_type="RAW")
    res = fut.results_from_storage(shot_filter=custom)
    assert res.shot_filter is custom


def test_partial_result_calls_fetch_then_returns_results(results_client):
    results_client.get.return_value = make_response(elements=[])

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    res = fut.partial_result()

    assert results_client.get.called
    assert res.shot_filter.task_ids == ("task-1",)


def test_result_waits_then_fetches_then_returns(tasks_client, results_client, mocker):
    tasks_client.get.return_value = make_task(TaskStatus.COMPLETED)
    results_client.get.return_value = make_response(elements=[])
    mocker.patch.object(future_module.time, "sleep", autospec=True)

    fut = GeminiLogicalFuture(task_id="task-1", storage=DictStorage())
    res = fut.result()

    assert tasks_client.get.called
    assert results_client.get.called
    assert res.shot_filter.task_ids == ("task-1",)


# === export_to / fetch_and_export_to ===


def make_shot(*, task_id="task-1", shot_index=0, bitstring=(True, False)):
    return ShotResult(
        task_id=task_id,
        shot_index=shot_index,
        subtask_index=0,
        subtask_shot_index=0,
        frame_type="DETECTED",
        bitstring=np.array(bitstring),
    )


def add_task_definition(
    storage,
    task_id: str,
    task_definition: TaskDefinition,
    creation_time: datetime = CREATION_TIME,
):
    storage.add_task_definition(task_id, task_definition, creation_time)


def _seed_storage(storage, *, task_id="task-1", n_shots=5):
    add_task_definition(
        storage,
        task_id,
        TaskDefinition(
            program_language="flair.v1",
            programs=[Program(content="kernel")],
            subtasks=[Subtask(program_index=0, num_shots=n_shots)],
        ),
    )
    storage.add_shots(
        [make_shot(task_id=task_id, shot_index=i) for i in range(n_shots)]
    )


def test_export_to_writes_in_chunks(mocker):
    src = DictStorage()
    dst = DictStorage()
    _seed_storage(src, n_shots=5)

    add_shots_spy = mocker.spy(dst, "add_shots")

    fut = GeminiLogicalFuture(task_id="task-1", storage=src)
    fut.export_to(dst, chunk_size=2)

    # 5 shots, chunk_size=2 → chunks of 2, 2, 1.
    chunk_sizes = [len(c.args[0]) for c in add_shots_spy.call_args_list]
    assert chunk_sizes == [2, 2, 1]
    assert len(list(dst.get_shots())) == 5


def test_export_to_copies_task_definitions():
    src = DictStorage()
    dst = DictStorage()
    _seed_storage(src, task_id="task-1", n_shots=2)
    _seed_storage(src, task_id="task-2", n_shots=2)

    fut = GeminiLogicalFuture(task_id="task-1", storage=src)
    fut.export_to(dst)

    assert dst.task_ids() == {"task-1", "task-2"}
    assert dst.get_task_definition("task-1") == src.get_task_definition("task-1")
    assert dst.get_task_creation_time("task-1") == src.get_task_creation_time("task-1")


def test_export_to_uses_filter_task_ids_when_set():
    src = DictStorage()
    dst = DictStorage()
    _seed_storage(src, task_id="task-1", n_shots=2)
    _seed_storage(src, task_id="task-2", n_shots=2)

    fut = GeminiLogicalFuture(task_id="task-1", storage=src)
    fut.export_to(dst, shot_filter=ShotFilter(task_ids=("task-1",)))

    # Only task-1's definition copied; task-2 was never selected.
    assert dst.task_ids() == {"task-1"}


def test_fetch_and_export_to_calls_fetch_then_export(results_client):
    results_client.get.return_value = make_response(elements=[])

    src = DictStorage()
    dst = DictStorage()
    _seed_storage(src, n_shots=3)

    fut = GeminiLogicalFuture(task_id="task-1", storage=src)
    fut.fetch_and_export_to(dst)

    assert results_client.get.called
    assert len(list(dst.get_shots())) == 3


# === from_storage ===


def test_from_storage_returns_future_with_single_task_id():
    storage = DictStorage()
    _seed_storage(storage, task_id="only-one", n_shots=1)

    fut = GeminiLogicalFuture.from_storage(storage=storage)
    assert fut.task_id == "only-one"
    assert fut.storage is storage


def test_from_storage_default_new_storage_is_storage():
    storage = DictStorage()
    _seed_storage(storage, task_id="t1", n_shots=1)

    fut = GeminiLogicalFuture.from_storage(storage=storage)
    assert fut.storage is storage


def test_from_storage_uses_new_storage_when_provided():
    storage = DictStorage()
    new_storage = DictStorage()
    _seed_storage(storage, task_id="t1", n_shots=1)

    fut = GeminiLogicalFuture.from_storage(storage=storage, new_storage=new_storage)
    assert fut.storage is new_storage


def test_from_storage_raises_when_storage_empty():
    with pytest.raises(ValueError, match="no task IDs"):
        GeminiLogicalFuture.from_storage(storage=DictStorage())


def test_from_storage_raises_on_unknown_task_id():
    storage = DictStorage()
    _seed_storage(storage, task_id="known", n_shots=1)

    with pytest.raises(ValueError, match="unknown"):
        GeminiLogicalFuture.from_storage(storage=storage, task_id="unknown")


def test_from_storage_raises_on_multiple_task_ids_without_explicit():
    storage = DictStorage()
    _seed_storage(storage, task_id="t1", n_shots=1)
    _seed_storage(storage, task_id="t2", n_shots=1)

    with pytest.raises(ValueError, match="More than one"):
        GeminiLogicalFuture.from_storage(storage=storage)


def test_from_storage_uses_explicit_task_id():
    storage = DictStorage()
    _seed_storage(storage, task_id="t1", n_shots=1)
    _seed_storage(storage, task_id="t2", n_shots=1)

    fut = GeminiLogicalFuture.from_storage(storage=storage, task_id="t2")
    assert fut.task_id == "t2"


# === from_task_id ===


def test_from_task_id_fetches_definition_and_stores_it(
    tasks_client, definitions_client, assert_called_with_kwargs
):
    group_id = uuid4()
    # One ID threaded through the task's ``definition_id``, the ID the client is
    # asked for, and the ID the response comes back with -- so a lookup keyed on
    # the wrong one cannot pass. ``Task.definition_id`` is a str while
    # ``TaskDefinitionResponse.id`` is a UUID, hence the str() on one side.
    definition_id = uuid4()
    # ``DefinitionsClient.get`` returns a ``TaskDefinitionResponse``, not the
    # ``TaskDefinition`` that gets stored -- the response declares ``group``,
    # which ``from_task_id`` flattens to ``group_id`` as of bloqade-core 0.6.8.
    # Returning the stored type here would not carry ``group`` at all.
    fetched_def = TaskDefinitionResponse(
        id=definition_id,
        program_language="flair.v1",
        programs=[DefinitionProgram(content="kernel")],
        subtasks=[DefinitionSubtask(program_index=0, num_shots=5)],
        created_date=datetime.now(timezone.utc),
        created_by=uuid4(),
        group=DefinitionGroupSummary(id=group_id),
    )
    task = make_task(TaskStatus.COMPLETED, definition_id=str(definition_id))
    tasks_client.get.return_value = task
    definitions_client.get.return_value = fetched_def

    storage = DictStorage()
    fut = GeminiLogicalFuture.from_task_id(task_id="task-1", storage=storage)

    assert fut.task_id == "task-1"
    assert fut.storage is storage
    assert_called_with_kwargs(tasks_client.get, times=1, id="task-1")
    assert_called_with_kwargs(definitions_client.get, times=1, id=str(definition_id))
    assert storage.task_ids() == {"task-1"}
    # What lands in storage is rebuilt, not the response object: the identifying
    # fields plus ``group_id`` flattened out of ``group``.
    assert storage.get_task_definition("task-1") == TaskDefinition(
        program_language="flair.v1",
        programs=[Program(content="kernel")],
        subtasks=[Subtask(program_index=0, num_shots=5)],
        group_id=group_id,
    )
    assert storage.get_task_creation_time("task-1") == task.created_date
