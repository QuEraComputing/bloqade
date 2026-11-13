"""End-to-end tests of the Gemini logical workflow:

    device.task / parameter_scan / batch_task
        → run_async(dry_run=False, storage=...)
        → future.result(timeout=...)
        → result.shot_results() / .arguments() / .return_values

The only things mocked are the four network boundaries: TasksClient,
ResultsClient, AuthMixin.authenticate, and time.sleep. Everything else
(kernel serialization, storage, post-processing pipeline, GeminiResult
filtering/merging) runs for real.
"""

from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
import pytest
from bloqade.core.device import (
    DictStorage,
    task as task_module,
    future as future_module,
)
from bloqade.core.device.mixins import AuthMixin
from qlam_core.plugins.tasks.api.client import TasksClient
from qlam_core.plugins.results.api.client import ResultsClient
from qlam_core.plugins.tasks.api.tasks_models import Task, TaskStatus, GroupSummary

from bloqade import squin
from bloqade.gemini import GeminiLogicalDevice, logical

# === response builders ===


def make_task_response(task_id, *, status=TaskStatus.COMPLETED, **extras):
    return Task(
        id=task_id,
        task_status=status,
        created_by=uuid4(),
        created_date=datetime.now(timezone.utc),
        # ``group`` became a required field in qlam-core 0.6.0. Tests only
        # ever parse Tasks the API hands back, so any resolvable group
        # will do; nothing here reads it.
        group=GroupSummary(id=uuid4()),
        error_reasons=[],
        **extras,
    )


def make_shot(
    *,
    shot_index,
    subtask_index=0,
    subtask_shot_index=0,
    frame_type="DETECTED",
    bitstring,
):
    return {
        "shot_index": shot_index,
        "subtask_index": subtask_index,
        "subtask_shot_index": subtask_shot_index,
        "frame_type": frame_type,
        "measurement": {"measurement_values": list(bitstring)},
    }


def make_subtask(subtask_index, *, status="COMPLETED", shots):
    """`shots` is a list of bitstrings; shot_index/subtask_shot_index are
    auto-assigned."""
    return {
        "subtask_index": subtask_index,
        "status": status,
        "shot_results": [
            make_shot(
                shot_index=subtask_index * 1000 + i,
                subtask_index=subtask_index,
                subtask_shot_index=i,
                bitstring=b,
            )
            for i, b in enumerate(shots)
        ],
    }


def make_results_page(subtasks):
    """Single 'elements' wrapper around the given subtask responses."""
    return {"elements": [{"subtasks": subtasks}]}


# === fixtures ===


@pytest.fixture(autouse=True)
def auth_no_op(mocker):
    mocker.patch.object(AuthMixin, "authenticate", autospec=True)


@pytest.fixture(autouse=True)
def fast_polling(mocker):
    mocker.patch.object(future_module.time, "sleep", autospec=True)


@pytest.fixture
def tasks_client(mocker):
    """Patches TasksClient in BOTH task.py (used for `.create` on submission)
    and future.py (used for `.get` on status polling). Both modules import
    TasksClient at the top level, so each holds its own binding that has
    to be patched independently.
    """
    client = mocker.MagicMock(spec=TasksClient)
    # `with TasksClient(ctx) as c:` — bind `c` to the same mock the fixture
    # returns, so `.create.side_effect` etc. land on the right object.
    client.__enter__.return_value = client
    mocker.patch.object(future_module, "TasksClient", return_value=client)
    mocker.patch.object(task_module, "TasksClient", return_value=client)
    return client


@pytest.fixture
def results_client(mocker):
    mock_class = mocker.patch.object(future_module, "ResultsClient", spec=ResultsClient)
    client = mock_class.return_value
    client.__enter__.return_value = client
    return client


# === kernels ===


@logical.kernel(aggressive_unroll=True)
def bell():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def ghz3():
    q = squin.qalloc(3)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.cx(q[1], q[2])
    return logical.terminal_measure(q)


# === tests ===


def test_e2e_single_kernel_task(
    tasks_client, results_client, assert_called_with_kwargs
):
    """Mirrors the workflow in debug/api_integration_squin_submission.py:
    submit a single-kernel task, wait, fetch, read back shots."""
    storage = DictStorage()

    tasks_client.create.return_value = make_task_response("task-1")
    tasks_client.get.return_value = make_task_response("task-1")
    # One page is enough: with the default subtasks_per_fetch=10, a single-
    # subtask response signals "not full → done" without a follow-up call.
    slm_shots = [(False,) * 160, (True,) * 160, (False,) * 160]
    results_client.get.return_value = make_results_page(
        [make_subtask(0, shots=slm_shots)]
    )

    device = GeminiLogicalDevice()
    task = device.task(kernel=bell, num_shots=3)
    future = task.run_async(dry_run=False, storage=storage)

    assert future is not None
    result = future.result()

    # Storage holds the task definition under the server-assigned id
    assert storage.task_ids() == {"task-1"}
    # Submission was a real TaskCreationRequest, polling used the returned id
    tasks_client.create.assert_called_once()
    assert_called_with_kwargs(tasks_client.get, id="task-1")

    shot_results = result.shot_results()
    assert len(shot_results) == 1  # one subtask
    np.testing.assert_array_equal(
        np.sort(shot_results[0], axis=0),
        np.sort(np.asarray(slm_shots), axis=0),
    )

    # return_values runs the real SLM mapping and post-processing pipeline
    # against the stored logical kernel JSON.
    return_values = result.return_values
    assert len(return_values) == 1
    assert len(return_values[0]) == 3


def test_e2e_parameter_scan(tasks_client, results_client):
    """Parameter scan: one program, N subtasks (one per parameter set).
    Each subtask gets its own shots; arguments survive the round trip."""
    storage = DictStorage()

    tasks_client.create.return_value = make_task_response("task-scan")
    tasks_client.get.return_value = make_task_response("task-scan")
    results_client.get.return_value = make_results_page(
        [
            make_subtask(0, shots=[(False, False), (True, True)]),
            make_subtask(1, shots=[(True, False), (False, True)]),
            make_subtask(2, shots=[(False, False), (False, False)]),
        ]
    )

    device = GeminiLogicalDevice()
    task = device.parameter_scan(
        bell,
        arguments=[{"x": 0.1}, {"x": 0.5}, {"x": 0.9}],
        num_shots=2,
    )
    future = task.run_async(dry_run=False, storage=storage)
    result = future.result()

    # Three subtasks, one per parameter set, all sharing program_index=0
    subtasks = storage.get_subtasks()
    assert len(subtasks) == 3
    assert {s["program_index"] for s in subtasks} == {0}
    assert sorted(s["arguments"]["x"] for s in subtasks) == [0.1, 0.5, 0.9]

    shot_results = result.shot_results()
    assert len(shot_results) == 3
    # arguments() surfaces the per-subtask scan parameters in subtask order
    assert [args["x"] for args in result.arguments()] == [0.1, 0.5, 0.9]  # type: ignore


def test_e2e_batch_task(tasks_client, results_client):
    """Batch task: multiple kernels, each its own subtask with its own
    program_index. Shot widths can differ across subtasks."""
    storage = DictStorage()

    tasks_client.create.return_value = make_task_response("task-batch")
    tasks_client.get.return_value = make_task_response("task-batch")
    results_client.get.return_value = make_results_page(
        [
            make_subtask(0, shots=[(False, False)]),  # bell — 2 bits
            make_subtask(1, shots=[(True, True, True)]),  # ghz3 — 3 bits
        ]
    )

    device = GeminiLogicalDevice()
    task = device.batch_task(kernels=[bell, ghz3], num_shots=1)
    future = task.run_async(dry_run=False, storage=storage)
    result = future.result()

    # Two distinct programs in storage, one subtask each pointing at its own program
    programs = storage.get_programs(task_ids=("task-batch",))
    assert sorted(p["program_index"] for p in programs) == [0, 1]
    assert sorted(s["program_index"] for s in storage.get_subtasks()) == [0, 1]

    shot_results = result.shot_results()
    assert len(shot_results) == 2
    np.testing.assert_array_equal(shot_results[0], np.array([[False, False]]))
    np.testing.assert_array_equal(shot_results[1], np.array([[True, True, True]]))


def test_e2e_two_tasks_in_same_storage(tasks_client, results_client):
    """Two independent submissions sharing one storage backend. Each future
    scopes its result to its own task_id; storage holds both definitions
    side by side."""
    storage = DictStorage()

    tasks_client.create.side_effect = [
        make_task_response("task-a"),
        make_task_response("task-b"),
    ]
    # Status polling: respond by id so each future polls its own task
    # **_ absorbs the kwargs bloqade-core threads through to the qlam client
    # (``qpu_mode``, ...), which are not this test's concern.
    tasks_client.get.side_effect = lambda *, id, **_: make_task_response(id)
    results_client.get.side_effect = [
        # first future fetches (task-a) — one page suffices
        make_results_page([make_subtask(0, shots=[(False, False)])]),
        # second future fetches (task-b)
        make_results_page([make_subtask(0, shots=[(True, True)])]),
    ]

    device = GeminiLogicalDevice()

    task_a = device.task(kernel=bell, num_shots=1)
    fut_a = task_a.run_async(dry_run=False, storage=storage)
    result_a = fut_a.result()

    task_b = device.task(kernel=bell, num_shots=1)
    fut_b = task_b.run_async(dry_run=False, storage=storage)
    result_b = fut_b.result()

    assert storage.task_ids() == {"task-a", "task-b"}

    # Each result is scoped to its own task_id via the default ShotFilter
    np.testing.assert_array_equal(
        result_a.shot_results()[0], np.array([[False, False]])
    )
    np.testing.assert_array_equal(result_b.shot_results()[0], np.array([[True, True]]))
