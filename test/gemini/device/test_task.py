"""Tests for the GeminiTask hierarchy in src/.../device/logical/task.py.

Pure-logic tests (kernel serialization, validation, task definition assembly,
summaries) run without any mocks. Tests for run_async(dry_run=False) and
submit_task_definition mock TasksClient — task.py imports it at the module
level, so the patch site is task_module specifically.
"""

import json
from uuid import uuid4
from datetime import datetime, timezone

import pytest
from bloqade.core.device import DictStorage, task as task_module
from bloqade.core.device.future import ApiFetchOptions
from bloqade.core.device.mixins import AuthMixin
from qlam_core.plugins.tasks.api.client import TasksClient
from qlam_core.plugins.tasks.api.tasks_models import (
    Task,
    Program,
    TaskStatus,
    GroupSummary,
)

from bloqade import squin
from bloqade.gemini import GeminiLogicalFuture, logical
from bloqade.gemini.device.logical.task import (
    GeminiKernelBatchTask,
    GeminiSingleKernelTask,
    GeminiParameterScanTask,
)

# === kernels ===


@logical.kernel(aggressive_unroll=True)
def bell():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def trio():
    q = squin.qalloc(3)
    squin.h(q[0])
    return logical.terminal_measure(q)


# === fixtures ===


@pytest.fixture(autouse=True)
def auth_no_op(mocker):
    mocker.patch.object(AuthMixin, "authenticate", autospec=True)


@pytest.fixture
def tasks_client(mocker):
    """Patches TasksClient at its task.py import site. Same self-loop pattern
    as elsewhere — bind `c` in `with TasksClient(ctx) as c:` to the same mock
    we configure here, otherwise `.create` lands on a fresh sub-mock."""
    client = mocker.MagicMock(spec=TasksClient)
    client.__enter__.return_value = client
    mocker.patch.object(task_module, "TasksClient", return_value=client)
    return client


def make_task_response(task_id, **extras):
    return Task(
        id=task_id,
        task_status=TaskStatus.CREATED,
        created_by=uuid4(),
        created_date=datetime.now(timezone.utc),
        # ``group`` became a required field in qlam-core 0.6.0. Tests only
        # ever parse Tasks the API hands back, so any resolvable group
        # will do; nothing here reads it.
        group=GroupSummary(id=uuid4()),
        error_reasons=[],
        **extras,
    )


# === serialize_kernel ===


def test_serialize_kernel_squin_round_trips_via_decode():
    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    serialized = task.serialize_kernel(bell)
    # Anything decode_json accepts is valid JSON for the encoder.
    decoded = logical.kernel.decode_json(serialized)
    assert decoded is not None


# === validate_arguments ===


def test_validate_arguments_passes_for_matching_lengths():
    task = GeminiKernelBatchTask(
        kernels=[bell, bell],
        arguments=[{"x": 0.1}, {"x": 0.5}],
        metadata=[{"a": 1}, {"a": 2}],
        num_shots=1,
    )
    task.validate_arguments()  # no raise


def test_validate_arguments_raises_for_argument_length_mismatch():
    task = GeminiKernelBatchTask(
        kernels=[bell, bell, bell],
        arguments=[{"x": 1}],  # 1 != 3
        num_shots=1,
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        task.validate_arguments()


def test_validate_arguments_raises_for_metadata_length_mismatch():
    task = GeminiKernelBatchTask(
        kernels=[bell, bell],
        metadata=[{"a": 1}],  # 1 != 2
        num_shots=1,
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        task.validate_arguments()


def test_validate_arguments_passes_when_arguments_and_metadata_none():
    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    task.validate_arguments()  # no raise


# === program_index_for_subtask defaults ===


def test_program_index_for_subtask_default_is_identity():
    task = GeminiKernelBatchTask(kernels=[bell, bell, bell], num_shots=1)
    assert task.program_index_for_subtask(0) == 0
    assert task.program_index_for_subtask(2) == 2


# === GeminiSingleKernelTask ===


def test_single_kernel_task_num_subtasks_is_one():
    task = GeminiSingleKernelTask(kernel=bell, num_shots=5)
    assert task.num_subtasks == 1


def test_single_kernel_task_programs_returns_one_program():
    task = GeminiSingleKernelTask(kernel=bell, num_shots=5)
    programs = task.programs()
    assert len(programs) == 1
    assert isinstance(programs[0], Program)


def test_single_kernel_task_create_task_definition_int_num_shots():
    task = GeminiSingleKernelTask(kernel=bell, num_shots=10)
    task_def = task.create_task_definition()

    expected_version = f"{task.program_language}.v{task.program_language_version}"
    assert task_def.program_language == expected_version
    assert len(task_def.programs) == 1
    assert len(task_def.subtasks) == 1
    subtask = task_def.subtasks[0]
    assert subtask.num_shots == 10
    assert subtask.program_index == 0
    assert subtask.arguments is None
    assert subtask.subtask_metadata is None


def test_single_kernel_task_create_task_definition_with_arguments_and_metadata():
    task = GeminiSingleKernelTask(
        kernel=bell,
        num_shots=10,
        arguments={"theta": 0.5},
        metadata={"experiment": "alpha"},
    )
    task_def = task.create_task_definition()

    subtask = task_def.subtasks[0]
    assert subtask.arguments == {"theta": 0.5}
    # Metadata is JSON-serialized into TaskMetadata.user_metadata.
    assert subtask.subtask_metadata is not None
    user_metadata = subtask.subtask_metadata.user_metadata
    assert user_metadata is not None
    assert json.loads(user_metadata) == {"experiment": "alpha"}


def test_single_kernel_task_summary_mentions_kernel_and_shots():
    task = GeminiSingleKernelTask(kernel=bell, num_shots=42)
    summary = task.summary()
    assert "DRY RUN" in summary
    assert "42 shots" in summary
    assert str(bell.sym_name) in summary


# === GeminiKernelBatchTask ===


def test_batch_task_num_subtasks_matches_kernel_count():
    task = GeminiKernelBatchTask(kernels=[bell, trio, bell], num_shots=1)
    assert task.num_subtasks == 3


def test_batch_task_programs_one_per_kernel():
    task = GeminiKernelBatchTask(kernels=[bell, trio], num_shots=1)
    assert len(task.programs()) == 2


def test_batch_task_create_task_definition_int_num_shots_broadcasts():
    task = GeminiKernelBatchTask(kernels=[bell, trio], num_shots=10)
    task_def = task.create_task_definition()
    assert [s.num_shots for s in task_def.subtasks] == [10, 10]


def test_batch_task_create_task_definition_list_num_shots_used_as_is():
    task = GeminiKernelBatchTask(kernels=[bell, trio], num_shots=[5, 7])
    task_def = task.create_task_definition()
    assert [s.num_shots for s in task_def.subtasks] == [5, 7]


def test_batch_task_create_task_definition_program_index_per_kernel():
    """Default program_index_for_subtask(i) == i — each subtask points at
    its own program slot."""
    task = GeminiKernelBatchTask(kernels=[bell, trio], num_shots=1)
    task_def = task.create_task_definition()
    assert [s.program_index for s in task_def.subtasks] == [0, 1]


def test_batch_task_summary_mentions_each_kernel():
    task = GeminiKernelBatchTask(kernels=[bell, trio], num_shots=1)
    summary = task.summary()
    assert str(bell.sym_name) in summary
    assert str(trio.sym_name) in summary


# === GeminiParameterScanTask ===


def test_parameter_scan_post_init_raises_when_arguments_none():
    with pytest.raises(TypeError, match="arguments"):
        GeminiParameterScanTask(kernel=bell, num_shots=1)  # type: ignore


def test_parameter_scan_num_subtasks_matches_argument_count():
    task = GeminiParameterScanTask(
        kernel=bell,
        arguments=[{"x": 0.1}, {"x": 0.5}, {"x": 0.9}],
        num_shots=1,
    )
    assert task.num_subtasks == 3


def test_parameter_scan_programs_returns_single_program():
    task = GeminiParameterScanTask(
        kernel=bell,
        arguments=[{"x": 0.1}, {"x": 0.5}],
        num_shots=1,
    )
    assert len(task.programs()) == 1


def test_parameter_scan_program_index_for_subtask_always_zero():
    task = GeminiParameterScanTask(
        kernel=bell,
        arguments=[{"x": 0.1}, {"x": 0.5}, {"x": 0.9}],
        num_shots=1,
    )
    for i in range(3):
        assert task.program_index_for_subtask(i) == 0


def test_parameter_scan_create_task_definition_one_subtask_per_argument():
    task = GeminiParameterScanTask(
        kernel=bell,
        arguments=[{"x": 0.1}, {"x": 0.5}, {"x": 0.9}],
        num_shots=2,
    )
    task_def = task.create_task_definition()

    assert len(task_def.programs) == 1  # single shared program
    assert len(task_def.subtasks) == 3
    assert all(s.program_index == 0 for s in task_def.subtasks)
    assert [s.arguments for s in task_def.subtasks] == [
        {"x": 0.1},
        {"x": 0.5},
        {"x": 0.9},
    ]
    # Int num_shots broadcasts across all subtasks.
    assert all(s.num_shots == 2 for s in task_def.subtasks)


def test_parameter_scan_summary_mentions_subtask_count():
    task = GeminiParameterScanTask(
        kernel=bell,
        arguments=[{"x": 0.1}, {"x": 0.5}],
        num_shots=1,
    )
    summary = task.summary()
    assert "DRY RUN" in summary
    assert "2 subtasks" in summary


# === run_async dry_run=True ===


def test_run_async_dry_run_returns_none(capsys):
    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    result = task.run_async(dry_run=True, storage=DictStorage())
    assert result is None
    assert "DRY RUN" in capsys.readouterr().out


def test_run_async_dry_run_validates_arguments_first():
    task = GeminiKernelBatchTask(
        kernels=[bell, bell],
        arguments=[{"x": 1}],
        num_shots=1,
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        task.run_async(dry_run=True, storage=DictStorage())


def test_run_async_dry_run_does_not_call_api(tasks_client):
    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    task.run_async(dry_run=True, storage=DictStorage())
    tasks_client.create.assert_not_called()


# === submit_task_definition / run_async dry_run=False ===


def test_submit_task_definition_creates_task_via_client(tasks_client):
    tasks_client.create.return_value = make_task_response("task-1")

    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    task_def = task.create_task_definition()
    storage = DictStorage()

    future = task.submit_task_definition(task_definition=task_def, storage=storage)

    assert isinstance(future, GeminiLogicalFuture)
    assert future.task_id == "task-1"
    assert future.storage is storage
    tasks_client.create.assert_called_once()
    # body= is a TaskCreationRequest wrapping the task definition.
    body = tasks_client.create.call_args.kwargs["body"]
    assert body.root == task_def


def test_submit_task_definition_stores_definition_in_storage(tasks_client):
    created_task = make_task_response("task-1")
    tasks_client.create.return_value = created_task

    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    task_def = task.create_task_definition()
    storage = DictStorage()
    task.submit_task_definition(task_definition=task_def, storage=storage)

    assert storage.task_ids() == {"task-1"}
    assert storage.get_task_definition("task-1") == task_def
    assert storage.get_task_creation_time("task-1") == created_task.created_date


def test_submit_task_definition_raises_when_task_id_missing(tasks_client):
    tasks_client.create.return_value = make_task_response(task_id=None)

    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    task_def = task.create_task_definition()

    with pytest.raises(ValueError, match="Couldn't get id"):
        task.submit_task_definition(task_definition=task_def, storage=DictStorage())


def test_submit_task_definition_propagates_context_name(tasks_client):
    tasks_client.create.return_value = make_task_response("task-1")

    task = GeminiSingleKernelTask(kernel=bell, num_shots=1, context_name="my-context")
    task_def = task.create_task_definition()
    future = task.submit_task_definition(
        task_definition=task_def, storage=DictStorage()
    )

    assert future.context_name == "my-context"


def test_submit_task_definition_propagates_fetch_options(tasks_client):
    tasks_client.create.return_value = make_task_response("task-1")

    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    task_def = task.create_task_definition()
    fetch_options = ApiFetchOptions(subtasks_per_fetch=5, shots_per_fetch=20)
    future = task.submit_task_definition(
        task_definition=task_def,
        storage=DictStorage(),
        fetch_options=fetch_options,
    )

    assert future.fetch_options is fetch_options


def test_run_async_real_run_returns_future(tasks_client):
    tasks_client.create.return_value = make_task_response("task-1")

    task = GeminiSingleKernelTask(kernel=bell, num_shots=1)
    storage = DictStorage()
    future = task.run_async(dry_run=False, storage=storage)

    assert isinstance(future, GeminiLogicalFuture)
    assert future.task_id == "task-1"
    assert storage.task_ids() == {"task-1"}


def test_run_async_real_run_validates_before_submission(tasks_client):
    """If validation fails, the API is never called."""
    task = GeminiKernelBatchTask(
        kernels=[bell, bell],
        arguments=[{"x": 1}],
        num_shots=1,
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        task.run_async(dry_run=False, storage=DictStorage())
    tasks_client.create.assert_not_called()
