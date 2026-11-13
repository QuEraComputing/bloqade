"""TaskBuilder-to-local-simulator integration with the existing result API."""

import sys
import json
import sqlite3
from uuid import UUID, uuid4
from inspect import Parameter, signature
from threading import Event, get_ident
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

import numpy as np
import pytest
from kirin import ir
from kirin.dialects import ilist
from kirin.validation import ValidationSuite
from bloqade.core.device import ShotFilter, DictStorage
from kirin.serialization import JSONSerializer
from bloqade.lanes.transform import LogicalPipeline
from bloqade.core.device.future import ApiFetchOptions
from bloqade.core.device.task_builder import TaskBuilder, TaskFinalizeError
from bloqade.core.device.local_storage import SQLiteStorage

from bloqade import squin
from bloqade.gemini import (
    GeminiLogicalDevice,
    GeminiLogicalResult,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorDevice,
    logical,
)
from bloqade.gemini.device import (
    logical as device_logical,
    simulator_backend as backend_module,
)
from bloqade.gemini.device.logical import simulator_device as device_module
from bloqade.gemini.device.simulator_backend import (
    BackendSample,
    PPVMSimulatorBackend,
    CliffTSimulatorBackend,
)


@logical.kernel
def pinned():
    q = logical.qalloc_at(ilist.IList([8, 1]))
    squin.x(q[0])
    return logical.terminal_measure(q)


@logical.kernel
def bell():
    q = logical.qalloc_at(ilist.IList([4, 5]))
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return logical.terminal_measure(q)


@logical.kernel
def all_slots():
    q = logical.qalloc_at(ilist.IList([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
    return logical.terminal_measure(q)


@logical.kernel
def annotated():
    q = logical.qalloc_at(ilist.IList([9]))
    measurements = logical.terminal_measure(q)
    squin.set_detector(measurements[0], [0])
    squin.set_observable(measurements[0])
    return measurements[0][0], measurements[0]


@logical.kernel(verify=False)
def parameterized(theta: float):
    q = squin.qalloc(1)
    squin.rx(theta, q[0])
    return logical.terminal_measure(q)


def builder_for(kernel: ir.Method = pinned, shots: int = 3) -> TaskBuilder:
    builder = TaskBuilder()
    builder.add_subtask(kernel, shots, metadata={"name": kernel.sym_name})
    return builder


def test_public_exports_are_the_same_class():
    assert device_logical.GeminiLogicalSimulatorDevice is GeminiLogicalSimulatorDevice


def test_noise_is_enabled_by_default():
    assert GeminiLogicalSimulatorDevice().with_noise is True


def test_ppvm_is_the_default_backend():
    assert GeminiLogicalSimulatorDevice().backend == "ppvm"


def test_batch_shots_defaults_to_none():
    assert GeminiLogicalSimulatorDevice().batch_shots is None


@pytest.mark.parametrize(
    "batch_shots,subtask_shots,expected_batches",
    [
        (None, [257, 3], [257, 3]),
        (2, [5, 3], [2, 2, 1, 2, 1]),
        (1000, [5, 3], [5, 3]),
    ],
)
def test_sampling_batches_follow_each_subtask_shot_count(
    monkeypatch, batch_shots, subtask_shots, expected_batches
):
    sample = PPVMSimulatorBackend.sample
    batches = []

    def record(self, kernel, *, shots):
        batches.append(shots)
        return sample(self, kernel, shots=shots)

    monkeypatch.setattr(PPVMSimulatorBackend, "sample", record)
    builder = TaskBuilder()
    for shots in subtask_shots:
        builder.add_subtask(pinned, shots)
    result = (
        GeminiLogicalSimulatorDevice(seed=12, batch_shots=batch_shots)
        .run_async(builder, dry_run=False)
        .result(60)
    )
    assert batches == expected_batches
    assert [shots.shape for shots in result.shot_results()] == [
        (count, 160) for count in subtask_shots
    ]
    rows = list(result.storage.get_shots(shot_filter=result.shot_filter))
    assert [row.shot_index for row in rows] == list(range(sum(subtask_shots)))
    assert [(row.subtask_index, row.subtask_shot_index) for row in rows] == [
        (subtask, shot)
        for subtask, count in enumerate(subtask_shots)
        for shot in range(count)
    ]


def test_default_backend_reports_missing_ppvm_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "ppvm", None)
    future = GeminiLogicalSimulatorDevice().run_async(
        builder_for(shots=1), dry_run=False
    )
    with pytest.raises(
        ImportError, match=r"bloqade-ppvm.*required dependency of `bloqade`"
    ):
        future.result(60)


def test_default_backend_initializes_ppvm_on_the_caller_thread(monkeypatch):
    ppvm = pytest.importorskip("ppvm")
    importing_threads = []

    def record_import():
        importing_threads.append(get_ident())
        return ppvm

    monkeypatch.setattr(device_module, "_ppvm", record_import)
    monkeypatch.setattr(backend_module, "_ppvm", record_import)
    future = GeminiLogicalSimulatorDevice().run_async(
        builder_for(shots=1), dry_run=False
    )
    assert future.result(60).shot_results()[0].shape == (1, 160)
    assert importing_threads[0] == get_ident()


def test_finalize_context_defaults_match_hardware_device():
    device = GeminiLogicalSimulatorDevice()
    context = device._finalize_context()
    hardware = GeminiLogicalDevice()._finalize_context()
    assert context.program_language == hardware.program_language
    assert context.language_version == hardware.language_version
    assert type(context.kernel_serializer) is type(hardware.kernel_serializer)
    assert context.dialect_group is hardware.dialect_group
    assert isinstance(context.validation_suite, ValidationSuite)
    assert isinstance(hardware.validation_suite, ValidationSuite)
    assert context.validation_suite.passes == hardware.validation_suite.passes
    assert context.validation_suite is not hardware.validation_suite


def test_create_task_definition_has_no_submission_side_effects(monkeypatch, capsys):
    device = GeminiLogicalSimulatorDevice(seed=12)
    builder = builder_for(shots=2)
    state = json.dumps(device._seed_rng.bit_generator.state)

    def forbidden(*args, **kwargs):
        pytest.fail("creating a definition must not start a worker")

    monkeypatch.setattr(device._executor, "submit", forbidden)
    definition = device.create_task_definition(builder)
    assert definition.program_language == f"squin.v{device.language_version}"
    assert definition.subtasks[0].num_shots == 2
    assert definition.profile_id is None
    assert definition.group_id is None
    assert json.dumps(device._seed_rng.bit_generator.state) == state
    assert capsys.readouterr().out == ""
    assert device.create_task_definition(builder) == definition
    builder.add_subtask(bell, 1)
    assert len(device.create_task_definition(builder).subtasks) == 2
    assert len(definition.subtasks) == 1


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_async_delegates_to_public_definition_methods(monkeypatch, dry_run, capsys):
    device = GeminiLogicalSimulatorDevice()
    builder = builder_for(shots=2)
    definition = builder._finalize(device._finalize_context())
    storage = DictStorage()
    options = ApiFetchOptions(shots_per_fetch=4)
    profile = uuid4()
    future = Future()
    created, submitted = [], []

    def create(task_builder):
        created.append(task_builder)
        return definition

    def submit(**kwargs):
        submitted.append(kwargs)
        return future

    monkeypatch.setattr(device, "create_task_definition", create)
    monkeypatch.setattr(device, "submit_task_definition", submit)
    returned = device.run_async(
        builder,
        dry_run=dry_run,
        group="ignored",
        profile_id=profile,
        storage=storage,
        fetch_options=options,
    )
    assert created == [builder]
    if dry_run:
        assert returned is None
        assert submitted == []
        assert "pinned" in capsys.readouterr().out
    else:
        assert returned is future
        assert submitted == [
            {
                "task_definition": definition,
                "storage": storage,
                "fetch_options": options,
                "group": "ignored",
                "profile_id": profile,
            }
        ]


@pytest.mark.parametrize("sqlite", [False, True])
def test_submit_task_definition_uses_existing_result_and_metadata(tmp_path, sqlite):
    device = GeminiLogicalSimulatorDevice(seed=12)
    definition = device.create_task_definition(builder_for(shots=2))
    profile = uuid4()
    storage = (
        SQLiteStorage(str(tmp_path / "direct.sqlite")) if sqlite else DictStorage()
    )
    try:
        future = device.submit_task_definition(
            task_definition=definition,
            storage=storage,
            profile_id=str(profile),
            group="ignored",
            fetch_options=ApiFetchOptions(shots_per_fetch=1),
        )
        assert type(future) is Future
        result = future.result(60)
        assert result.storage is storage
        assert result.shot_results()[0].shape == (2, 160)
        task_id = next(iter(result.task_ids()))
        assert storage.get_profile_id(task_id) == profile
        assert definition.profile_id is None
    finally:
        if isinstance(storage, SQLiteStorage):
            storage.close()


def test_submit_task_definition_snapshots_input(monkeypatch):
    device = GeminiLogicalSimulatorDevice()
    definition = device.create_task_definition(builder_for(shots=2))
    entered, proceed = Event(), Event()
    run_task = device_module._run_task

    def blocked(*args):
        entered.set()
        assert proceed.wait(30)
        return run_task(*args)

    monkeypatch.setattr(device_module, "_run_task", blocked)
    future = device.submit_task_definition(task_definition=definition)
    try:
        assert entered.wait(30)
        definition.subtasks[0].num_shots = 5
    finally:
        proceed.set()
    assert future.result(60).shot_results()[0].shape == (2, 160)


@pytest.mark.parametrize("kind", ["empty", "index", "arguments", "shots"])
def test_submit_task_definition_rejects_invalid_definition_before_worker(
    monkeypatch, kind
):
    device = GeminiLogicalSimulatorDevice(seed=12)
    definition = device.create_task_definition(builder_for(shots=2))
    if kind == "empty":
        definition = definition.model_copy(update={"subtasks": []})
    else:
        updates = {
            "index": {"program_index": 99},
            "arguments": {"arguments": {"theta": 0.1}},
            "shots": {"num_shots": 0},
        }
        definition = definition.model_copy(
            update={
                "subtasks": [definition.subtasks[0].model_copy(update=updates[kind])]
            }
        )
    state = json.dumps(device._seed_rng.bit_generator.state)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid definitions must not start a worker")

    monkeypatch.setattr(device._executor, "submit", forbidden)
    with pytest.raises(ValueError):
        device.submit_task_definition(task_definition=definition)
    assert json.dumps(device._seed_rng.bit_generator.state) == state


def test_custom_finalize_context_is_used_by_submission(monkeypatch):
    serializer = JSONSerializer()
    suite = ValidationSuite([])
    device = GeminiLogicalSimulatorDevice(
        program_language="custom-squin",
        language_version="1.2.3",
        kernel_serializer=serializer,
        dialect_group=logical.kernel,
        validation_suite=suite,
    )
    context = device._finalize_context()
    assert context.program_language == "custom-squin"
    assert context.language_version == "1.2.3"
    assert context.kernel_serializer is serializer
    assert context.dialect_group is logical.kernel
    assert context.validation_suite is suite
    contexts = []
    finalize = TaskBuilder._finalize

    def record(builder, ctx):
        contexts.append(ctx)
        return finalize(builder, ctx)

    def forbidden(*args, **kwargs):
        pytest.fail("simulator must use its own finalization context")

    monkeypatch.setattr(TaskBuilder, "_finalize", record)
    monkeypatch.setattr(GeminiLogicalDevice, "_finalize_context", forbidden)
    result = device.run_async(builder_for(shots=1), dry_run=False).result(60)
    assert contexts == [context]
    task_id = next(iter(result.task_ids()))
    assert result.storage.get_program_language(task_id) == "custom-squin.v1.2.3"
    program = next(iter(result.storage.get_programs(task_ids=(task_id,))))
    assert json.loads(program["content"])["version"] == "1.2.3"


@pytest.mark.parametrize("as_string", [False, True])
def test_submission_uses_supplied_storage_and_profile_metadata(as_string):
    storage = DictStorage()
    profile = uuid4()
    profile_arg = str(profile) if as_string else profile
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(
            builder_for(shots=1), dry_run=False, storage=storage, profile_id=profile_arg
        )
        .result(60)
    )
    assert result.storage is storage
    task_id = next(iter(result.task_ids()))
    assert storage.get_profile_id(task_id) == profile
    assert result.shot_results()[0].shape == (1, 160)


def test_definition_profile_takes_precedence_over_submission_profile(monkeypatch):
    profile = uuid4()
    finalize = TaskBuilder._finalize

    def with_profile(builder, ctx):
        return finalize(builder, ctx).model_copy(update={"profile_id": profile})

    monkeypatch.setattr(TaskBuilder, "_finalize", with_profile)
    result = (
        GeminiLogicalSimulatorDevice()
        .run_async(builder_for(shots=1), dry_run=False, profile_id=uuid4())
        .result(60)
    )
    task_id = next(iter(result.task_ids()))
    assert result.storage.get_profile_id(task_id) == profile


def test_remote_submission_options_are_accepted_and_ignored(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local simulation must not resolve groups or authenticate")

    monkeypatch.setattr(GeminiLogicalDevice, "_resolve_group_id", forbidden)
    monkeypatch.setattr(GeminiLogicalDevice, "authenticate", forbidden)
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(
            builder_for(shots=1),
            dry_run=False,
            group="no-local-group-resolution",
            fetch_options=ApiFetchOptions(shots_per_fetch=1, poll_interval_initial=999),
        )
        .result(60)
    )
    task_id = next(iter(result.task_ids()))
    assert result.storage.get_task_group_id(task_id) is None
    assert result.shot_results()[0].shape == (1, 160)


def test_device_signature_requires_explicit_dry_run():
    assert (
        signature(GeminiLogicalSimulatorDevice.run_async).parameters["dry_run"].default
        is Parameter.empty
    )


def test_dry_run_with_submission_options_does_not_write_storage(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not start a worker or write storage")

    device = GeminiLogicalSimulatorDevice(seed=12)
    storage = DictStorage()
    state = json.dumps(device._seed_rng.bit_generator.state)
    monkeypatch.setattr(device._executor, "submit", forbidden)
    monkeypatch.setattr(storage, "add_task_definition", forbidden)
    monkeypatch.setattr(storage, "add_shots", forbidden)
    assert (
        device.run_async(
            builder_for(),
            dry_run=True,
            group="ignored",
            profile_id=UUID(int=1),
            storage=storage,
            fetch_options=ApiFetchOptions(shots_per_fetch=1),
        )
        is None
    )
    assert not storage.task_ids()
    assert json.dumps(device._seed_rng.bit_generator.state) == state


@pytest.mark.parametrize("db_file", [":memory:", ""])
def test_private_sqlite_storage_rejected_before_starting_worker(monkeypatch, db_file):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported storage must be rejected before starting a worker")

    device = GeminiLogicalSimulatorDevice(seed=12)
    state = json.dumps(device._seed_rng.bit_generator.state)
    monkeypatch.setattr(device._executor, "submit", forbidden)
    with SQLiteStorage(db_file) as storage:
        with pytest.raises(ValueError, match="file-backed"):
            device.run_async(builder_for(shots=1), dry_run=False, storage=storage)
        assert not storage.task_ids()
    assert json.dumps(device._seed_rng.bit_generator.state) == state


def test_sqlite_results_use_caller_storage_and_close_worker_connection(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("worker must not write through the caller's SQLite connection")

    caller_thread = get_ident()
    closed = []
    close = SQLiteStorage.close

    def record_close(self):
        close(self)
        closed.append(get_ident())
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            self.conn.execute("SELECT 1")

    monkeypatch.setattr(SQLiteStorage, "close", record_close)
    path = str(tmp_path / "simulation.sqlite")
    with SQLiteStorage(path) as storage:
        monkeypatch.setattr(storage, "add_task_definition", forbidden)
        monkeypatch.setattr(storage, "add_shots", forbidden)
        monkeypatch.setattr(storage, "update_subtasks_completed_date", forbidden)
        future = GeminiLogicalSimulatorDevice(
            seed=12, batch_shots=2, with_noise=False
        ).run_async(builder_for(shots=3), dry_run=False, storage=storage)
        assert type(future) is Future
        result = future.result(60)
        assert result is future.result()
        assert result.storage is storage
        assert len(closed) == 1 and closed[0] != caller_thread
        assert result.shot_results()[0].shape == (3, 160)
        assert np.all(
            np.logical_xor.reduce(np.asarray(result.return_values[0]), axis=2)
            == [True, False]
        )
        assert all(
            subtask["completed_date"] is not None for subtask in result.full_subtasks()
        )
        task_id = next(iter(result.task_ids()))
    with SQLiteStorage(path) as reopened:
        assert reopened.task_ids() == {task_id}
        assert len(list(reopened.get_shots())) == 6


def test_queued_simulations_write_to_same_sqlite_database(tmp_path):
    with SQLiteStorage(str(tmp_path / "concurrent.sqlite")) as storage:
        device = GeminiLogicalSimulatorDevice(seed=12, batch_shots=2)
        first = device.run_async(builder_for(shots=5), dry_run=False, storage=storage)
        second = device.run_async(builder_for(shots=5), dry_run=False, storage=storage)
        results = [first.result(60), second.result(60)]
        assert all(result.storage is storage for result in results)
        assert len(storage.task_ids()) == 2
        assert len(list(storage.get_shots())) == 20
        assert all(result.shot_results()[0].shape == (5, 160) for result in results)


def test_sqlite_worker_factory_preserves_path_timeout_and_wal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with SQLiteStorage("relative.sqlite") as storage:
        storage.conn.execute("PRAGMA busy_timeout=2000")
        assert storage.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        factory = device_module._sqlite_connection_factory(storage)
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        with factory() as worker:
            assert worker.conn is not storage.conn
            assert worker.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2000
            assert worker.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert worker.conn.execute("PRAGMA database_list").fetchone()[2] == str(
                tmp_path / "relative.sqlite"
            )
        assert not (other / "relative.sqlite").exists()


def test_sqlite_worker_factory_closes_connection_on_constructor_failure(
    tmp_path, monkeypatch
):
    with SQLiteStorage(str(tmp_path / "invalid-schema.sqlite")) as storage:
        factory = device_module._sqlite_connection_factory(storage)
        storage.conn.execute("UPDATE bloqade_schema SET version_number='999.0.0'")
        storage.conn.commit()
        connect = sqlite3.connect
        connections = []

        def record(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connections.append(connection)
            return connection

        monkeypatch.setattr(sqlite3, "connect", record)
        with pytest.raises(ValueError, match="Schema version mismatch"):
            factory()
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")


def test_sqlite_worker_waits_for_writer_to_commit(tmp_path, monkeypatch):
    entered, proceed = Event(), Event()
    sample = PPVMSimulatorBackend.sample

    def blocked(self, kernel, *, shots):
        result = sample(self, kernel, shots=shots)
        entered.set()
        assert proceed.wait(30)
        return result

    monkeypatch.setattr(PPVMSimulatorBackend, "sample", blocked)
    with SQLiteStorage(str(tmp_path / "contention.sqlite")) as storage:
        future = GeminiLogicalSimulatorDevice().run_async(
            builder_for(shots=1), dry_run=False, storage=storage
        )
        try:
            assert entered.wait(30)
            storage.conn.execute("BEGIN IMMEDIATE")
            proceed.set()
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.05)
        finally:
            storage.conn.rollback()
            proceed.set()
        assert future.result(60).shot_results()[0].shape == (1, 160)


def test_sqlite_worker_connection_closes_on_backend_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("sample failed")

    closed = []
    close = SQLiteStorage.close

    def record_close(self):
        close(self)
        closed.append(get_ident())

    monkeypatch.setattr(SQLiteStorage, "close", record_close)
    monkeypatch.setattr(PPVMSimulatorBackend, "sample", fail)
    with SQLiteStorage(str(tmp_path / "failure.sqlite")) as storage:
        future = GeminiLogicalSimulatorDevice().run_async(
            builder_for(shots=1), dry_run=False, storage=storage
        )
        with pytest.raises(RuntimeError, match="sample failed"):
            future.result(60)
        assert len(closed) == 1 and closed[0] != get_ident()
        storage.conn.execute("BEGIN IMMEDIATE")
        storage.conn.rollback()
        assert storage.conn.execute("SELECT 1").fetchone()[0] == 1


def test_dry_run_accepts_sqlite_storage_without_accessing_it(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not access storage or start a worker")

    device = GeminiLogicalSimulatorDevice()
    monkeypatch.setattr(device._executor, "submit", forbidden)
    with SQLiteStorage(":memory:") as storage:
        monkeypatch.setattr(storage, "add_task_definition", forbidden)
        monkeypatch.setattr(storage, "add_shots", forbidden)
        assert device.run_async(builder_for(), dry_run=True, storage=storage) is None
        assert not storage.task_ids()


@pytest.mark.parametrize("backend", ["tsim", "ppvm", "clifft"])
def test_real_backend_result_frames_and_postprocessing(backend):
    pytest.importorskip("bloqade.tsim")
    if backend == "ppvm":
        pytest.importorskip("ppvm")
    if backend == "clifft":
        pytest.importorskip("clifft")
    future = GeminiLogicalSimulatorDevice(
        backend=backend, seed=1234, with_noise=False
    ).run_async(builder_for(), dry_run=False)
    assert type(future) is Future
    result = future.result(timeout=60)
    assert type(result) is GeminiLogicalResult
    assert [shots.shape for shots in result.shot_results()] == [(3, 160)]
    values = np.asarray(result.return_values[0], dtype=bool)
    assert values.shape == (3, 2, 7)
    assert np.all(np.logical_xor.reduce(values, axis=2) == [True, False])
    assert np.asarray(result.measurements[0]).shape == (3, 14)
    assert np.all(result.filling_at_start[0])
    assert result.postselect_on_fully_filled().shot_results()[0].shape == (3, 160)
    sorted_shots = list(
        result.storage.get_shots(
            shot_filter=ShotFilter(
                task_ids=tuple(result.task_ids()), frame_type="SORTED"
            )
        )
    )
    expected = set(range(128, 135)) | set(range(16, 23))
    assert all(set(np.flatnonzero(shot.bitstring)) == expected for shot in sorted_shots)
    assert all(shot.bitstring.dtype == bool for shot in sorted_shots)
    assert all(
        subtask["completed_date"] is not None for subtask in result.full_subtasks()
    )


def test_detectors_observables_and_structured_returns_use_existing_result():
    result = (
        GeminiLogicalSimulatorDevice(seed=123)
        .run_async(builder_for(annotated, 3), dry_run=False)
        .result(60)
    )
    raw = np.asarray(result.measurements[0])
    parity = np.logical_xor.reduce(raw, axis=1)
    np.testing.assert_array_equal(np.asarray(result.detectors[0])[:, 0], parity)
    np.testing.assert_array_equal(np.asarray(result.observables[0])[:, 0], parity)
    for returned, measurements in zip(result.return_values[0], raw):
        assert returned[0] == measurements[0]
        np.testing.assert_array_equal(returned[1], measurements)


def test_multiple_programs_repeated_subtasks_and_metadata(monkeypatch):
    calls = []
    emit = LogicalPipeline.emit

    def record(self, kernel, no_raise=True):
        calls.append(kernel.sym_name)
        return emit(self, kernel, no_raise=no_raise)

    monkeypatch.setattr(LogicalPipeline, "emit", record)
    builder = builder_for(shots=3)
    builder.add_subtask(bell, 5, metadata={"name": "bell"})
    builder.add_subtask(pinned, 2, metadata={"name": "repeat"})
    result = (
        GeminiLogicalSimulatorDevice(seed=12, batch_shots=2, with_noise=False)
        .run_async(builder, dry_run=False)
        .result(60)
    )
    assert calls == ["pinned", "bell"]
    assert [shots.shape for shots in result.shot_results()] == [
        (3, 160),
        (5, 160),
        (2, 160),
    ]
    assert [row["program_index"] for row in result.full_subtasks()] == [0, 1, 0]
    assert [
        json.loads(row["metadata"]["user_metadata"])["name"]
        for row in result.full_subtasks()
    ] == ["pinned", "bell", "repeat"]
    rows = list(result.storage.get_shots(shot_filter=result.shot_filter))
    assert [row.shot_index for row in rows] == list(range(10))
    assert [row.subtask_shot_index for row in rows] == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]
    bell_values = np.logical_xor.reduce(np.asarray(result.return_values[1]), axis=2)
    assert np.all(bell_values[:, 0] == bell_values[:, 1])


def test_local_execution_verifies_deserialized_kernel(monkeypatch):
    builder = builder_for(shots=1)
    run_pass = logical.kernel.run_pass
    assert run_pass is not None
    verification = []

    def record(kernel, **kwargs):
        verification.append(kwargs.get("verify", True))
        return run_pass(kernel, **kwargs)

    monkeypatch.setattr(logical.kernel, "run_pass", record)
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(builder, dry_run=False)
        .result(60)
    )
    assert verification == [True]
    assert result.shot_results()[0].shape == (1, 160)


def test_local_execution_uses_pipelines_without_simulator_tasks(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local execution must not construct simulator tasks")

    monkeypatch.setattr(GeminiLogicalSimulator, "task", forbidden)
    result = (
        GeminiLogicalSimulatorDevice(seed=12, with_noise=False)
        .run_async(builder_for(shots=2), dry_run=False)
        .result(60)
    )
    assert result.shot_results()[0].shape == (2, 160)
    parity = np.logical_xor.reduce(np.asarray(result.return_values[0]), axis=2)
    assert np.all(parity == [True, False])


@pytest.mark.parametrize("with_noise", [None, False, True])
def test_direct_pipeline_preserves_physical_kernel(with_noise, monkeypatch):
    simulator = GeminiLogicalSimulator()
    reference_task = simulator.task(pinned)
    expected = (
        reference_task.physical_squin_kernel
        if with_noise is not False
        else reference_task.noiseless_physical_squin_kernel
    )
    sampled = []
    sample = PPVMSimulatorBackend.sample

    def record(self, kernel, *, shots):
        sampled.append(kernel)
        return sample(self, kernel, shots=shots)

    monkeypatch.setattr(PPVMSimulatorBackend, "sample", record)
    if with_noise is None:
        device = GeminiLogicalSimulatorDevice(
            seed=12, noise_model=simulator.noise_model
        )
    else:
        device = GeminiLogicalSimulatorDevice(
            seed=12, with_noise=with_noise, noise_model=simulator.noise_model
        )
    result = device.run_async(builder_for(shots=1), dry_run=False).result(60)
    assert len(sampled) == 1
    assert sampled[0].is_structurally_equal(expected)
    assert result.shot_results()[0].shape == (1, 160)


def test_all_logical_slots_have_the_correct_physical_filling():
    result = (
        GeminiLogicalSimulatorDevice(seed=123)
        .run_async(builder_for(all_slots, 1), dry_run=False)
        .result(60)
    )
    sorted_shot = next(
        iter(result.storage.get_shots(shot_filter=ShotFilter(frame_type="SORTED")))
    )
    expected = {16 * slot + atom for slot in range(10) for atom in range(7)}
    assert set(np.flatnonzero(sorted_shot.bitstring)) == expected
    assert np.asarray(result.measurements[0]).shape == (1, 70)
    assert np.all(result.filling_at_start[0])


def test_dry_run_has_no_worker_and_does_not_consume_seed(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not start a worker or sample")

    device = GeminiLogicalSimulatorDevice(seed=12)
    state = json.dumps(device._seed_rng.bit_generator.state)
    monkeypatch.setattr(device._executor, "submit", forbidden)
    assert device.run_async(builder_for(), dry_run=True) is None
    assert "pinned" in capsys.readouterr().out
    assert json.dumps(device._seed_rng.bit_generator.state) == state


def test_simulation_does_not_authenticate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local simulation must not authenticate")

    monkeypatch.setattr(GeminiLogicalDevice, "authenticate", forbidden)
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(builder_for(shots=1), dry_run=False)
        .result(60)
    )
    assert result.shot_results()[0].shape == (1, 160)


def test_submissions_reuse_one_worker_and_do_not_overlap(monkeypatch):
    entered, release, second_entered = Event(), Event(), Event()
    run_task = device_module._run_task
    worker_ids = []
    pools = []
    executor_type = device_module.ThreadPoolExecutor

    def make_executor(*args, **kwargs):
        executor = executor_type(*args, **kwargs)
        pools.append(executor)
        return executor

    def blocked(*args):
        worker_ids.append(get_ident())
        if len(worker_ids) == 1:
            entered.set()
            assert release.wait(30)
        else:
            second_entered.set()
        return run_task(*args)

    monkeypatch.setattr(device_module, "ThreadPoolExecutor", make_executor)
    monkeypatch.setattr(device_module, "_run_task", blocked)
    device = GeminiLogicalSimulatorDevice(seed=12)
    definition = device.create_task_definition(builder_for(shots=1))
    first = device.submit_task_definition(task_definition=definition)
    second = None
    try:
        assert entered.wait(30)
        second = device.submit_task_definition(task_definition=definition)
        assert not second_entered.wait(0.1)
    finally:
        release.set()
        first.result(60)
        if second is not None:
            second.result(60)
    assert len(pools) == 1
    assert len(worker_ids) == 2 and len(set(worker_ids)) == 1


def test_submission_snapshots_mutable_builder(monkeypatch):
    entered, proceed = Event(), Event()
    run_task = device_module._run_task

    def blocked(*args):
        entered.set()
        assert proceed.wait(60)
        return run_task(*args)

    monkeypatch.setattr(device_module, "_run_task", blocked)
    builder = builder_for(shots=1)
    device = GeminiLogicalSimulatorDevice(seed=1)
    future = device.run_async(builder, dry_run=False)
    try:
        assert entered.wait(60)
        builder.add_subtask(bell, 2)
        metadata = builder._subtasks[0].qlam_subtask.subtask_metadata
        assert metadata is not None
        metadata.user_metadata = '{"changed":true}'
    finally:
        proceed.set()
    result = future.result(60)
    assert len(result.full_subtasks()) == 1
    assert json.loads(result.full_subtasks()[0]["metadata"]["user_metadata"]) == {
        "name": "pinned"
    }


@pytest.mark.parametrize("backend", ["tsim", "clifft"])
def test_seeded_devices_replay_the_same_submission_sequence(backend):
    if backend == "clifft":
        pytest.importorskip("clifft")
    a = GeminiLogicalSimulatorDevice(backend=backend, seed=123, batch_shots=2)
    b = GeminiLogicalSimulatorDevice(backend=backend, seed=123, batch_shots=2)
    builder = builder_for(bell, 5)
    for _ in range(2):
        ra = a.run_async(builder, dry_run=False).result(60)
        rb = b.run_async(builder, dry_run=False).result(60)
        np.testing.assert_array_equal(ra.shot_results()[0], rb.shot_results()[0])
        assert ra.task_ids() != rb.task_ids()


@pytest.mark.parametrize("seed", [None, 123])
def test_clifft_receives_submission_seed(seed, monkeypatch):
    pytest.importorskip("clifft")
    sample = CliffTSimulatorBackend.sample
    received = []

    def record(self, kernel, *, shots):
        received.append(self.seed)
        return sample(self, kernel, shots=shots)

    monkeypatch.setattr(CliffTSimulatorBackend, "sample", record)
    result = (
        GeminiLogicalSimulatorDevice(backend="clifft", seed=seed)
        .run_async(builder_for(shots=1), dry_run=False)
        .result(60)
    )
    expected = (
        int(np.random.default_rng(seed).integers(0, 2**63))
        if seed is not None
        else None
    )
    assert received == [expected]
    assert result.shot_results()[0].shape == (1, 160)


def test_backend_failure_is_raised_by_the_standard_future(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("sample failed")

    monkeypatch.setattr(PPVMSimulatorBackend, "sample", fail)
    future = GeminiLogicalSimulatorDevice().run_async(
        builder_for(shots=1), dry_run=False
    )
    with pytest.raises(RuntimeError, match="sample failed"):
        future.result(60)
    assert future.done()


@pytest.mark.parametrize("payload", [None, np.zeros((1, 1), dtype=bool)])
def test_invalid_sample_shapes_are_not_stored(monkeypatch, payload):
    monkeypatch.setattr(
        PPVMSimulatorBackend,
        "sample",
        lambda *args, **kwargs: BackendSample(measurements=payload),
    )
    future = GeminiLogicalSimulatorDevice().run_async(
        builder_for(shots=1), dry_run=False
    )
    with pytest.raises(ValueError, match="measurements|Measurement shape"):
        future.result(60)


def test_ppvm_loss_becomes_absence_in_detected_frame(monkeypatch):
    def all_lost(*args, shots):
        return BackendSample(measurements=np.full((shots, 14), None, dtype=object))

    monkeypatch.setattr(PPVMSimulatorBackend, "sample", all_lost)
    result = (
        GeminiLogicalSimulatorDevice(backend="ppvm")
        .run_async(builder_for(), dry_run=False)
        .result(60)
    )
    assert not np.any(result.shot_results()[0])
    assert np.all(result.measurements[0])
    assert np.all(result.filling_at_start[0])


def test_noisy_ppvm_sampling_uses_existing_noise_model():
    pytest.importorskip("ppvm")
    pytest.importorskip("bloqade.tsim")
    result = (
        GeminiLogicalSimulatorDevice(backend="ppvm", seed=123, with_noise=True)
        .run_async(builder_for(shots=3), dry_run=False)
        .result(60)
    )
    assert result.shot_results()[0].shape == (3, 160)
    assert np.all(result.filling_at_start[0])


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**63},
        {"batch_shots": 0},
        {"batch_shots": True},
        {"batch_shots": -1},
        {"batch_shots": 1.5},
        {"batch_shots": "2"},
        {"backend": "unknown"},
        {"with_noise": "true"},
    ],
)
def test_invalid_device_configuration(change):
    with pytest.raises((ValueError, TypeError)):
        GeminiLogicalSimulatorDevice(**change)


def test_empty_builder_rejected():
    with pytest.raises(ValueError, match="programs|subtasks"):
        GeminiLogicalSimulatorDevice().run_async(TaskBuilder(), dry_run=False)


def test_runtime_arguments_rejected_before_submission():
    builder = TaskBuilder()
    builder.add_subtask(parameterized, 1, theta=0.1)
    with pytest.raises(
        (TaskFinalizeError, ValueError), match="argument|parameter|validation"
    ):
        GeminiLogicalSimulatorDevice().run_async(builder, dry_run=False)
