"""Local TaskBuilder execution with hardware-compatible Gemini logical results."""

from __future__ import annotations

from uuid import UUID, uuid4
from typing import TYPE_CHECKING, Any, Literal, overload
from datetime import datetime, timezone
from contextlib import nullcontext
from dataclasses import field, dataclass
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from kirin import ir
from kirin.validation import ValidationSuite
from bloqade.core.device import ShotFilter, ShotResult, DictStorage
from kirin.serialization import JSONSerializer
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import LogicalPipeline, MoveToSquinLogical
from bloqade.core.device.task import KernelSerializer
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.core.device.future import DEFAULT_FETCH_OPTIONS, ApiFetchOptions
from bloqade.lanes.analysis.atom import AtomInterpreter
from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.core.device.task_builder import TaskBuilder, FinalizeContext
from bloqade.core.device.local_storage import SQLiteStorage, StorageBackend
from qlam_core.plugins.tasks.api.tasks_models import TaskDefinition
from bloqade.lanes.analysis.atom._shot_remapping import ShotRemappingErr

from bloqade.gemini.logical.group import kernel as logical_kernel
from bloqade.gemini.device.logical.utils import ShotRemappingException
from bloqade.gemini.device.logical.device import (
    _bloqade_version,
    _default_validation_suite,
)
from bloqade.gemini.device.logical.result import GeminiLogicalResult
from bloqade.gemini.device.simulator_backend import (
    PPVMSimulatorBackend,
    TsimSimulatorBackend,
    CliffTSimulatorBackend,
    _ppvm,
    _validate_seed,
)

if TYPE_CHECKING:
    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

    from bloqade.gemini.device.simulator_backend import AbstractSimulatorBackend


def _slm_layout(move_kernel: ir.Method, arch: ArchSpec) -> tuple[list[int], np.ndarray]:
    zone = ZoneAddress(0)
    width = len(list(arch.yield_zone_locations(zone)))
    remapping = AtomInterpreter(
        move_kernel.dialects, arch_spec=arch
    ).get_shot_remapping(move_kernel)
    if isinstance(remapping, ShotRemappingErr):
        raise ShotRemappingException(
            f"Cannot map physical measurements: {remapping.diagnostic}"
        )
    mapping = remapping.mapping
    if (
        not mapping
        or remapping.frame_size != width
        or len(set(mapping)) != len(mapping)
        or any(index < 0 or index >= width for index in mapping)
    ):
        raise ValueError("Measurements must map one-to-one to the Zone-0 SLM frame")

    fills = [
        stmt
        for stmt in move_kernel.callable_region.walk()
        if isinstance(stmt, move.Fill)
    ]
    if len(fills) != 1:
        raise ValueError("Only one initial SLM Fill is supported; no dynamic refills")
    sorted_frame = np.zeros(width, dtype=bool)
    for address in fills[0].location_addresses:
        if address.zone_id != 0:
            raise ValueError("Only SLM filling in Zone 0 is supported")
        index = arch.get_zone_index(address, zone)
        if index is None or not 0 <= index < width:
            raise ValueError("Initial filling has no corresponding Zone-0 SLM site")
        sorted_frame[index] = True
    return mapping, sorted_frame


def _validate_task_definition(definition: TaskDefinition) -> None:
    if not isinstance(definition, TaskDefinition):
        raise TypeError("submit_task_definition requires a TaskDefinition")
    if not definition.programs or not definition.subtasks:
        raise ValueError("At least one program and subtask are required")
    for subtask in definition.subtasks:
        if not 0 <= subtask.program_index < len(definition.programs):
            raise ValueError("Invalid subtask program index")
        if subtask.arguments:
            raise ValueError("Runtime subtask arguments are unsupported")
        if subtask.num_shots <= 0:
            raise ValueError("num_shots must be positive")


def _sqlite_connection_factory(storage: SQLiteStorage) -> Callable[[], SQLiteStorage]:
    """Capture SQLite file/config on the caller thread, not its connection."""
    db_file = next(
        (
            row[2]
            for row in storage.conn.execute("PRAGMA database_list")
            if row[1] == "main"
        ),
        "",
    )
    if not db_file:
        raise ValueError("Worker connections require a file-backed SQLite database")
    timeout_ms = int(storage.conn.execute("PRAGMA busy_timeout").fetchone()[0])

    def connect() -> SQLiteStorage:
        # SQLiteStorage's existing constructor uses its default timeout during
        # schema initialization. Restore the caller's timeout for later writes.
        # Allocate first so even a partially initialized core storage can be
        # closed if its constructor raises after opening the connection.
        worker = SQLiteStorage.__new__(SQLiteStorage)
        try:
            SQLiteStorage.__init__(worker, db_file)
            worker.conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
        except BaseException:
            if hasattr(worker, "conn"):
                worker.close()
            raise
        return worker

    return connect


def _run_task(
    definition: TaskDefinition,
    backend: AbstractSimulatorBackend,
    with_noise: bool,
    noise_model: LogicalNoiseModelABC | None,
    batch_shots: int | None,
    storage: StorageBackend,
    connection_factory: Callable[[], SQLiteStorage] | None = None,
) -> GeminiLogicalResult[Any]:
    # Open, use and close a SQLite writer on the worker thread. Merely retain
    # the caller's storage for the returned view; never query its connection.
    context = (
        connection_factory() if connection_factory is not None else nullcontext(storage)
    )
    with context as writer:
        result = _simulate_task(
            definition, backend, with_noise, noise_model, batch_shots, writer
        )
    return GeminiLogicalResult(storage=storage, shot_filter=result.shot_filter)


def _simulate_task(
    definition: TaskDefinition,
    backend: AbstractSimulatorBackend,
    with_noise: bool,
    noise_model: LogicalNoiseModelABC | None,
    batch_shots: int | None,
    storage: StorageBackend,
) -> GeminiLogicalResult[Any]:
    resolved_noise_model = (
        noise_model if noise_model is not None else generate_logical_noise_model()
    )
    arch_spec = physical.get_arch_spec()
    serializer = JSONSerializer()
    prepared = []
    for program in definition.programs:
        kernel = logical_kernel.decode(serializer.decode(program.content))
        if kernel.args:
            raise ValueError("Simulator device programs must have no runtime arguments")
        # Kirin JSON omits analysis hints needed by explicitly pinned qalloc_at.
        run_pass = logical_kernel.run_pass
        if run_pass is None:
            raise RuntimeError("Gemini logical kernel has no pass pipeline")
        run_pass(kernel, aggressive_unroll=True, verify=True)
        move_kernel = LogicalPipeline(transversal_rewrite=True, simulation=True).emit(
            kernel
        )
        mapping, sorted_frame = _slm_layout(move_kernel, arch_spec)
        physical_kernel = MoveToSquinLogical(
            arch_spec=arch_spec,
            noise_model=resolved_noise_model,
            add_noise=with_noise,
        ).emit(move_kernel)
        prepared.append((physical_kernel, mapping, sorted_frame))

    task_id = str(uuid4())
    storage.add_task_definition(task_id, definition, datetime.now(timezone.utc))
    shot_index = 0
    for subtask_index, subtask in enumerate(definition.subtasks):
        physical_kernel, mapping, sorted_frame = prepared[subtask.program_index]
        batch_size = subtask.num_shots if batch_shots is None else batch_shots
        for start in range(0, subtask.num_shots, batch_size):
            count = min(batch_size, subtask.num_shots - start)
            sample = backend.sample(physical_kernel, shots=count)
            if sample.measurements is None:
                raise ValueError(
                    "The simulator backend must return physical measurements"
                )
            raw = np.asarray(sample.measurements)
            if raw.shape != (count, len(mapping)):
                raise ValueError(
                    f"Measurement shape {raw.shape} != {(count, len(mapping))}"
                )
            # PPVM represents atom loss as None. Like |1>, loss means absence
            # in DETECTED; normalize it before inverting measurement bits.
            lost = raw == None  # noqa: E711 - elementwise comparison for NumPy arrays
            measurements = np.asarray(np.where(lost, True, raw), dtype=bool)
            detected = np.zeros((count, len(sorted_frame)), dtype=bool)
            detected[:, mapping] = ~measurements
            rows = []
            for offset, detected_frame in enumerate(detected):
                for frame_type, bitstring in (
                    ("SORTED", sorted_frame),
                    ("DETECTED", detected_frame),
                ):
                    rows.append(
                        ShotResult(
                            task_id=task_id,
                            shot_index=shot_index + offset,
                            subtask_index=subtask_index,
                            subtask_shot_index=start + offset,
                            frame_type=frame_type,
                            bitstring=bitstring.copy(),
                        )
                    )
            storage.add_shots(rows)
            shot_index += count
        storage.update_subtasks_completed_date(
            task_id,
            [
                {
                    "completed_date": datetime.now(timezone.utc),
                    "shot_results": [{"subtask_index": subtask_index}],
                }
            ],
        )
    return GeminiLogicalResult(
        storage=storage,
        shot_filter=ShotFilter(task_ids=(task_id,), frame_type="DETECTED"),
    )


@dataclass(kw_only=True)
class GeminiLogicalSimulatorDevice:
    """Run TaskBuilder workloads locally and return GeminiLogicalResult views.

    Uses the same builder finalization and validation as GeminiLogicalDevice.
    Each submission returns a standard concurrent.futures.Future; its result is
    an existing GeminiLogicalResult backed by the supplied storage or a fresh
    DictStorage. No QLAM server, authentication, context manager, or explicit
    close is required.
    One single-worker executor is shared by submissions on this device. Its
    thread starts on first submission and is reused while the device is alive.
    Separate device instances do not share this execution queue.

    Sorting is ideal, and raw frames use physical Zone-0 SLM order (160 sites
    on the default architecture). Runtime arguments, dynamic refills and
    multiple SLM zones are unsupported. With default storage, all result frames
    remain in memory; explicit batching bounds temporary sampling arrays, not
    the size of the result.

    Args:
        backend: Sampling backend, "ppvm" (default), "tsim", or "clifft".
            PPVM and Tsim are included with bloqade. Install ``bloqade[sim]``
            for CliffT (Python 3.12 or later).
        seed: Optional root seed. Submissions receive successive child seeds;
            replay requires the same submission order and batching.
        with_noise: Insert circuit noise, default True. Sorting remains ideal.
        noise_model: Optional custom logical noise model. None uses the same
            default model as GeminiLogicalSimulator.
        batch_shots: Maximum shots per sampling batch. The default, None, samples
            all shots of each subtask in one batch.
        program_language: Language name recorded in task metadata, default "squin".
        language_version: Language version used for serialization and metadata.
        kernel_serializer: Kernel encoder, default JSONSerializer. Local execution
            requires compatible Kirin JSON output.
        dialect_group: Allowed dialects for builder finalization, default logical.kernel.
        validation_suite: Submission validation, using the Gemini hardware defaults.
    """

    program_language: str = "squin"
    language_version: str = _bloqade_version
    kernel_serializer: KernelSerializer = field(default_factory=JSONSerializer)
    dialect_group: ir.DialectGroup | None = field(
        default_factory=lambda: logical_kernel
    )
    validation_suite: ValidationSuite | None = field(
        default_factory=_default_validation_suite
    )
    backend: Literal["tsim", "ppvm", "clifft"] = "ppvm"
    seed: int | None = None
    with_noise: bool = True
    noise_model: LogicalNoiseModelABC | None = None
    batch_shots: int | None = None
    _seed_rng: np.random.Generator = field(init=False, repr=False)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend not in ("tsim", "ppvm", "clifft"):
            raise ValueError("backend must be 'tsim', 'ppvm', or 'clifft'")
        _validate_seed(self.seed)
        if not isinstance(self.with_noise, bool):
            raise TypeError("with_noise must be bool")
        if self.batch_shots is not None and (
            isinstance(self.batch_shots, bool)
            or not isinstance(self.batch_shots, int)
            or self.batch_shots <= 0
        ):
            raise ValueError("batch_shots must be None or a positive integer")
        self._seed_rng = np.random.default_rng(self.seed)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemini-logical-simulator"
        )

    def _finalize_context(self) -> FinalizeContext:
        """Return this simulator device's validation and serialization settings."""
        return FinalizeContext(
            program_language=self.program_language,
            language_version=self.language_version,
            kernel_serializer=self.kernel_serializer,
            dialect_group=self.dialect_group,
            validation_suite=self.validation_suite,
        )

    def create_task_definition(self, task_builder: TaskBuilder) -> TaskDefinition:
        """Validate and serialize a builder without submitting or printing.

        Uses this device's finalization context. No worker, storage write,
        authentication, or seed consumption occurs. The builder remains editable;
        group and profile metadata are selected at submission.
        """
        if not isinstance(task_builder, TaskBuilder):
            raise TypeError("create_task_definition requires a TaskBuilder")
        definition = task_builder._finalize(self._finalize_context())
        _validate_task_definition(definition)
        return definition

    @overload
    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: Literal[False],
        group: str | None = None,
        profile_id: str | UUID | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> Future[GeminiLogicalResult[Any]]: ...

    @overload
    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: Literal[True],
        group: str | None = None,
        profile_id: str | UUID | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> None: ...

    @overload
    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: bool,
        group: str | None = None,
        profile_id: str | UUID | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> Future[GeminiLogicalResult[Any]] | None: ...

    def run_async(
        self,
        task_builder: TaskBuilder,
        *,
        dry_run: bool,
        group: str | None = None,
        profile_id: str | UUID | None = None,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
    ) -> Future[GeminiLogicalResult[Any]] | None:
        """Finalize a builder and execute a snapshot in a background thread.

        A dry run validates, serializes, and prints the builder without starting
        an executor or consuming a seed. Builder edits after submission do not
        affect the submitted task. Validation errors are raised immediately;
        compilation and sampling errors are raised by ``future.result()``.
        Standard future timeouts do not cancel a running simulation.

        ``storage`` receives task metadata and shots; None creates a fresh
        DictStorage. File-backed SQLiteStorage uses a separate worker-owned
        connection and returns a result referencing the original storage.
        Read that result on the thread owning the original connection, and
        keep the original storage open while using the result. Private
        in-memory/temporary SQLite databases are unsupported. Other backends
        must support worker-thread access. Storage is ignored for a dry run.
        ``profile_id`` is recorded as metadata only, and does not
        override a profile already supplied by the finalized task definition.
        ``group`` and ``fetch_options`` are accepted for Device compatibility
        and ignored: local simulation performs no group lookup or API polling.

        Delegates to :meth:`create_task_definition` and, unless this is a dry
        run, :meth:`submit_task_definition`.
        """
        definition = self.create_task_definition(task_builder)
        if dry_run:
            print(task_builder.summary())
            return None
        return self.submit_task_definition(
            task_definition=definition,
            storage=storage,
            fetch_options=fetch_options,
            group=group,
            profile_id=profile_id,
        )

    def submit_task_definition(
        self,
        *,
        task_definition: TaskDefinition,
        storage: StorageBackend | None = None,
        fetch_options: ApiFetchOptions = DEFAULT_FETCH_OPTIONS,
        group: str | None = None,
        profile_id: str | UUID | None = None,
    ) -> Future[GeminiLogicalResult[Any]]:
        """Submit a prepared definition to local simulation.

        Validates its structure and snapshots it before starting a worker.
        Compilation/sampling failures propagate through the returned standard
        Future. File-backed SQLite storage uses a worker-owned connection;
        results reference the original storage, which the caller must keep
        open and access on its owning thread. None creates a fresh DictStorage.

        An existing definition profile takes precedence over ``profile_id``;
        the latter fills missing metadata only. ``group`` and ``fetch_options``
        are accepted and ignored, without QLAM requests or authentication.
        """
        _validate_task_definition(task_definition)
        definition = task_definition.model_copy(deep=True)
        connection_factory = (
            _sqlite_connection_factory(storage)
            if isinstance(storage, SQLiteStorage)
            else None
        )
        if definition.profile_id is None and profile_id is not None:
            definition = definition.model_copy(
                update={
                    "profile_id": (
                        UUID(profile_id) if isinstance(profile_id, str) else profile_id
                    )
                }
            )
        if storage is None:
            storage = DictStorage()

        if self.backend == "ppvm":
            try:
                # Avoid first initializing PPVM's native module in a
                # short-lived executor thread.
                _ppvm()
            except ImportError as exc:
                failed: Future[GeminiLogicalResult[Any]] = Future()
                failed.set_exception(exc)
                return failed

        seed = int(self._seed_rng.integers(0, 2**63)) if self.seed is not None else None
        backend_cls = {
            "tsim": TsimSimulatorBackend,
            "ppvm": PPVMSimulatorBackend,
            "clifft": CliffTSimulatorBackend,
        }[self.backend]
        backend = backend_cls(seed=seed)
        return self._executor.submit(
            _run_task,
            definition,
            backend,
            self.with_noise,
            self.noise_model,
            self.batch_shots,
            storage,
            connection_factory,
        )
