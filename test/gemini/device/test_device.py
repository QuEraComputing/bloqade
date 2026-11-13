from bloqade import squin
from bloqade.gemini import GeminiLogicalDevice, logical
from bloqade.gemini.device.logical.task import (
    GeminiKernelBatchTask,
    GeminiSingleKernelTask,
    GeminiParameterScanTask,
)

# === GeminiLogicalDevice ===


@logical.kernel(aggressive_unroll=True)
def device_bell():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def device_trio():
    q = squin.qalloc(3)
    squin.h(q[0])
    return logical.terminal_measure(q)


# --- task ---


def test_device_task_returns_single_kernel_task():
    device = GeminiLogicalDevice()
    task = device.task(kernel=device_bell, num_shots=3)
    assert isinstance(task, GeminiSingleKernelTask)
    assert task.kernel is device_bell
    assert task.num_shots == 3


def test_device_task_default_program_language_is_squin():
    device = GeminiLogicalDevice()
    task = device.task(kernel=device_bell, num_shots=1)
    assert task.program_language == "squin"


def test_device_task_explicit_program_language_propagates():
    device = GeminiLogicalDevice()
    task = device.task(kernel=device_bell, num_shots=1, program_language="qasm")
    assert task.program_language == "qasm"


def test_device_task_arguments_propagate():
    device = GeminiLogicalDevice()
    task = device.task(
        kernel=device_bell,
        num_shots=1,
        arguments={"theta": 0.5},
    )
    assert task.arguments == {"theta": 0.5}


def test_device_task_metadata_dict_wrapped_into_list():
    """device.task takes a single metadata dict (or None) and stores it as
    a one-element list to match GeminiTaskABC's per-subtask metadata API."""
    device = GeminiLogicalDevice()
    task = device.task(
        kernel=device_bell,
        num_shots=1,
        metadata={"experiment": "alpha"},
    )
    assert task.metadata == {"experiment": "alpha"}


def test_device_task_metadata_none_stays_none():
    device = GeminiLogicalDevice()
    task = device.task(kernel=device_bell, num_shots=1)
    assert task.metadata is None


def test_device_task_default_context_name():
    device = GeminiLogicalDevice()
    task = device.task(kernel=device_bell, num_shots=1)
    assert task.context_name == "gemini-logical"


def test_device_task_context_name_propagates_from_device():
    device = GeminiLogicalDevice(context_name="my-context")
    task = device.task(kernel=device_bell, num_shots=1)
    assert task.context_name == "my-context"


# --- batch_task ---


def test_device_batch_task_returns_batch_task():
    device = GeminiLogicalDevice()
    task = device.batch_task(kernels=[device_bell, device_trio], num_shots=1)
    assert isinstance(task, GeminiKernelBatchTask)
    assert task.kernels == [device_bell, device_trio]


def test_device_batch_task_arguments_and_metadata_passed_through():
    """Unlike .task, .batch_task takes arguments/metadata as lists directly —
    they must reach the resulting task object unchanged."""
    device = GeminiLogicalDevice()
    task = device.batch_task(
        kernels=[device_bell, device_trio],
        arguments=[{"x": 1}, {"x": 2}],
        metadata=[{"a": 1}, {"a": 2}],
        num_shots=1,
    )
    assert task.arguments == [{"x": 1}, {"x": 2}]
    assert task.metadata == [{"a": 1}, {"a": 2}]


# --- parameter_scan ---


def test_device_parameter_scan_returns_parameter_scan_task():
    device = GeminiLogicalDevice()
    task = device.parameter_scan(
        device_bell,
        arguments=[{"x": 0.1}, {"x": 0.5}],
        num_shots=1,
    )
    assert isinstance(task, GeminiParameterScanTask)
    assert task.kernel is device_bell
    assert task.arguments == [{"x": 0.1}, {"x": 0.5}]


def test_device_parameter_scan_metadata_passed_through():
    device = GeminiLogicalDevice()
    task = device.parameter_scan(
        device_bell,
        arguments=[{"x": 0.1}],
        metadata=[{"a": 1}],
        num_shots=1,
    )
    assert task.metadata == [{"a": 1}]
