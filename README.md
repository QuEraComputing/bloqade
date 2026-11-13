# Welcome to Bloqade -- QuEra's Neutral Atom SDK

[![CI](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml/badge.svg)](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuEraComputing/bloqade/graph/badge.svg?token=BpHsAYuzdo)](https://codecov.io/gh/QuEraComputing/bloqade)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/bloqade.svg?color=%2334D058)](https://pypi.org/project/bloqade)
[![Documentation](https://img.shields.io/badge/Documentation-6437FF)](https://queracomputing.github.io/bloqade/)
[![DOI](https://zenodo.org/badge/629628885.svg)](https://zenodo.org/doi/10.5281/zenodo.11114109)


Bloqade is a Python SDK for neutral atom quantum computing. It provides a set of embedded domain-specific languages (eDSLs) for programming neutral atom quantum computers. Bloqade is designed to be a high-level, user-friendly SDK that abstracts away the complexities of neutral atom quantum computing, allowing users to focus on developing quantum algorithms and compilation strategies for neutral atom quantum computers.

> [!IMPORTANT]
>
> This project is in the early stage of development. API and features are subject to change.

## Installation

### Install via `uv` (Recommended)

```py
uv add bloqade
```

This distribution includes `bloqade.gemini`: logical and physical kernel
decorators, device/simulator interfaces, and decoding workflows. Compiler
infrastructure is provided by the `bloqade-lanes` dependency. See
[Gemini migration notes](GEMINI_MIGRATION.md) for the coordinated release
requirements and compatibility details.

## Local TaskBuilder simulation

Install Bloqade with `pip install bloqade`. For the optional CliffT backend,
use `pip install 'bloqade[sim]'` (Python 3.12 or later).
Use `GeminiLogicalSimulatorDevice` to execute a `TaskBuilder` locally:

```python
from bloqade import squin
from bloqade.core.device.task_builder import TaskBuilder
from bloqade.gemini import GeminiLogicalSimulatorDevice, logical

@logical.kernel
def bell():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return logical.terminal_measure(q)

builder = TaskBuilder()
builder.add_subtask(bell, num_shots=100)

device = GeminiLogicalSimulatorDevice(backend="tsim", seed=1234, with_noise=False)
future = device.run_async(builder, dry_run=False)
result = future.result(timeout=80)
print(result.return_values)
```

The standard Python future returns the existing `GeminiLogicalResult`, including
raw 160-site SLM shots, metadata, and sorting postselection. Each submission uses
fresh in-memory storage unless `storage=` is supplied. Sorting is ideal, and
circuit noise defaults to on; set `with_noise=False` for ideal simulation.
PPVM is the default backend; `backend="tsim"` and `backend="clifft"` are also
available.
Programs must take no runtime arguments. No server or authentication is needed.
Each device owns one single-worker executor: submissions queue and execute
sequentially, reusing the worker while the device is alive. Separate device
instances have independent queues; this does not make the compiler globally
thread-safe.
`device.run_async(builder, dry_run=True)` validates and prints the task without
starting a simulation. A future timeout does not stop a running simulation.

The simulator owns the same configurable finalization fields as the hardware
device: `program_language`, `language_version`, `kernel_serializer`,
`dialect_group`, and `validation_suite`. Local execution requires a serializer
that produces compatible Kirin JSON. `run_async` also accepts `profile_id` as
local task metadata. The remote-only `group` and `fetch_options` arguments are
accepted and ignored; no group resolution, authentication, or polling occurs.
File-backed `SQLiteStorage` uses a separate worker-owned connection. Results
reference the caller's original storage, which must remain open and be read on
its owning thread. Private in-memory/temporary SQLite databases are unsupported.
Other supplied storage must support worker-thread access. Dry runs ignore
storage without writing to it.

## Documentation

The documentation is available at [https://queracomputing.github.io/bloqade/latest/](https://queracomputing.github.io/bloqade/latest/). We are at an early stage of completing the documentation with more details and examples, so comments and contributions are most welcome!

## Contributing

We welcome contributions! Please check the [contribution guidelines](https://queracomputing.github.io/bloqade/latest/contrib/) for a detailed guide.

> [!NOTE]
>
> We welcome contributions that use AI assistance. However, **we ask that first-time contributors do not submit AI-generated PRs**.


## License

Apache License 2.0 with LLVM Exceptions
