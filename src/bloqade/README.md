# The Bloqade package

The `bloqade` distribution ships the [Gemini SDK](gemini/) in the shared
`bloqade` Python namespace. The logical and physical kernel APIs, devices,
simulator interfaces, and decoding workflows live here. Import the SDK with
`from bloqade import gemini`.

## What gets installed

`pip install bloqade` installs the Gemini SDK and these companion distributions:

| Distribution | What it provides |
| --- | --- |
| `bloqade-circuit` | Circuit programming, with the `cirq`, `qasm2`, `qasm3`, `qbraid`, `vis`, and `stim` extras enabled by default. |
| `bloqade-analog` | Analog neutral atom programming with `bloqade.analog`. |
| `bloqade-lanes` | Neutral atom compiler dialects and passes, including placement and atom movement. |
| `bloqade-tsim` | Sampling noisy quantum circuits with non-Clifford gates. |
| `bloqade-core` | Shared device, task, and result interfaces. |
| `bloqade-decoders` | Quantum error correction decoders and their interfaces. |
| `bloqade-ppvm` | Pauli propagation and stabilizer-tableau simulation. |

The other direct dependencies are `kirin-toolchain`, `numpy`, `scipy`,
and `typing-extensions`. Version requirements are in
[pyproject.toml](../../pyproject.toml).

## Optional extras

The `bloqade` distribution offers `cudaq`, `sim`, `decoding`, `atom-vis`,
`circ-opt`, and `pyqrack-accel`. For example, install the additional decoding
solvers with `pip install 'bloqade[decoding]'`. See the
[installation guide](../../docs/install.md) for the other options.
