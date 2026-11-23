# Installation

Bloqade is compatible with Python 3.11+ and available on [PyPI](https://pypi.org/project/bloqade/).
Install it into your Python environment with:

```bash
pip install bloqade
```

## What the default install includes

The `bloqade` distribution ships the Gemini SDK. Use
`from bloqade.gemini import logical, physical` to access the kernel APIs.
The same installation brings in the circuit and analog programming packages,
the Lanes compiler, and simulation and decoding packages. QASM2, QASM3, Cirq,
qBraid, circuit visualization, and Stim support are enabled through
`bloqade-circuit`. See the [full list of direct dependencies](https://github.com/QuEraComputing/bloqade/blob/main/src/bloqade/README.md#what-gets-installed).

## Optional extras

| Extra | What it adds |
| --- | --- |
| `cudaq` | CUDA-Q and QIR integration on supported platforms. |
| `sim` | Additional simulation backends, including CliffT on Python 3.12 and later. |
| `decoding` | Additional decoding solvers, including Tesseract on Python 3.12 and later. |
| `atom-vis` | Atom movement and architecture visualization tools. |
| `circ-opt` | Circuit parallelization with Cirq. |
| `pyqrack-accel` | OpenCL and CUDA acceleration support for PyQrack. |

For example, to install the atom visualization tools:

```bash
pip install 'bloqade[atom-vis]'
```

## Development

If you want to contribute to Bloqade, you can clone the repository from GitHub:

```bash
git clone https://github.com/QuEraComputing/bloqade.git
```

We use [`uv`](https://docs.astral.sh/uv/) to manage the development environment.

You can install `uv` via the following:

=== "Linux and macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```cmd
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then you can install the development dependencies executing one of the following commands:

```bash
# For contributing to code
uv sync --group dev
# For contributions to documentation
uv sync --group doc
# For just getting everything mentioned above
uv sync --all-groups
```

Our code review requires that you pass the tests and linting checks. We recommend
you install `pre-commit` to run the checks before you commit your changes.  `pre-commit`
is already specified as a development dependency for bloqade and once installed,
you can setup `pre-commit` using the following command:

```bash
pre-commit install
```
