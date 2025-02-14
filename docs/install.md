# Installation

Bloqade is available in [PyPI](https://pypi.org/) and
thus can be installed via [`pip`](https://pypi.org/project/pip/).
Install Bloqade using the following command:

```bash
pip install bloqade
```

Bloqade support python 3.10+.

We strongly recommend developing your compiler project using [`uv`](https://docs.astral.sh/uv/),
which is the official development environment for Bloqade. You can install `uv` using the following command:


=== "Linux and macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    then

    ```bash
    uv add kirin-toolchain
    ```

=== "Windows"

    ```cmd
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    then

    ```cmd
    uv add kirin-toolchain
    ```

## Bloqade and its extensions

Bloqade also comes with a few friends that you might find useful:

### `bloqade-runtime-pyqrack`

PyQrack is a high-performance simulator for quantum circuits. We provide a Bloqade extension that allows you to run your Bloqade programs with PyQrack via interpreting the Bloqade programs with PyQrack as a runtime. You can install `bloqade-runtime-pyqrack` using the following command:

```bash
pip install bloqade[pyqrack]
```

or

```bash
pip install bloqade[pyqrack-cpu]
```

if you want to install the CPU-only version of PyQrack.

### qBraid extension

The qBraid extension allows you to run your Bloqade programs on the qBraid platform. You can install this extension via

```bash
pip install bloqade[qbraid]
```

### QASM2 extension

The QASM2 extension allows you to compile your Bloqade programs to/from QASM2, or QuEra's custom extension of QASM2. You can install this extension via

```bash
pip install bloqade[qasm2]
```

## Development

If you want to contribute to Bloqade, you can clone the repository from GitHub:

```bash
git clone https://github.com/QuEraComputing/bloqade.git
```

We use `uv` to manage the development environment, after you install `uv`, you can install the development dependencies using the following command:

```bash
uv sync
```

Our code review requires that you pass the tests and the linting checks. We recommend
you to install `pre-commit` to run the checks before you commit your changes, the command line
tool `pre-commit` has been installed as part of the development dependencies. You can setup
`pre-commit` using the following command:

```bash
pre-commit install
```
