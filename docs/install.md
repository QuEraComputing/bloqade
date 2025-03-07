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


## Bloqade and its friends

Bloqade is a Python namespace package, we officially provide several sub-packages, each of which is an eDSL for neutral atom quantum computing. The following is a list of the sub-packages in Bloqade:

### `bloqade.qasm2`

QASM2 and its extensions support for neutral atom quantum computing. Available via:

```bash
pip install bloqade[qasm2]
```

### `bloqade.analog`

Analog quantum computing eDSL for neutral atom quantum computing (previously `bloqade-python`). Available via:

```bash
pip install bloqade-analog
```

### `bloqade.pyqrack`

Support of the PyQrack simulator as a runtime backend for QASM2 and extensions.

```bash
pip install bloqade-pyqrack
```

### `bloqade.qbraid`

Support of the qBraid cloud service as a runtime backend for retrieving noise models and running circuits.

```bash
pip install bloqade[qbraid]
```

### `bloqade.stim` (Experimental)

STIM and its extensions support for neutral atom quantum computing. Available via:

```bash
pip install bloqade[stim]
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
