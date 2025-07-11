# Installation

Bloqade is compatible with Python 3.10+ and available on [PyPI](https://pypi.org/project/bloqade/).
You can install it via [`pip`](https://pypi.org/project/pip/) into your environment:


```bash
pip install bloqade
```


## Bloqade and its friends

Bloqade is a Python namespace package, we officially provide several sub-packages, each of which is an eDSL for neutral atom quantum computing. The following is a list of the sub-packages in Bloqade:

!!! note

    If you have already installed Bloqade via the instructions above, all the following subpckages are already installed with the exception of the `stim` eDSL which is currently experimental.

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

### `bloqade.qbraid`

Support of the qBraid cloud service as a runtime backend for retrieving noise models and running circuits.

```bash
pip install bloqade[qbraid]
```

### `bloqade.stim` (Experimental)

Stim and its extensions support for neutral atom quantum computing. Available via:

```bash
pip install bloqade[stim]
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

=== 

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
