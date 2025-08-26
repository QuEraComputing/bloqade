!!! warning
    This page is under construction. The content may be incomplete or incorrect. Submit an issue
    on [GitHub](https://github.com/QuEraComputing/bloqade/issues/new) if you need help or want to
    contribute.


# Digital Bloqade

The digital submodule of Bloqade, called `bloqade-circuit` defines a set of embedded domain-specific languages (eDSLs) that can be used to define digital quantum programs.
These programs are intended for both simulation and to be run on hardware. This package is open source and can be found [on GitHub](https://github.com/QuEraComputing/bloqade-circuit).

Please refer to the [Dialects and Kernels](./dialects_and_kernels.md) section of this documentation for an overview over the most important eDSLs.
The infrastructure behind these compilers is built on top of [Kirin](https://queracomputing.github.io/kirin/latest/).

It is easiest to learn how to use this package by checking out the [examples & tutorials section](tutorials_index.md), where we show how you can build and study different quantum programs written in different DSLs.
You can also find the corresponding scripts in [jupytext format](https://jupytext.readthedocs.io/en/latest/) at the [bloqade repository](https://github.com/QuEraComputing/bloqade) under `docs/digital/examples/`.

Finally, if you want the full details on the API, please refer to the [API reference documentation](../../reference/bloqade-circuit/src/bloqade/device/).

## Installation

The package comes as a submodule of Bloqade, so you can just run

```
pip install bloqade
```

in order to obtain it.

Sometimes, you may want to reduce the number of dependencies, in which case you can also only install the submodule

```
pip install bloqade-circuit
```

Note, that bloqade-circuit also has some optional dependencies, which you may want to install.
For example

```
pip install bloqade-circuit[cirq,qasm2,stim]
```

## TL;DR

Here's a GHZ preparation circuit written in the [`squin`](../../reference/bloqade-circuit/src/bloqade/squin/) dialect:

```python
from bloqade import squin

@squin.kernel
def ghz(n: int):
    q = squin.qubit.new(n)

    squin.gate.h(q[0])

    for i in range(1, n):
        squin.gate.cx(q[i - 1], q[i])

    return squin.qubit.measure(q)
```

Here are [some more examples](./tutorials_index.md).
