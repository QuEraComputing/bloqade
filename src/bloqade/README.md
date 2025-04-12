# Surprise! Where are all the source codes?!

the `bloqade` repository is a Python namespace package[^1] [^2] that does not contain any source code. This repository only contains the examples and documentation for the `bloqade` package. This allows us to provide better installation experience for new users (they just install everything in bloqade universe), and also allows us to have a more modularized codebase.

See also the discussion in [#213](https://github.com/QuEraComputing/bloqade/issues/213) regarding the design decision of using namespace package vs. mono-repo.

## Sub-packages

When you install `bloqade`, you will get the following sub-packages:

### [`bloqade-circuit`](https://github.com/QuEraComputing/bloqade-circuit): the sub-package for quantum circuits.

You can install it with:

```bash
pip install bloqade-circuit
```

there are some extra you can choose to install:

- `qasm2`: features that allow you to convert QASM files or program with our extended QASM2 eDSL in Python.
- `vis`: features that allow you to use visualization.
- `qbraid`: features that allow you to use Qbraid.
- `cirq`: features that allow you to use Cirq.

For example, you can enable `qasm2` and `vis` by running:

```bash
pip install bloqade-circuit[qasm2,vis]
```

### [`bloqade-analog`](https://github.com/QuEraComputing/bloqade-circuit): the sub-package for analog quantum computing.

You can install it with:

```bash
pip install bloqade-analog
```

This is actually the older version of `bloqade` when we only have analog quantum devices. If you have older codebase just change `from bloqade import ...` to `from bloqade.analog import ...` and it should work.


### More to come!

We are working on one more sub-package for lower-level programming functionality of
neutral atom quantum computers. Stay tuned for more updates!

## References

[^1]: [PEP 420 – Implicit Namespace Packages](https://peps.python.org/pep-0420/)
[^2]: [Real Python: What's a Python Namespace Package, and What's It For?](https://realpython.com/python-namespace-package/)
