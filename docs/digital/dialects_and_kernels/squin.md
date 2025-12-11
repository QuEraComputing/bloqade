---
title: SQUIN
---

# Structural Quantum Instructions dialect

This dialect constitutes the central domain-specific language used in bloqade-circuit.
It allows you define your program in terms of gates applied to qubits, adding powerful control flow, such as `for` loops.

## Squin overview

The SQUIN DSL consists of three sub-groups of dialects:

* `squin.qubit`, which can be used for manipulating qubits via gate applications and measurements.
* `squin.gate`, which defines a set of gates that can be applied to qubits.
* `squin.noise`, which defines noise channels applied to qubits.

## Standard library for gate applications

Gates are exported using a standard library directly under the `squin` namespace.
This allows you to write programs in a concise way.
Here's a short example:

```python
from bloqade import squin

@squin.kernel
def main():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return squin.broadcast.measure(q)

# have a look at the IR
main.print()
```

The resulting IR looks like this:

![main IR](./squin-ir-1.png)

As you can see, calls such as `squin.h(q[0])` are lowered as `func.invoke` statements, i.e. to function calls into the standard library of SQUIN.

For a complete list of all available gates, please see the [API reference](../../../reference/bloqade-circuit/src/bloqade/squin/stdlib/simple/gate/).


## Using control flow

One of the central aspects of SQUIN is that you are also able to use standard control flow such as for loops.

For example, we can generalize the two-qubit GHZ kernel function from above to an arbitrary numbers of qubits:

```python
from bloqade import squin

@squin.kernel
def ghz(n: int):
    q = squin.qalloc(n)

    squin.h(q[0])
    for i in range(n - 1):
        squin.cx(q[i], q[i + 1])

```

Note that the fact that gate applications are represented by `func.invoke` also shows that it's possible to call user-defined kernel functions in SQUIN!

For example, we could split the above program into two steps

```python
from bloqade import squin
from bloqade.types import Qubit

from kirin.dialects import ilist
from typing import Any

@squin.kernel
def allocate_qubits_for_ghz(n: int) -> ilist.IList[Qubit, Any]:
    q = squin.qalloc(n)
    squin.h(q[0])
    return q

@squin.kernel
def ghz_split(n: int):
    q = allocate_qubits_for_ghz(n)
    for i in range(n - 1):
        squin.cx(q[i], q[i + 1])
```

## Noise

The squin dialect also includes noise, with a fixed set of noise channels defined.
Just like gates, they are exported under the `squin` namespace.

For example, we can use this to add noise into the simple kernel from before, which entangles two qubits:

```python
from bloqade import squin

@squin.kernel
def main_noisy():
    q = squin.qalloc(2)

    squin.h(q[0])
    squin.depolarize(p=0.1, qubit=q[0])

    squin.cx(q[0], q[1])
    squin.depolarize2(0.05, q[0], q[1])

    return squin.broadcast.measure(q)

# have a look at the IR
main_noisy.print()
```

The result looks like this:

![main_noisy IR](./squin-ir-2.png)

Note, that you could equivalently write the depolarization error in the above as

```python
dpl = squin.noise.depolarize(p=0.1)
squin.qubit.apply(dpl, q[0])
```

A full list of available noise channels can be found in the [API reference](../../../reference/bloqade-circuit/src/bloqade/squin/stdlib/simple/noise/).


## Parallelizing gate applications

There is also a standard library available for broadcasting gates, i.e. applying a gate to multiple qubits in parallel.
For example, the following kernel functions apply the same operations to qubits:

```python
from bloqade import squin

@squin.kernel
def sequential():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.h(q[1])


@squin.kernel
def parallel():
    q = squin.qalloc(2)
    squin.broadcast.h(q)

```

Note that noise can also be parallelized, e.g. by calling `squin.broadcast.depolarize(0.1, q)`.

See the [API reference for broadcast](../../../reference/bloqade-circuit/src/bloqade/squin/stdlib/broadcast) for all the available functionality.
Note that it will be precisely the same functions as for the standard gate application, but applied to lists of qubits rather than single ones.

## See also
* [Tutorial: Circuits with Bloqade](../../tutorials/circuits_with_bloqade/)
* [SQUIN API reference](../../../reference/bloqade-circuit/src/bloqade/squin/)
* [Examples & Tutorials](../../examples/)
