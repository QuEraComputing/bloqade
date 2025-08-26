---
title: SQUIN
---

# Structural Quantum Instructions dialect

This dialect is, in a sense, more expressive than the qasm2 dialects: it allows you to specify operators rather than just gate applications.
That can be useful if you're trying to e.g. simulate a Hamiltonian time evolution.

That said, gate applications also have short-hand standard library definitions defined in the `squin.gate` submodule.
So you can also just write a squin program like you would a quantum circuit.
Here's a short example:

```python
from bloqade import squin

@squin.kernel
def main():
    q = squin.qubit.new(2)
    squin.gate.h(q[0])
    squin.gate.cx(q[0], q[1])
    return squin.qubit.measure(q)

# have a look at the IR
main.print()
```

As mentioned above, you can also build up more complex "operators" that are then applied to any number of qubits.
To show how you can do that, here's an example on how to write the above kernel defining the gates as separate operators.
This isn't exactly a practical use-case, but serves as an example.

```python
from bloqade import squin

@squin.kernel
def main():
    q = squin.qubit.new(2)
    h = squin.op.h()

    # apply a hadamard to only the first qubit
    h1 = squin.op.kron(h, squin.op.identity(sites=1))

    squin.qubit.apply(h1, q[0], q[1])

    cx = squin.op.cx()
    squin.qubit.apply(cx, q[0], q[1])

    return squin.qubit.measure(q)

# have a look at the IR
main.print()
```

## Noise

The squin dialect also includes noise.
Each noise channel is again represented by an operator.
Therefore, you can separate the application of a noise channel from the qubits and do algebra on them.
These noise channel operators are available under the `squin.noise` module.
For example, you can create a depolarization channel with a set probability inside a kernel with `squin.noise.depolarize(p=0.1)`.

To make it easier to use if you are just writing a circuit, however, there is again a standard library for short-hand applications.
That standard library is called `squin.channel`. For example, we can use this to add noise into the simple kernel from before, which entangles two qubits:

```python
from bloqade import squin

@squin.kernel
def main_noisy():
    q = squin.qubit.new(2)

    squin.gate.h(q[0])
    squin.channel.depolarize(p=0.1, qubit=q[0])

    squin.gate.cx(q[0], q[1])
    squin.channel.depolarize2(0.05, q[0], q[1])

    return squin.qubit.measure(q)

# have a look at the IR
main_noisy.print()
```

## See also
* [squin API reference](../../../reference/bloqade-circuit/src/bloqade/squin/)
