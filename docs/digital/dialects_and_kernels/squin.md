---
title: SQUIN
---

# Structural Quantum Instructions dialect

This dialect is, in a sense, more expressive than the qasm2 dialects: it allows you to specify operators rather than just gate applications.
That can be useful if you're trying to e.g. simulate a Hamiltonian time evolution.

## Squin overview

The SQUIN DSL consists of three sub-groups of dialects:

* `squin.qubit`, which can be used for manipulating qubits via gate applications and measurements.
* `squin.op`, which is used to define gates as operators separately from qubits and perform algebraic operations on them.
* `squin.noise`, which defines noise channels as operators.

Furthermore, there are two standard library modules, which are mainly used for convenience:

* `squin.gate`, which combines `squin.op` and `squin.qubit`, so you can write gates as you would when defining a quantum circuit.
* `squin.channel`, which combines `squin.noise` operators and `squin.qubit` in a similar way.


## Operators: Separating Quantum Gates from Qubits

When you define a quantum circuit, you usually think about gates applied to a fixed number of qubits.
What this actually means in terms of the underlying physics is that a unitary operator (describing the time evolution of a Hamiltonian corresponding to the gate you want to apply) is applied to the qubits of interest.

In the SQUIN dialect, the notion of operators is introduced to reflect that lower level: you can define gates as operators, without applying them to a qubit right away.
Furthermore, you can perform algebraic operations on these operators, which will result in yet another operator that you can apply to qubits.

Since function calls are also supported in this DSL, you can define functions that build and return operators which you can later apply.

Here is a (somewhat artificial) example, that illustrates the flexibility of SQUIN: we can define a gate that uses two control qubits and applies the operator $X \otimes Y$ to two targets.


```python
from bloqade import squin

@squin.kernel
def ccxy():
    """Operator that uses two control qubits in order to apply X and Y to two distinct target qubits."""
    x = squin.op.x()
    y = squin.op.y()
    xy = squin.op.kron(x, y)
    return squin.op.control(xy, n_controls=2)
```

You can then call it from another kernel by just invoking the function

```python
@squin.kernel
def main():
    q = squin.qubit.new(4)
    op = ccxy()

    h = squin.op.h()

    # broadcast applies an operator in parallel to the list of qubits, in this case the first two
    squin.qubit.broadcast(h, [q[0], q[1]])

    # the first two qubits are used as controls
    squin.qubit.apply(op, q[0], q[1], q[2], q[3])
```


## Standard library for gate applications

While constructing operators is certainly powerful, most of the time you may want to simply apply standard quantum gates.
Fortunately, these can be represented as operators and are provided in SQUIN through the `squin.gate` library.

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

The resulting IR looks like this:

![main IR](./squin-ir-1.png)

## Noise

The squin dialect also includes noise, with each noise channel represented by an operator.
Therefore, you can separate the application of a noise channel from the qubits and do algebra on them.
These noise channel operators are available under the `squin.noise` module.
For example, you can create a depolarization channel with a set probability inside a kernel with `squin.noise.depolarize(p=0.1)`.

To make it easier to use if you are just writing a circuit, however, there is again a standard library for short-hand applications available through `squin.channel`.

For example, we can use this to add noise into the simple kernel from before, which entangles two qubits:

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

The result looks like this:

![main_noisy IR](./squin-ir-2.png)

Note, that you could equivalently write the depolarization error in the above as

```python
dpl = squin.noise.depolarize(p=0.1)
squin.qubit.apply(dpl, q[0])
```

## See also
* [squin API reference](../../../reference/bloqade-circuit/src/bloqade/squin/)
