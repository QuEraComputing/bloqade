# Dialects and kernels

!!! info
    A **kernel** function is a piece of code that runs on specialized hardware such as a quantum computer.

    A **dialect** is a domain-specific language (DSL) with which you can write such a kernel.
    Each dialect comes with a specific set of statements and instructions you can use in order to write your program.

Bloqade provides a set of pre-defined dialects, with which you can write your programs and circuits.

Once you have your kernel, you can inspect their intermediate representation (IR), apply different optimizations using [compiler passes](../quick_start/circuits/compiler_passes/index.md), or run them on a [(simulator) device](./simulator_device/simulator_device.md).


# Available dialects

Here's a quick overview of the most important available dialects.

## qasm2

There are a number of dialects with which you can write kernels that represent QASM2 programs.
See also the [qasm2 API reference](../reference/qasm2.md)

### qasm2.main

This dialect allows you to write native QASM2 programs.
As such, it includes definitions gates, measurements and quantum and classical registers, which are part of the QASM2 specification.
For details on the language, see the [specification](https://arxiv.org/abs/1707.03429).

Here's an example kernel

```python
from bloqade import qasm2

@qasm2.main
def main():
    q = qasm2.qreg(2)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])

    c = qasm2.creg(2)
    qasm2.measure(q, c)
    return c
```

Here's how you can look at the QASM2 program this kernel represents:

```python
from bloqade.qasm2.emit import QASM2
from bloqade.qasm2.parse import pprint


target = QASM2()
qasm2_program = target.emit(main)
pprint(qasm2_program)
```

### qasm2.extended

This dialect can also be used to write QASM2 programs.
However, it adds a couple of statements that makes it easier to write programs.
For example, QASM2 does not support for-loops.
With `qasm2.extended`, however, you can use for-loops and can let the compiler worry about unrolling these loops such that valid QASM2 code is produced.

```python
from bloqade import qasm2

@qasm2.extended
def main():
    n = 2
    q = qasm2.qreg(n)

    for i in range(n):
        qasm2.h(q[i])

    qasm2.cx(q[0], q[1])
    c = qasm2.creg(n)
    qasm2.measure(q, c)
    return c
```

If you run this through the code emission as shown above, you'll see that the for-loop gets unrolled into separate hadamard gate applications for each qubit.
At the same time, if you try to define this kernel using the `qasm2.main` dialect only, you will receive a `BuildError` telling you to take that crazy for-loop out of there as it's not supported.


## noise.native

Using this dialect, you can represent different noise processes in your kernel.
As of now, there are essentially two different noise channels:

* A pauli noise channel, which can represent different types of decoherence.
* An atomic loss channel, which can be used to model effects of losing a qubit during the execution of a program.

Usually, you don't want to write noise statements directly.
Instead, use a [NoisePass][bloqade.qasm2.passes.NoisePass] in order to inject noise statements automatically according to a specific noise model.

!!! note
    Right now, only the `qasm2.extended` dialect fully support noise.

For example, you may want to do something like this:

```python
from bloqade import qasm2
from bloqade.qasm2.passes import NoisePass

@qasm2.extended
def main():
    n = 2
    q = qasm2.qreg(n)

    for i in range(n):
        qasm2.h(q[i])

    qasm2.cx(q[0], q[1])
    c = qasm2.creg(n)
    qasm2.measure(q, c)
    return c

# Define the noise pass you want to use
noise_pass = NoisePass(main.dialects)  # just use the default noise model for now

# Inject the noise - note that the main method will be updated in-place
noise_pass(main)

# Look at the IR and all the glorious noise in there
main.print()
```


## squin

This dialect is, in a sense, more expressive than the qasm2 dialects: it allows you to specify operators rather than just gate applications.
That can be useful if you're trying to e.g. simulate a Hamiltonian time evolution.

!!! warning
    The squin dialect is in an early stage of development.
    Expect substantial changes to it in the near future.

Here's a short example:

```python
from bloqade import squin

@squin.kernel
def main():
    q = squin.qubit.new(2)
    h = squin.op.h()

    # apply a hadamard to only the first qubit
    h1 = squin.op.kron(h, squin.op.identity(sites=1))

    squin.qubit.apply(h1, q)

    cx = squin.op.cx()
    squin.qubit.apply(cx, q)

    return squin.qubit.measure(q)

# have a look at the IR
main.print()
```

See also the [squin API reference](../reference/squin.md)

## stim

!!! warning
    Sorry folks, still under construction.

See also the [stim API reference](../reference/stim.md)
