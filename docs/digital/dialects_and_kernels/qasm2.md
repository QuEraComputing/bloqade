---
title: QASM2
---

# Open Quantum Assembly Language and beyond

We have chosen to closely mirror the semantics of the Open Quantum Assembly Language (QASM2) in bloqade-circuits.
For details on the language, see the [specification](https://arxiv.org/abs/1707.03429).

## qasm2.main

This dialect allows you to write native QASM2 programs, with all its features and restricitions.
As such, it includes definitions for gates, measurements and registers (quantum and classical), which are part of the QASM2 specification.

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

You can also look at the QASM2 program this kernel represents by emitting QASM2 code from it:

```python
from bloqade.qasm2.emit import QASM2
from bloqade.qasm2.parse import pprint


target = QASM2()
qasm2_program = target.emit(main)
pprint(qasm2_program)
```


## qasm2.extended

The QASM2 dialect is a simple quantum assembly language that allows you to write quantum circuits in a human-readable format. However, one should note that QASM2 is a very restricted language and does not support all the features of a high-level language.

For example, there is a separation of **gate routines** declared with `gate` and main program written as a sequence of gate applications. While the gate routine is similar to a function in many ways, it does not support high-level features such as recursion (due to lack of `if` statement support inside) or control flows.

Indeed, `bloqade-circuit` is designed with the notion of [kernels](https://queracomputing.github.io/kirin/latest/blog/2025/02/28/introducing-kirin-a-new-open-source-software-development-tool-for-fault-tolerant-quantum-computing/?h=kernel#what-are-kernel-functions) in mind by decorating functions with a `@qasm2.extended` decorator. The Python code is interpreted and parsed by the [Kirin](https://queracomputing.github.io/kirin/latest/) compiler toolchain and lowered to an abstract representation of the program. These kernels can include classical computation and the usual programming structures-- if/else, for and while loops, function inputs, and the like, as one is used to in Python.

Additionally, the QASM2 representations of bloqade-circuits have been extended to include a key advantage of reconfigurable neutral atom hardware: parallelism. For example, one can represent a CZ gate applied to many qubit pairs at once as

```python
from bloqade import qasm2
from kirin.dialects import ilist
from typing import Any

@qasm2.extended
def parallel_cz(controls: ilist.IList[qasm2.Qubit, Any], targets: ilist.IList[qasm2.Qubit, Any]):
    for ctr in range(len(controls)):
        qasm2.cz(ctrl=controls[0],qarg=targets[1])
```

or equivalently use a SIMD (Single Instruction Multiple Data)-like instruction to explicitly flag the parallelism

```python
@qasm2.extended
def simd_cz(controls: ilist.IList[qasm2.Qubit, Any], targets: ilist.IList[qasm2.Qubit, Any]):
    qasm2.parallel.cz(ctrls=controls,qargs=targets)
```

Both will ultimately emit the exact same QASM code, but the latter snippet represents the kind of parallelism that can be leveraged by reconfigurable neutral atom hardware to more efficiently execute a program.

!!! warning
    Since `qasm2.extended` has more advanced features that QASM2 in general, it is not always possible to emit a valid QASM2 program from a `qasm2.extended` kernel.
    You have to make sure that the control flow is simple enough it can be unrolled. See below for an example of such a case.
    Alternatively, a sure-fire, but restrictive, way is to stick to writing your kernel using `qasm2.main`.


### Quick Example

You can program kernels and quantum programs using the `qasm2.extended` decorator, such as the following Quantum Fourier Transform (QFT) circuit:

```python
import math
from bloqade import qasm2

@qasm2.extended
def qft(qreg: qasm2.QReg, n: int, k: int):
    if k == n:
        return qreg

    qasm2.h(qreg[k])
    for i in range(k + 1, n):
        qasm2.cu1(qreg[i], qreg[k], math.pi / 2**i)
    qft(qreg, n, k + 1)  # recursion
    return qreg

qft.print()
```

While the syntax is similar to Python, the `qasm2.extended` decorator actually compiles the `qft` function
into lower-level Intermediate Representation (IR) code that can be later interpreted, analyzed, or executed on quantum hardware. Observe that this function cannot immediately compile down to QASM as it takes parametrized inputs, and is called recursively.

You can inspect the initial IR code by calling the pretty printer:

```python
qft.print()
```

![QFT IR](qft-pprint.png)

We can also emit QASM2 code from it.
Note that native QASM2 does not support arguments or return values.
Therefore, we wrap the `qft` kernel from above in another one, that simply invokes `qft` for a specific set of arguments.
Then, we emit this new kernel as a QASM2 program.

```python
from bloqade.qasm2.emit import QASM2 # the QASM2 target
from bloqade.qasm2.parse import pprint # the QASM2 pretty printer

# NOTE: we wrap the qft kernel calling it with a set of arguments
@qasm2.extended
def main():
    n = 3
    q = qasm2.qreg(n)
    qft(q, n, 0)

target = QASM2()
ast = target.emit(main)
pprint(ast)
```

![QFT QASM2](qft-qasm2.png)


## Noise

You can represent different noise processes in your QASM2 kernel.
As of now, there are essentially two different noise channels:

* A Pauli noise channel, which can represent different types of decoherence.
* An atomic loss channel, which can be used to model effects of losing a qubit during the execution of a program.

Usually, you don't want to write noise statements directly.
Instead, use a [NoisePass][bloqade.qasm2.passes.NoisePass] in order to inject noise statements automatically according to a specific noise model.

!!! note
    Only the `qasm2.extended` dialect supports noise.

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
