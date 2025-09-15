# Dialects and kernels

Bloqade provides a set of pre-defined domain specific languages (DSLs), with which you can write your programs and circuits.
We call these DSLs *dialects*.
For a list of available dialects [see blow](#available-dialects).

Once you have defined your kernel, you can inspect their Intermediate Representation (IR), apply different optimizations using compiler passes, or run them on a [(simulator) device](../simulator_device/simulator_device.md).

!!! info "Kernels & dialects in a nutshell"
    A **kernel** function is a piece of code that runs on specialized hardware such as a quantum computer or a GPU.

    A **dialect** is a domain-specific language (DSL) with which you can write such a kernel.
    Each dialect comes with a specific set of statements and instructions you can use in order to write your program.


When running code that targets a specialized execution environment, there are typically several layers involved.
At the surface, the programmer writes functions in a syntax that may resemble a host language (e.g., Python), but is actually expressed in a dialect — a domain-specific variant with its own semantics.
A decorator marks these functions so they can be intercepted before normal host-language execution.
All dialects can be used by decorating a function.

!!! info "Primer on Python decorators"
    A decorator in Python is simply a function (or any callable really) that takes in another function as argument and returns yet another function (callable).
    Usually, the returned function will be a modified version of the input.
    Decorators are used with the `@` syntax.


Instead of running directly, the kernel function body is parsed and translated (lowered) into an intermediate representation (IR).
This IR can be manipulated (e.g. to perform optimizations) and can later be executed by an interpreter that understands the dialect's semantics.
The interpreter uses an internal instruction set to execute the code on the intended backend, which may be a simulator, virtual machine, or physical device.
This separation lets developers write high-level, expressive code while the interpreter ensures it runs correctly in the target environment.
[QuEra's Kirin](https://queracomputing.github.io/kirin/latest/) provides the infrastructure that allows us to define custom dialects tailored towards the needs of programming neutral atom quantum computers in Bloqade
While the dialects are not Python syntax, Kirin still uses the Python interpreter to execute the code.


!!! warning "Note"
    It is important to understand that when you are writing a kernel function in a dialect you are generally **not writing Python** code, even though it looks a lot like it.
    Therefore, kernel functions are not (usually) directly callable.
    Think of this as trying to execute another programming language with the Python interpreter: of course, that will error.


# Available dialects

Bloqade offers a few different dialects with which you can write your programs.
All dialects have some advantages for particular applications.

If you are unsure which dialect best suits your needs, have a look at the high-level overview of the (non-exhaustive) list of use cases below.
Also, we recommend having a look at [the Structural QUantum INstructions (SQUIN) dialect](./squin.md) as it is the most general purpose dialect available and is centrally used in the compilation pipeline.

While the documentation in this section provides some information on the background and a high-level overview, it is also often convenient to learn from examples.
Have a look at the (growing) [examples collection](../examples/), where you can find different implementations of quantum programs using different dialects.


## [squin](./squin.md)

With this dialect, you can structure your program in terms of operators rather than gate applications, or define circuits.
Have a look at [the dedicated documentation page](./squin.md) and the corresponding [API reference](../../reference/bloqade-circuit/src/bloqade/squin/).

**Use cases**:

* Writing a program that represents a circuit.
* If you require control flow (loops and if-statements, ...) and composability (function definitions, recursion, ...).
* Simulation including noise.
* Hamiltonian simulation (work in progress).


## [qasm2](./qasm2.md)

There are a number of dialects with which you can write kernels that represent programs in the Quantum Assembly Language (QASM2).
More details can be found [here](./qasm2.md).
Also, have a look at the full [qasm2 API reference](../../reference/bloqade-circuit/src/bloqade/qasm2/)

**Use cases**:

* Write circuits using the QASM2 standard.
* Composability with other tools that integrate with QASM2, but not with bloqade directly.
* Control flow via the extended dialect (not always compatible with native QASM2).


## [stim](./stim.md)

For quantum error correction applications, you may want to use this dialect.
See this [documentation page](./stim.md) for more details.
The full API documentation is [available here](../../reference/bloqade-circuit/src/bloqade/stim/).

**Use cases**:

* Quantum error correction.
* Stabilizer codes.
