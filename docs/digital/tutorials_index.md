# Tutorials on digital circuits

In this section you will find a number of tutorials that show how you can use the digital bloqade subpackage, `bloqade-circuit`, in order to write quantum programs.
The tutorials are split into sub-sections featuring the different [dialects](./dialects_and_kernels.md) and submodules.


## Squin

Squin is bloqade-circuits central dialect used to build circuits and run them on simulators and hardware.

<div class="grid cards style=font-size:1px;" markdown>

-   [Deutsch-Jozsa Algorithm](../examples/squin/deutsch_squin/)

    ---

    See how you can implement the fundamental Deutsch-Jozsa algorithm with a Squin kernel function.


-   [GHZ state preparation and noise](../examples/squin/ghz/)

    ---

    Inject noise manually in a simple squin kernel.


</div>


## Interoperability with other SDKs

While bloqade-circuit provides a number of different dialects (eDSLs), it may also be convenient to transpile circuits written using other SDKs.

<div class="grid cards style=font-size:1px;" markdown>

-   [Heuristic noise models applied to GHZ state preparation](../examples/interop/noisy_ghz/)

    ---

    Learn how to apply our heuristic noise models built to work with the cirq SDK.

</div>


## QASM2

One of the most central languages used to define quantum programs is QASM2.
You can also write your quantum programs using the QASM2 dialect directly in bloqade-circuit.

!!! warning

    Some of the examples below use the `qasm2.extended` dialect, which adds more advanced language features, such as control flow.
    However, this dialect is deprecated and we recommend using `squin` instead.



<div class="grid cards style=font-size:1px;" markdown>

-   [Quantum Fourier Transform](../examples/qasm2/qft/)

    ---

    An example showing how to implement the well-known Quantum Fourier Transform (QFT).

-   [GHZ Preparation and Parallelism](../examples/qasm2/ghz/)

    ---

    Learn how to use parallelism to reduce the circuit (execution) depth.

-   [Pauli Exponentiation for Quantum Simulation](../examples/qasm2/pauli_exponentiation/)

    ---

    Simulating Hamiltonian dynamics by exponentiating Pauli operators.


-   [Repeat until success with STAR gadget](../examples/qasm2/repeat_until_success/)

    ---

    Here's how to implement a Z phase gate with the repeat-until-success protocol.

</div>
