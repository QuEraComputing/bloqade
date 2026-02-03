# Tutorials on digital circuits

In this section you will find a number of tutorials and examples that show how you can use the digital bloqade subpackage, `bloqade-circuit`, in order to write quantum programs.
The examples are split into sub-sections featuring the different [dialects](./dialects_and_kernels) and submodules.

## General tutorials

<div class="grid cards style=font-size:1px;" markdown>

-   [Circuits with Bloqade](../tutorials/circuits_with_bloqade/)

    ---

    Learn how to use `bloqade-circuit` to write your quantum programs.


-   [Automatic Parallelism](../tutorials/auto_parallelism/)

    ---

    Explore the benefits of parallelizing your circuits.

</div>


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
