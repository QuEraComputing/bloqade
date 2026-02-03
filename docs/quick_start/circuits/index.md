# Digital Quantum Computing with bloqade

This section provides the quick start guide for developing quantum programs represented by circuits using Bloqade. Circuits are a general-purpose and powerful way of representing arbitrary computations. For a few examples please refer to our [examples](../../digital/index.md).

## Pick your frontend: choose a DSL

bloqade-circuit provides a number of different [domain specific languages (DSLs)](../../digital/dialects_and_kernels/) for writing quantum programs.
If you are unsure which one to choose, head over to the [DSL documentation](../../digital/dialects_and_kernels/) for an overview of all available ones.

If you are looking to write a circuit, we recommend giving [SQUIN](../../digital/dialects_and_kernels/#squin) a go.
Here's an example of how you would write a simple GHZ preparation circuit:

```python
from bloqade import squin

@squin.kernel
def ghz(n: int):
    q = squin.qalloc(n)
    squin.h(q[0])
    for i in range(1, n):
        squin.cx(q[i - 1], q[i])
```

One of the features here is that the SQUIN DSL support control flow, such as for loops, which allows you to write your programs in a concise way.
At some point, before execution on hardware, such a loop will have to be unrolled.
However, you can let the compiler worry about that and use it as a high-level feature.


## Optimize your program

!!! note
    This step is optional and you may just skip ahead to choosing your backend.

When you define a program, such as the one above, it creates an intermediate representation (IR) of that program.
In the above, since `ghz` is annotated with the `@squin.kernel` decorator, it is not a function, but a `Method` object that stores the IR of the GHZ program.

You can run different optimizations and compiler passes on your IR in order to tailor your program to run optimally on the chosen backend.

While it is possible to write your own compiler passes and optimizations - for that, please refer to the [kirin](https://queracomputing.github.io/kirin/latest/) documentation - bloqade-circuit also offers a number of different, pre-defined optimizations.

!!! warning
    Compiler and optimization passes are currently under development.
    While quite a lot of them are used internally, they are not in a user-friendly state.
    Please skip this step for the time being.

## Pick your backend: simulation and hardware

Once you have your program written and optimized to a point at which you are satisfied, it is time to think about execution.


### Simulation with PyQrack

In order to simulate your quantum program, bloqade-circuit integrates with the [Qrack](https://pyqrack.readthedocs.io/en/latest/) simulator via its Python bindings.
Let's run a simulation of the above GHZ program:

```python
from bloqade.pyqrack import StackMemorySimulator
sim = StackMemorySimulator(min_qubits=4)
sim.run(ghz, args=(4,))  # need to pass in function arguments separately
```

There are also some things available in the simulator which cannot be obtained when running on hardware, such as the actual state vector of the system:

```python
sim.state_vector(ghz, args=(4,))
```

### Simulation with STIM and TSIM


For QEC workflows, it may be required to sample millions or billions of shots from the same kernel.
For this, bloqade-circuit provides tight integration with the [STIM](https://github.com/quantumlib/Stim) and [TSIM](https://github.com/QuEraComputing/tsim) libraries.

STIM is a sampling simulator for Clifford circuits. TSIM is a sampling simulator for universal quantum circuits
that contain few non-Clifford gates. Both simulators support Pauli noise channels. TSIM optionally provides GPU acceleration. For more information, please refer to the [TSIM documentation](https://queracomputing.github.io/tsim/latest/).

To use these simulators, first instantiate a `Circuit` object from your kernel. Then compile the circuit into a sampler. This step enables efficient sampling of millions or billions of shots:
```python
from bloqade.tsim import Circuit

@squin.kernel
def main():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.t(q[0])
    squin.broadcast.depolarize(0.01, q)
    squin.cx(q[0], q[1])
    bits = squin.broadcast.measure(q)
    squin.set_detector(bits, coordinates=[0, 1])
    squin.set_observable([bits[0]], idx=0)


circuit = Circuit(main)
sampler = circuit.compile_sampler()
sampler.sample(shots=1_000_000, batch_size=100_000)  # On GPU, large batch size improves performance
```

TSIM and STIM provide two types of samplers that are created via the `compile_sampler` and `compile_detector_sampler` methods, respectively. The regular sampler ignores any `set_detector` and `set_observable` statements and returns measurement bits for each `measure` instruction in the order
of measurement. The detector sampler samples detector and observable bits.

```python
from bloqade.stim import Circuit

circuit = Circuit(main)
sampler = circuit.compile_detector_sampler()
detector_bits, observable_bits = sampler.sample(shots=1_000_000, separate_observables=True)

```
A detailed tutorial of how to use TSIM for QEC workflows is available [here](https://bloqade.quera.com/latest/digital/examples/tsim/magic_state_distillation/).


### Hardware execution

!!! note
    We're all very excited for this part, but we will have to wait just a bit longer for it to become available.
    Stay tuned!


## Further reading and examples

For more details on domain specific languages available in bloqade-circuits, please refer to the [dedicated documentation section on dialects](../../digital/dialects_and_kernels/).
We also recommend that you check out our [collection of examples](../../digital/examples/), where we show some more advanced usage examples.

There is also some more documentation available on the [PyQrack simulation backend](../../digital/simulator_device/simulator_device.md).

Finally, if you want to learn more about compilation and compiler passes, please refer to [this documentation page](../../digital/compilation.md).
We also highly recommend that you have a look at the [kirin framework](https://queracomputing.github.io/kirin/latest/).
