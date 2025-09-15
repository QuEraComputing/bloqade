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
    q = squin.qubit.new(n)
    squin.gate.h(q[0])
    for i in range(1, n):
        squin.gate.cx(q[i - 1], q[i])
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
Bloqade Digital is a hardware-first SDK, which means that simulation tries to mirror execution on hardware as closely as possible.
Choosing the hardware you want to run on is therefore mostly interchangeable with simulator backends.

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
