# Converting cirq to squin

If you want to obtain a squin kernel from a circuit, you can use the `load_circuit` method in the `squin.cirq` submodule.
What you're effectively doing is lowering a circuit to a squin IR.
This IR can then be further lowered to eventually run on hardware.

## Basic examples

Here are some basic usage examples to help you get started.

```python
from bloqade import squin
import cirq

qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CX(qubits[0], qubits[1]),
    cirq.measure(qubits)
)

# let's have a look
print(circuit)

main_loaded = squin.cirq.load_circuit(circuit, kernel_name="main_loaded")
```

The above is equivalent to writing the following kernel function yourself:

```python
@squin.kernel
def main():
    q = squin.qubit.new(2)
    H = squin.op.h()
    CX = squin.op.cx()
    squin.qubit.apply(H, q[0])
    squin.qubit.apply(CX, q)
    squin.qubit.measure(q)
```

You can further inspect the lowered kernel as usual, e.g. by printing the IR.
Let's compare the manually written version and the loaded version:

```python
main.print()
main_loaded.print()
```

The resulting IR is equivalent, yet the loaded is a bit longer since the automated loading can make fewer assumptions about the code.
Still, you can use the kernel as any other, e.g. by calling it from another kernel or running it via a simulator.

## Noise

Lowering a noisy circuit to squin is also supported.
All common channels in cirq will be lowered to an equivalent noise statement in squin.

```python
from bloqade import squin
import cirq

qubits = cirq.LineQubit.range(2)
noisy_circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CX(qubits[0], qubits[1]),
    cirq.depolarize(p=0.01).on_each(qubits),
)

# let's have a look
print(noisy_circuit)

noisy_kernel = squin.cirq.load_circuit(noisy_circuit)
noisy_kernel.print()
```

This becomes especially useful when used together with a `cirq.NoiseModel` that automatically adds noise to a circuit via `circuit.with_noise(model)`.

## Composability of kernels

You may also run into a situation, where you define a circuit that is used as part of a larger one, maybe even multiple times.
In order to allow you to do something similar here, you can pass in and / or return the qubit register in a loaded kernel.
Both these options are controlled by simple keyword arguments.

### Qubits as argument to the kernel function

Setting `register_as_argument=True` when loading a kernel, will result in a squin kernel function that accepts (and requires) a single argument of type `IList[Qubit]`.
This means you can use a loaded circuit as part of another kernel function.
Check it out:

```python
from bloqade import squin
import cirq

qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CX(qubits[0], qubits[1]),
)

sub_kernel = squin.cirq.load_circuit(circuit, register_as_argument=True, kernel_name="sub_kernel")


@squin.kernel
def main():
    q = squin.qubit.new(4)

    # entangle qubits 1 and 2
    sub_kernel([q[0], q[1]])

    # entangle qubits 3 and 4
    sub_kernel([q[2], q[3]])


main.print()
```

Looking at the IR of the resulting kernel, you can see that there is are `invoke sub_kernel` statements present, which call the lowered circuit with the given arguments.

### Qubits as return value from the kernel

Similarly to above, you may also want to return a list of qubits from a loaded kernel.
Let's adapt the above to instantiate and return a pair of entangled qubits using the same circuit:

```python

sub_kernel = squin.cirq.load_circuit(circuit, return_register=True, kernel_name="sub_kernel")

@squin.kernel
def main():
    # instantiate and entangle a list of two qubits
    q1 = sub_kernel()

    # do it again, to get another set
    q2 = sub_kernel()

    # now we have 4 qubits to work with
    ...

main.print()
```


!!! note
    You can also mix both options by setting `register_as_argument = True` and `return_register = True` in order to obtain a kernel function that both accepts and returns a list of qubits.


## Limitations

There are some limitations when loading circuits.
One, for example, is that custom gates are not supported as you can't generally know how to lower them to a squin statement.

If you find a missing feature, please feel free to [open a GitHub issue](https://github.com/QuEraComputing/bloqade-circuit/issues).
