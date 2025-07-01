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

## Composability of kernels

WIP

### Qubits as argument to the kernel function

### Qubits as return value from the kernel
