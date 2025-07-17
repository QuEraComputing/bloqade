# Converting squin to Cirq

You can convert a squin kernel function to a `cirq.Circuit` object.
The output circuit will feature gates that most closely resemble the kernel you put in.

## Basic usage

You can obtain a circuit using the `squin.cirq.emit_circuit` function.

```python
from bloqade import squin

@squin.kernel
def main():
    q = squin.qubit.new(2)
    h = squin.op.h()
    squin.qubit.apply(h, q[0])
    cx = squin.op.cx()
    squin.qubit.apply(cx, q[0], q[1])
    squin.qubit.measure(q)

circuit = squin.cirq.emit_circuit(main)
print(circuit)
```

There is one crucial difference between a squin kernel and a cirq circuit:
the qubits are defined inside a kernel, whereas for a circuit they are defined outside.

The default behavior here is to emit a set of `cirq.LineQubit`, which is of the correct length.
They will be sorted by their `Qid` (position) according to the order they appear in the kernel.

## Customizing qubits

By default, a set of `cirq.LineQubit`s of the appropriate size is created internally, on which the resulting circuit operates.
This may be undesirable sometimes, e.g. when you want to combine multiple circuits or if you want to have qubits of a different type.

To allow modifications here, you can simply pass in a list of qubits (a sequence of `cirq.Qid`s) into the emit function.

```python
import cirq

qubits = cirq.GridQubit.rect(rows=1, cols=2)
circuit = squin.cirq.emit_circuit(main, qubits=qubits)
print(circuit)
```

Note, that the qubits will be used in the resulting circuit in the order they appear in `squin.qubit.new` statements.

!!! warning

    When passing in a list of qubits, you need to make sure there is sufficiently many qubits.
    Otherwise, you may get indexing errors.

## Limitations

Please note that there are some limitations, especially regarding control flow.
Using `if` statements or loops inside a kernel function may lead to errors.

If you run into an issue that you think should be supported, please [report an issue on the GitHub repository](https://github.com/QuEraComputing/bloqade-circuit/issues).
