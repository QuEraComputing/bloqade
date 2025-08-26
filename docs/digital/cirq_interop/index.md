# Interoperability with cirq

The [Cirq](https://quantumai.google/cirq) framework is a powerful tool for writing quantum circuits targeting near-term devices.
Instead of reinventing the wheel, Bloqade offers convenient interoperability with Cirq that allows you to jointly use both libraries in order to develop your quantum program.

Specifically, you can turn a [`cirq.Circuit`](https://quantumai.google/reference/python/cirq/Circuit) object into a [squin](../dialects_and_kernels.md#squin) kernel function and vice versa.

For details on each of these, please see the documentation pages below:

* [Obtaining a squin kernel function from a `cirq.Circuit`](./cirq_to_squin.md)
* [Emitting a `cirq.Circuit` from a squin kernel](./squin_to_cirq.md)

For the API reference, please see the `cirq` submodule in the [squin API docs](../../reference/bloqade-circuit/src/bloqade/squin.md).

## TL;DR

Here's a short example:

```python
from bloqade import squin
import cirq

q = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q[0]),
    cirq.CX(q[0], q[1])
)
print(circuit)

main = squin.cirq.load_circuit(circuit)
main.print()

roundtrip_circuit = squin.cirq.emit_circuit(main)
print(roundtrip_circuit)
```
