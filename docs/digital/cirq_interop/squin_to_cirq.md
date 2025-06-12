# Converting squin to cirq

## Customizing qubits

By default, a set of `cirq.LineQubit`s of the appropriate size is created internally, on which the resulting circuit operates.
This may be undesirable sometimes, e.g. when you want to combine multiple circuits or if you want to have qubits of a different type.

TODO
