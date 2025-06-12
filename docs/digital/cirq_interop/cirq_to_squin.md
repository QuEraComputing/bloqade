# Converting cirq to squin

If you want to obtain a squin kernel from a circuit, you can use the `load_circuit` method in the `squin.cirq` submodule.
What you're effectively doing is lowering a circuit to a squin IR.
This IR can then be further lowered to eventually run on hardware.

## Noise

## Composability of kernels

WIP

### Qubits as argument to the kernel function

### Qubits as return value from the kernel
