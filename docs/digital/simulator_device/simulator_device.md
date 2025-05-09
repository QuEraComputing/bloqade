# Simulation devices

A simulation device can run a [task](./tasks.md), such as executing a kernel.
It acts just like a device that is an actual hardware, but runs everything in a local simulation.
As such, it can also be used to inspect the results of your program beyond what is possible on a QPU.
For example, you can return the `state_vector` of the quantum register at the end of the task execution.

Here's how you can use it in order to run a simple `qasm2.extended` kernel.

```python
from bloqade.pyqrack import StackMemorySimulator
from bloqade import qasm2

@qasm2.extended
def main():
    q = qasm2.qreg(2)

    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])

    return q

sim = StackMemorySimulator(min_qubits=2)

# get the state vector -- oohh entanglement
state = sim.state_vector(main)
```
