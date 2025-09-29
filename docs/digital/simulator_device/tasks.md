# Tasks

Tasks are generally executed by [devices](./simulator_device.md).
A task can be run locally or remotely.
On the one hand, when running locally, your local machine will execute the task at hand and wait for the result.
On the other hand, when a task is submitted to be executed on a remote device, you will obtain an object similar to a future object in async programming, which can await the result (or not).

In order to interact with a task, you'll usually want to instantiate a device and create a new task on that device.
For example,

```python
from bloqade.pyqrack import StackMemorySimulator
from bloqade import squin

@squin.kernel
def main():
    q = squin.qubit.new(2)

    squin.gate.h(q[0])
    squin.gate.cx(q[0], q[1])

    return q

sim = StackMemorySimulator(min_qubits=2)
task = sim.task(main)
result = task.run()
```

!!! info
    Most methods that directly execute a kernel on a device are just wrappers for the above:
    a new task is created internally and then run on the device.


!!! warning "Note"
    Currently, there are only local simulation devices available.
    However, in the near future, you will also be able to submit tasks to a remote machine and even actual quantum hardware.
