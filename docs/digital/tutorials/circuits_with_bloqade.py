# %% [markdown]
# # Circuits with Bloqade
#
# In this tutorial we will demonstrate how to write circuits and quantum executions with Bloqade. Specifically, we will use the `squin` dialect set from the compiler toolchain `kirin`. `SQUIN` stands for `S`tructural `Qu`antum `IN`struction set and is the circuit-level representation of quantum executions. It is built on top of the `kirin` framework, an [open-source compiler toolchain](https://queracomputing.github.io/kirin/latest/) for embedded domain-specific languages (eDSLs) that target scientific computing kernels. A key feature of squin is the _kernel_, which can roughly be seen as the object which will be executed on the target hardware. Naturally, this hardware could be a quantum computer, but it also extends to classical execution as well, such as mid-circuit feedforward or even non-quantum execution such as robotics.
#
# These kernels can be built using decorators of python functions. We will use the `@squin.kernel` decorator in this notebook but keep in mind that other eDSLs have different decorators inherited from base Kirin decorators. The decorator lowers Python's abstract syntax tree (AST) into a kirin SSA (single static assignment) form, which is a useful intermediate representation for compiler analysis. You don't have to worry too much about SSA or compilers here, but if you want to learn more check out the [kirin documentation](https://queracomputing.github.io/kirin/latest/).

# %%
from typing import Any

import numpy as np
from kirin.ir import Method

# Some types we will use, useful for type hints
from kirin.dialects.ilist import IList

import bloqade.types
from bloqade import squin
from bloqade.squin.types import MeasurementResult

Register = IList[bloqade.types.Qubit, Any]

# %%
@squin.kernel
def hello_world(theta: float) -> IList[MeasurementResult, Any]:
    """
    Prepare a Bell state and measure in a basis that might have a Bell violation
    """
    qubits = squin.qubit.new(2)
    squin.gate.h(qubits[0])
    squin.gate.cx(qubits[0], qubits[1])
    squin.gate.rx(theta, qubits[0])
    bits = squin.qubit.measure(qubits)
    return bits


# [kernel].print() prints the raw SSA, which is the intermediate representation of the kernel
# as used internally by Kirin.
hello_world.print()
# %% [markdown]
# ### Anatomy of a Bloqade kernel
#
# A kernel is a representation of a hybrid quantum/classical execution that will run "on hardware". The decorator @squin.kernel can be considered as the "label" to represent this fact. A kernel can take arguments (such as `theta` here) and can return values (such as `bits` here). To help the compiler, the inputs and outputs must be decorated with a type. The return values can be considered as the "results" which are returned from the quantum computer, though they can also be intermediate values that are used as a part of a larger computation (e.g. a function call). A kernel can represent both quantum and classical execution: while lots of classical computation is supported (control flow, algebra, etc.) there is less support for arbitrary python calls, as this won't necessarily be supported on the microcontroller "on hardware". Note that shot-level execution and averaging is handled outside of the kernel itself: averaging over multiple shots within a kernel is an anti-pattern to avoid.

# %% [markdown]
# ## Squin kernel statements
#
# To represent quantum executions, we write and construct squin kernels. These kernels can use the typical structure of python functions -- inputs, outputs, for loops, control flow, subroutines, and so forth -- as a feature of the underlying base Kirin statements. Because one can intermix control flow with operations and measurement, arbitrary mid-circuit feed forward comes for "free".
#
# There are four sets of statements which you can use to represent quantum programs:
# 1. `squin.qubit` - Manipulation and declaration of the qubits themselves.
# 2. `squin.gate` - Wrapper functions around operations which execute gates on qubits
# 3. `squin.op` - The wider class of quantum operations on qubits, on which to build custom gates
# 4. `squin.noise` - Representing noise effects on the qubits.
#
# The first set of expressions are the familiar representations of quantum gates, measurements, and qubits, under the `squin.qubit` and `squin.gate` namespaces. The gate namespace is a collection of common gates written as methods; [the gate API documentation is here](https://bloqade.quera.com/latest/reference/bloqade-circuit/src/bloqade/squin/stdlib/gate/).  The qubit namespace is used to create, measure, and apply operations to qubits; [the qubit API documentation is here](https://bloqade.quera.com/latest/reference/bloqade-circuit/src/bloqade/squin/qubit/).
#
# You are also able to define your own custom gates using the `op` dialects. [The op API documentation is here](https://bloqade.quera.com/latest/reference/bloqade-circuit/src/bloqade/squin/op/stdlib/). Operators are gates and other quantum operations decoupled from application on particular target qubits. Each `squin.gate` method is actually its own kernel which creates an operation and then applies it to a particular qubit. Separating the definition of operators from their application to qubits enables several convenient analysis techniques, such as identifying equivalent gates with common subexpressions elimination. Note that operators do not necessarily have to be unitary and can more widely represent objects such as observables and Hamiltonians.
#
# Finally, the `squin.noise` namespace contains statements to represent noise. [The API documentation is here](https://bloqade.quera.com/latest/reference/bloqade-circuit/src/bloqade/squin/noise/stmts/).

# %% [markdown]
# ### Writing custom gates with the `squin.op` dialect
#
# Let's make some operators and gates that go beyond the basic operators using the `squin.op` dialects. Note that more complicated functionalities, such as T state teleportation, may need function signatures that do not match that of the built-in `squin.gate` functions, and might need to be written using `op`. For example, consider a kernel that applies a controlled T gate to a register of qubits:


# %%
@squin.kernel
def controlled_t(qubit1: bloqade.types.Qubit, qubit2: bloqade.types.Qubit) -> None:
    """
    A controlled T gate, aka a sqrt{CZ} gate.
    ---o---
       |
    ---T---
    """
    t = squin.op.t()
    ctrl = squin.op.control(t, n_controls=1)
    squin.qubit.apply(ctrl, qubit1, qubit2)


# %% [markdown]
# # Using Bloqade kernels
# A key feature of kernels is the ability to do complex control flow similar to how one might program python. For example, one can use a for loop to apply the same gate to multiple qubits to prepare a GHZ state.
#
# A useful pattern is to use factory functions that return bloqade kernels. This way, you can fix parameters (such as the number of qubits) pythonically without needing to introduce the variable directly into the kernel itself, using a closure.

# %%
# Bell state prep.


def GHZ_method_factory(nqubits: int) -> Method:
    @squin.kernel
    def GHZ_state() -> Register:
        qubits = squin.qubit.new(nqubits)
        squin.gate.h(qubits[0])
        for i in range(nqubits):
            squin.gate.cx(qubits[i], qubits[i + 1])

        return qubits

    return GHZ_state


kernel = GHZ_method_factory(8)
kernel.print()
# %% [markdown]
# Alternatively, kernels could be parameterized; for example, we could write the same GHZ state preparation, except it prepares a variable number of qubits that is not declared until the kernel is run. In order to run in some `main` function, the qubits need to be declared elsewhere, either in the task declaration or within a larger kernel that calls this method as a subroutine.


# %%
@squin.kernel
def GHZ_state_factory(nqubits: int) -> Register:
    qubits = squin.qubit.new(nqubits)
    squin.gate.h(qubits[0])
    for i in range(nqubits - 1):
        squin.gate.cx(qubits[i], qubits[i + 1])
    return qubits


GHZ_state_factory.print()
# %% [markdown]
# ## Building circuits in Cirq
# Instead of writing your circuit directly in bloqade, you may build circuits using Cirq and then lower them to and from bloqade kernels. This has the advantage of being able to leverage the excellent and in-depth resources of transpilation and circuit optimization without having to reinvent the wheel. However, for certain programs, such as those requiring more complex mid-circuit feed-forward, it is still required to write bloqade kernels as there is no adequate representation in other SDKs. Cirq is our initial choice of SDK, and other transformations are coming soon-- though in principle interoperability with many SDK is possible through an intermediate Cirq representation.
#
# Let us begin by writing a simple GHZ state preparation circuit, in analogy to the bloqade kernel above. Observe that the resulting object is a static representation of a circuit, similar to the `GHZ_state` kernel, and differentiated from the dynamic `GHZ_state_factory` kernel which can return a dynamically sized GHZ state.

# %%
import cirq


def ghz_prep(nqubits: int) -> cirq.Circuit:
    """
    Builder function that returns a simple N-qubit
    GHZ state preparation circuit
    """
    qubits = cirq.LineQubit.range(nqubits)
    output = cirq.Circuit()
    output.append(cirq.H(qubits[0]))
    for i in range(nqubits - 1):
        output.append(cirq.CX(qubits[i], qubits[i + 1]))
    return output


print(ghz_prep(4))
# %% [markdown]
# The cirq circuit can be converted to a bloqade kernel with a transpilation function `squin.cirq.load_circuit`. The kernel can be considered as a transformation on the register of qubits it is applied to as arguments, with the return being the qubits that still persist.

# %%
# Load a cirq circuit into squin
kernel = squin.cirq.load_circuit(
    ghz_prep(4),
    kernel_name="ghz_prep_cirq",  # Define the name of the kernel as if one were using @squin.kernel on a function
    register_as_argument=False,  # If the resulting kernel should take in a qubit register (True) or make a new one (False)
    return_register=True,  # If the resulting kernel should return the register of the qubits it acts on.
)

# Then, we can convert the circuit back to cirq.
# Note that this is **not possible** in a general case because
# cirq cannot represent complex control flow.
circuit2: cirq.Circuit = squin.cirq.emit_circuit(kernel, ignore_returns=True)
print(circuit2)
# %% [markdown]
# The circuit loading also works with classical feed forward, though it is generally more difficult to extract a cirq circuit from a generic feedforward cirq kernel. For example, the T teleportation gadget can be written and loaded as

# %%
reg = cirq.LineQubit.range(2)
circuit = cirq.Circuit()
circuit.append(cirq.T(reg[0]))
circuit.append(cirq.CNOT(reg[1], reg[0]))
circuit.append(cirq.measure(reg[0], key="m"))
circuit.append(cirq.S(reg[1]).with_classical_controls("m"))
circuit.append(cirq.X(reg[1]).with_classical_controls("m"))
print(circuit)
kernel = squin.cirq.load_circuit(
    circuit, kernel_name="teleport", register_as_argument=True, return_register=True
)
kernel.print()
# %% [markdown]
# Due to the difficulty of representing mid-circuit control flow in cirq, attempting to lower these kernels back to cirq will result in an error.


# %%
@squin.kernel
def t_teleport_noargs() -> None:
    """
    A simple T teleportation circuit that requires mid circuit control flow.
    """
    ancilla = squin.qubit.new(1)[0]
    target = squin.qubit.new(1)[0]
    squin.gate.t(ancilla)
    squin.gate.cx(target, ancilla)
    if squin.qubit.measure(target):
        squin.gate.s(ancilla)
        squin.gate.x(ancilla)


try:
    print(squin.cirq.emit_circuit(t_teleport_noargs))
    raise (RuntimeError("Oops this should have errored."))
except Exception as e:
    print("ERROR:", e)


# Though measurement without feedforward is possible
@squin.kernel
def coinflip() -> MeasurementResult:
    qubit = squin.qubit.new(1)[0]
    squin.gate.h(qubit)
    return squin.qubit.measure(qubit)


circuit = squin.cirq.emit_circuit(coinflip, ignore_returns=True)
print(circuit)
# %% [markdown]
# ## Simulation, emulation, and analysis
#
# A kernel is simply a representation of an execution and is not much use without being able to analyze and execute that kernel. We can simulate the action of kernels using concrete interpreters. The emulator must a) keep track of the classical state of the variables, b) keep track of the quantum state of the qubits, and thus c) faithfully represent the execution of that program as if it was run on a hybrid quantum/classical computer. Bloqade's emulator is built on top of the excellent [PyQrack quantum simulator](https://pyqrack.readthedocs.io/en/latest/) and satisfies the three goals above.
#
# There are four main objects when considering simulation:
# 1. **The emulator object** - Representing the thing that some kernel is going to be executed on. Today it is the PyQrack simulator, but eventually it could also include other simulators or physical hardware. The `*.task` method of the `emulator` object builds a `task` object.
# 2. **The task object** - Binding together the emulator, kernel, and input parameters for that kernel. This task is not executed until the `*.run` method is called. Upon calling `run`, the kernel is interpreted, the quantum circuit is executed, any classical co-processing is done, and the kernel completes, returning . Repeated calling of `run` will "reset" the executor into its initial state and rerun the kernel. Alternatively, one could call `*.batch_run` to repeatedly run the kernel's result to get stochastic averaging. Products of the task include the `results` and `QuantumState` objects.
# 3. **The results object** - Whatever the `return` of the kernel is returned, with the same type signature. This is generated from `*run()` (as a ResultType) or `*.batch_run` (as a dict keyed by ResultType and valued by frequency) These could be qubits or qubit registers (list of qubits), values, or whatever object you like.
# 4. **The QuantumState object** - The final quantum state of the emulator object. While this is a nonphysical quantity, the QuantumState is useful for debugging and analysis. This can be extracted from either the `emulator.quantum_state` method (for a single run after the `run()` method), or with `task.batch_state` (for a stochastic average over many samples). The quantum state is efficiently represented as an eigensystem of a reduced density matrix.

# %%
from bloqade.pyqrack import StackMemorySimulator, DynamicMemorySimulator

# StackMemorySimulator - static number of qubits
# DynamicMemorySimulator - dynamic number of qubits, but slower. Use if you don't know how many qubits you need in advance.
emulator = StackMemorySimulator(min_qubits=8)

task = emulator.task(GHZ_state_factory, args=(4,))
results = task.run()
# The results are the same ResultType as the kernel return.
# In this case, it is a list of qubits.
print(results)
# %% [markdown]
# Note that, while it is instinctive to simply call `GHZ_state_factory(4)` and expect it to run, this isn't necessarily the correct abstraction. `emulator.task` links the kernel to a specific interpreter-- for example, if you wanted to run your program on a noisy emulator vs a perfect emulator vs. real hardware, this is the way you would specify it. Furthermore, the _instantiation_ of the task is not necessarily linked with the _execution_ of that task. For example, if that task must be run asynchronously on hardware, there must be an object which represents the run itself as well as the (future) results.
#
# For simpler kernels that only use kirin built-in statements -- such as control flow, for loops, arithmetic, and the like -- it is possible to directly call the kernel and use the default Kirin interpreter.


# %%
def foo(x: int, y: int) -> bool:
    return x < y


assert foo(1, 2) == True
# %% [markdown]
# ## Extracting quantum states
#
# PyQRack is mainly intended as an *emulator*, attempting to recreate the physical action and interaction of quantum hardware with classical controls. However, it is often useful for user analysis and debugging to extract out non-physical values, such as the quantum statevector, fidelity, or exact expectation values of observables and correlation functions.
#
# These nonphysical values are based in being able to directly observe and evaluate the quantum state. The state can be extracted via the `PyQrackSimulator.quantum_state` method, with an input signature of a list of (pyqrack) qubits. These qubits can be generated as a return value from a kernel (as is the case for the `GHZ_state_factory` function) or from the `task.qubits()` method.
#
# The resulting state is a reduced density matrix represented by its eigensystem of eigenvalues and eigenvectors, or a dense $2^N \times 2^N$ array.
# %%
from bloqade.pyqrack import PyQrackQubit

# This kernel returns a list of Qubit objects. We can use these to analyze the
# quantum state of the register.
results: list[PyQrackQubit] = task.run()

# If the kernel does not return the qubit register, one can still collect it. Note that
# it is now the onus of you, the user, to determine which qubits are which.
# Qubits are typically added sequentially, so if you make multiple registers, the qubits
# will be in the order they were added. The StackMemorySimulator may have extra qubits.
qubits: list[PyQrackQubit] = task.qubits()

# Extract the quantum state as a reduced density matrix. Note that he qubits themselves
# point to the internal state of the emulator
state = emulator.quantum_state(results)
density_matrix = emulator.reduced_density_matrix(results)

# Note that the RDM is represented in its eigenbasis for efficiency.
# If the state is pure, there is only one nonzero eigenvalue. This is the case for the GHZ state.
print("A pure state has only a single eigenvalue [1] of the RDM:", state.eigenvalues)
statevector = state.eigenvectors[:, 0]
print("The statevector of the GHZ state looks like [1, 0, 0, ..., 0, 1]")
print(statevector)
# %% [markdown]
# If the output is randomized, one can average over many runs using `task.batch_run`. This returns a dictionary of probabilities of each output. Note that the output must be hashable.

# %%
# Define the emulator and task
emulator = StackMemorySimulator()
task = emulator.task(coinflip)
results = task.batch_run(shots=1000)
state = task.batch_state(shots=1000)
print("Results:", results)
print("State:", state)


# %% [markdown]
# # Composition of kernels
#
# Bloqade kernels allow all the typical syntax of for loops, if-else statements, function calls, and other powerful abstractions. Let us use this to write efficient representations of complex circuits.
#
# For this example, we will use a [Trotterization of the 1d Transverse Ising model](https://qiskit-community.github.io/qiskit-algorithms/tutorials/13_trotterQRTE.html).
#
# The first option we will explore is to write the entire circuit in Cirq and then convert it into a bloqade kernel using the `squin.cirq.load_circuit` lowering. Observe that the return objects of these builder functions are static objects.
# %%
def trotter_layer(
    qubits: list[cirq.Qid], dt: float = 0.01, J: float = 1, h: float = 1
) -> cirq.Circuit:
    """
    Cirq builder function that returns a circuit of
    a Trotter step of the 1D transverse Ising model
    """
    op_zz = cirq.ZZ ** (dt * J / np.pi)
    op_x = cirq.X ** (dt * h / np.pi)
    circuit = cirq.Circuit()
    for i in range(0, len(qubits) - 1, 2):
        circuit.append(op_zz.on(qubits[i], qubits[i + 1]))
    for i in range(1, len(qubits) - 1, 2):
        circuit.append(op_zz.on(qubits[i], qubits[i + 1]))
    for i in range(len(qubits)):
        circuit.append(op_x.on(qubits[i]))
    return circuit


def trotter_circuit(
    N: int, steps: int = 10, dt: float = 0.01, J: float = 1, h: float = 1
) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(N)
    circuit = cirq.Circuit()
    for _ in range(steps):
        circuit += trotter_layer(qubits, dt, J, h)
    return circuit


cirq_trotter_circuit = trotter_circuit(N=8, steps=4, dt=0.01, J=1, h=1)

print(cirq_trotter_circuit)

# Convert the circuit to a bloqade kernel
bloqade_trotter_circuit = squin.cirq.load_circuit(
    cirq_trotter_circuit,
    kernel_name="trotter",
    register_as_argument=False,
    return_register=True,
)
# %% [markdown]
# As an intermediate, one can mix between writing kernels converted from Cirq circuits and direct bloqade kernels. For example, each layer has fixed parameters as defined by a cirq circuit, but a variable number of layers as parameterized by a kernel input and for loop. This option has the benefit of being able to use Cirq infrastructure to optimize and represent individual layers, while still being able to use bloqade kernels to represent parameterized circuits. In this case, the output kernel has the timestep and Ising parameters fixed (as they are fixed in the cirq circuit), but the number of steps is variable.


# %%
def factory_trotter(N: int, dt: float = 0.01, J: float = 1, h: float = 1) -> Method:
    bloqade_trotter_layer = squin.cirq.load_circuit(
        trotter_layer(qubits=cirq.LineQubit.range(N), dt=dt, J=J, h=h),
        kernel_name="trotter",
        register_as_argument=True,
        return_register=True,
    )

    @squin.kernel
    def trotter_for_loop(steps: int) -> Register:
        """
        Main function that runs the Trotter circuit for a given number of steps
        """
        qubits = squin.qubit.new(N)
        for _ in range(steps):
            qubits = bloqade_trotter_layer(qubits)
        return qubits

    return trotter_for_loop


# %% [markdown]
# Alternatively, you could just write everything directly as a Bloqade kernel. Note that the ZZ operator that is native to Cirq must be expanded into its own "helper" kernel via a decomposition into cx / z / cx. The resulting kernel is fully parameterized, with the values not actually evaluated until runtime (or further compilation and folding).
# %%
# Define an operator that looks like the ZZ power gate
@squin.kernel
def op_zz(theta: float, qb1: bloqade.types.Qubit, qb2: bloqade.types.Qubit) -> None:
    """
    A kernel that returns an operator that looks like ZZ^{theta/2pi}
    """
    squin.gate.cx(qb1, qb2)
    squin.gate.rz(theta, qb2)
    squin.gate.cx(qb1, qb2)


@squin.kernel
def bloqade_trotter(
    N: int, steps: int, dt: float = 0.01, J: float = 1, h: float = 1
) -> Register:
    """
    Main function that runs the Trotter circuit for a given number of steps
    """
    qubits = squin.qubit.new(N)
    xpow = squin.op.rot(squin.op.x(), angle=dt * h)
    for _ in range(steps):
        for i in range(0, len(qubits) - 1):
            op_zz(theta=dt * J, qb1=qubits[i], qb2=qubits[i + 1])
        for i in range(0, len(qubits)):
            squin.qubit.apply(operator=xpow, qubits=[qubits[i]])
    return qubits


# %% [markdown]
# Of course, both Cirq and the (converted) Bloqade kernel have the same execution and same output state.

# %%
cirq_trotter = trotter_circuit(N=12, steps=10, dt=0.01, J=1, h=1)

cirq_statevector = cirq.Simulator().simulate(cirq_trotter).state_vector()

# Or converting to a bloqade kernel and simulating with PyQrack
cirq_trotter_kernel = squin.cirq.load_circuit(
    cirq_trotter,
    kernel_name="cirq_trotter",
    register_as_argument=False,
    return_register=True,
)
cirq_trotter_qubits = (
    StackMemorySimulator(min_qubits=12).task(cirq_trotter_kernel).run()
)
cirq_bloqade_state = StackMemorySimulator.quantum_state(cirq_trotter_qubits)
# The state is, of course, the same. A little bit of work is needed to extract out the (single) state vector from the RDM.
print(
    "Overlap:",
    np.abs(np.dot(np.conj(cirq_statevector), cirq_bloqade_state.eigenvectors[:, 0]))
    ** 2,
)
# %% [markdown]
# Similarly, the execution and output state of the cirq state and kernel written fully in bloqade have the same state. Note that for the `bloqade_trotter` kernel, the arguments are not declared until the simulator is run.

# %%
cirq_trotter_qubits = StackMemorySimulator(min_qubits=12).run(
    bloqade_trotter, args=(12, 10, 0.01, 1, 1)
)
cirq_bloqade_state = StackMemorySimulator.quantum_state(cirq_trotter_qubits)
print(
    "Overlap:",
    np.abs(np.dot(np.conj(cirq_statevector), cirq_bloqade_state.eigenvectors[:, 0]))
    ** 2,
)


# %% [markdown]
# # Mid-circuit feed forward
#
# Bloqade kernels can natively represent mid-circuit feed-forward using control flow represented by standard pythonic if-else and while structures. While the possibilities are endless, including measurement-based quantum computing and error correction, we show two examples here.
#
# The first is T state teleportation, which teleports a T gate that was applied to an ancilla (a "T state") onto the target state using only Clifford gates and feedforward. Due to the property of being Clifford, the circuit itself is fault tolerant and thus plays an important role in many error corrected algorithms.
# %%
@squin.kernel
def t_teleport(target: squin.qubit.Qubit) -> squin.qubit.Qubit:
    ancilla = squin.qubit.new(1)[0]
    squin.gate.h(ancilla)
    squin.gate.t(ancilla)
    squin.gate.cx(control=target, target=ancilla)
    bit = squin.qubit.measure(target)
    if bit:
        squin.gate.s(ancilla)
    return ancilla  # The state of the target qubit is also teleported to the ancilla


# And now let’s wrap it into a larger context to run. In this case,
# apply to a |+> state and see that we get a T|+> state out.
@squin.kernel
def t_teleport_wrapper() -> squin.qubit.Qubit:

    target = squin.qubit.new(1)[0]
    squin.gate.h(target)
    target = t_teleport(target)
    return target


# And run it. Observe that the batch_state uses a qubit_map to select which qubits to include in the batch state.
# This is important because there are two qubits total (the target and the ancilla) but we only want inspect
# the state of the output qubit.
emulator = StackMemorySimulator(min_qubits=2)
task = emulator.task(t_teleport_wrapper)
state = task.batch_state(shots=1000, qubit_map=lambda x: [x])
# Even though there is measurement and feedforward, the final state is still pure. Neat!
print(state)
# %% [markdown]
# ### Constant depth GHZ state
#
# Remarkably, it is also possible to prepare a GHZ state with a constant number of gates. At first glance, this seems impossible: quantum information can only propagate as fast as the information lightcone, and so a constant depth circuit can only have a small amount of entanglement—not enough to prepare the long-range correlated GHZ state. The trick is, we can replace quantum gates with ancillas and classical feedforward measurements so that the information is propagated classically instead of quantumly. For more details, check out [1] and [2].
#
# [1] [Efficient Long-Range Entanglement Using Dynamic Circuits](https://doi.org/10.1103/PRXQuantum.5.030339)
#
# [2] [Constant-Depth Preparation of Matrix Product States with Adaptive Quantum Circuits](https://doi.org/10.1103/PRXQuantum.5.030344)
#
# The explicit circuit for the GHZ circuit is shown in Fig. 5 of [1]. There is classical feedforward in the form of a parity check, which requires a classical XOR operation that is irrepresentable by CIRQ.


# %%
def ghz_constant_depth(n_qubits: int):

    @squin.kernel
    def main() -> Register:
        qreg = squin.qubit.new(n_qubits)
        ancilla = squin.qubit.new(n_qubits - 1)

        for i in range(n_qubits):
            squin.gate.h(qreg[i])

        for i in range(n_qubits - 1):
            squin.gate.cx(qreg[i], ancilla[i])
        for i in range(n_qubits - 1):
            squin.gate.cx(qreg[i + 1], ancilla[i])

        parity: int = 0
        bits = squin.qubit.measure(ancilla)
        for i in range(n_qubits - 1):
            parity = parity ^ bits[i]
            if parity == 1:
                squin.gate.x(qreg[i + 1])
        return qreg

    return main


# %%
# At this point, you know the drill. We can simulate this with multirun via PyQrack
emulator = StackMemorySimulator(min_qubits=7)
task = emulator.task(ghz_constant_depth(3))

state = task.batch_state(shots=1000, qubit_map=lambda x: x)
# Even though there is measurement and feedforward, the final state is still pure. Neat!
print(state.eigenvalues)
# %% [markdown]
# As a final note, consider how difficult it would be to represent this circuit in Cirq. In particular, there is a for loop, where inside the for loop there is an algebraic operation (XOR) that feeds forward onto a variable (parity). This circuit is very hard to express in Cirq without some serious hacking of ancilla registers.
