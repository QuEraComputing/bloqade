# %% [markdown]
# # Repeat Until Success with STAR Gadget
# In this example, we will demonstrate a near-term fault tolerant gadget
# which is a repeat-until-success protocol to implement a Z phase gate
# using a resource state (similar to a T state), Pauli gates, and feed-forward measurement.
#
# For more information, see https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337,
# especially Fig. 7.

# %%
from bloqade import qasm2

# %% [markdown]
# This example highlights a few interesting capabilities of having a full kernel structure with
# runtime control flow. One example is the ability to dynamically allocate qubits, possibly
# based on previous run-time measurement outcomes.
#
# In this case, we prepare a resource state, which is a generalization of the T state with
# an arbitrary Z rotation $|0\rangle + e^{i\theta}|1\rangle$.


# %%
@qasm2.extended
def prep_resource_state(theta: float):
    qreg = qasm2.qreg(1)
    qubit = qreg[0]
    qasm2.h(qubit)
    qasm2.rz(qubit, theta)
    return qubit


# %% [markdown]
# Using this resource state, we can teleport the Z phase gate to a target qubit using
# only Clifford gates, which are much easier to implement fault-tolerantly.
# This is implemented by first applying a CNOT gate controlled by the resource state
# on the target qubit, then measuring the target qubit in the computational basis.
# If the measurement outcome is 1 (which occurs with 50% probability), the gadget
# executed a Z(theta) gate on the target qubit and teleported it
# to the new resource state.
#
# However, if the measurement outcome is 0 (which occurs with 50% probability),
# we apply an X gate, and the gadget executed a Z(-theta) gate on the target qubit.
# In order to correct this gate, we must apply a Z(+2*theta) gate on the new target state.
# Of course, we can apply this Z(+2*theta) gate by applying the same gadget with twice
# the angle, and repeat until we get the correct outcome.

# %% [markdown]
# The simplest way to implement the gadget is to simply post-select the correct measurement outcome
# using an assert statement. This is straightforward, but comes with an exponential overhead in the
# number of resource states: there is a 50% chance of success at each step, so there is only a
# $2^{-n}$ chance of success after $n$ Z phase gates.


# %%
@qasm2.extended
def z_phase_gate_postselect(target: qasm2.Qubit, theta: float) -> qasm2.Qubit:
    ancilla = prep_resource_state(theta)
    qasm2.cx(ancilla, target)
    creg = qasm2.creg(1)
    qasm2.measure(target, creg[0])
    if creg[0] == 1:
        qasm2.x(ancilla)
    return ancilla


# %% [markdown]
# To (deterministically) implement the gate, we can recursively apply the gadget by correcting
# the angle of the Z gate by applying Z(+2*theta).
# Observe that, while it is efficient to represent this as a composition of kernels,
# there is no equivalent representation as a circuit, as the number of resource qubits and
# total number of gates is not known until runtime.


# %%
@qasm2.extended
def z_phase_gate_recursive(target: qasm2.Qubit, theta: float) -> qasm2.Qubit:
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    ancilla = prep_resource_state(theta)
    qasm2.cx(ancilla, target)
    creg = qasm2.creg(1)
    qasm2.measure(target, creg[0])
    if creg[0] == 0:
        return z_phase_gate_recursive(ancilla, 2 * theta)
    if creg[0] == 1:
        qasm2.x(ancilla)
    return ancilla


# %% [markdown]
# An alternative representation uses control flow to
# implement the same gate. If the number of repeats is fixed, this can be represented
# as a static circuit, though it would require a large number of resource qubits and
# may still fail with a small probability $2^{-attempts}$.


# %%
@qasm2.extended
def z_phase_gate_loop(target: qasm2.Qubit, theta: float, attempts: int):
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    creg = qasm2.creg(1)  # Implicitly initialized to 0, thanks qasm...
    for ctr in range(attempts):
        ancilla = prep_resource_state(theta * (2**ctr))
        if creg[0] == 0:
            qasm2.cx(ancilla, target)
            qasm2.measure(target, creg[0])
            target = ancilla
    qasm2.x(target)


# %% [markdown]
# Before we analyze these circuits, we must declare a main function
# which takes no inputs, as qasm2 does not support parameterized circuits or
# subcircuits.

# %%
theta = 0.1  # Specify some Z rotation angle. Note that this is being defined

# %% [markdown]
# outside the main function and being used inside the function via closure.


# %%
@qasm2.extended
def postselect_main():
    target = qasm2.qreg(1)
    z_phase_gate_postselect(target[0], theta)


@qasm2.extended
def recursion_main():
    target = qasm2.qreg(1)
    z_phase_gate_recursive(target[0], theta)


@qasm2.extended
def loop_main():
    target = qasm2.qreg(1)
    z_phase_gate_loop(target[0], theta, 5)


# %% [markdown]
# Now lets explore running some interpreters on these circuits.
# We support the quantum emulation backend PyQrack, which simulates quantum
# circuits using state vectors.

# %%
from bloqade.pyqrack import PyQrack  # noqa: E402

device = PyQrack()
device.run(postselect_main)
