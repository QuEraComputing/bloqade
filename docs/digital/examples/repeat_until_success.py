# %% [markdown]
# # Repeat Until Success with STAR Gadget
# In this example, we will demonstrate a near-term fault tolerant gadget
# which is a repeat-until-success protocol to implement a Z phase gate
# using a resource state (similar to a T state), Pauli gates, and feed-forward measurement.
#
# For more information, see https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337,
# especially Fig. 7.

from bloqade import qasm2

# %% [markdown]
# This example highlights a few interesting capabilities of having a full kernel structure with
# runtime control flow. One example is the ability to dynamically allocate qubits, possibly
# based on previous run-time measurement outcomes.
#
# In this case, we prepare a resource state, which is a generalization of the T state with
# an arbitrary Z rotation $|0\rangle + e^{i\theta}|1\rangle$.


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


@qasm2.extended
def z_phase_gate_postselect(target: qasm2.Qubit, theta: float):
    ancilla = prep_resource_state(theta)
    qasm2.cx(ancilla, target)
    creg = qasm2.creg(1)
    qasm2.measure(target, creg[0])


# %% [markdown]
# To (deterministically) implement the gate, we can recursively apply the gadget.
# Observe that, while it is efficient to represent this as a composition of kernels,
# there is no equivalent representation as a circuit, as the number of resource qubits and
# total number of gates is not known until runtime.


@qasm2.extended
def z_phase_gate_recursive(target: qasm2.Qubit, theta: float):
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    ancilla = prep_resource_state(theta)
    qasm2.cx(ancilla, target)
    creg = qasm2.creg(1)
    qasm2.measure(target, creg[0])
    if creg[0] == 1:
        qasm2.reset(ancilla)
        z_phase_gate_recursive(ancilla, 2 * theta)
        return


# %% [markdown]
# An alternative representation uses control flow to
# implement the same gate. If the number of repeats is fixed, this can be represented
# as a static circuit, though it would require a large number of resource qubits and
# may still fail with a small probability $2^{-attempts}$.


@qasm2.extended
def z_phase_gate_loop(target: qasm2.types.Qubit, theta: float, attempts: int):
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    creg = qasm2.creg(1)

    for ctr in range(attempts):
        ancilla = prep_resource_state(theta * (2**ctr))
        if creg[0] == 0:
            qasm2.cx(ancilla, target)
            qasm2.measure(target, creg[0])

        if creg[0] == 1:
            qasm2.reset(ancilla)
            target = ancilla


# # %% [markdown]
# # Lets try to write this in a way that can be unwrapped into a qasm circuit.
# @qasm2.extended
# def z_phase_gate_qasm(target: qasm2.types.Qubit, theta: float, attempts: int = 10):
#     """
#     https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
#     """
#     creg = qasm2.creg(1)

#     for ctr in range(attempts):
#         ancilla = prep_resource_state(theta * (2**ctr))

#         if creg[0]:
#             qasm2.id(ancilla[0])

#         else:
#             qasm2.cx(ancilla[0], target)
#             qasm2.measure(target, creg[0])

#             if creg[0]:
#                 qasm2.id(ancilla[0])
#             else:
#                 qasm2.x(ancilla[0])
#                 target = ancilla[0]

#     # assert not creg, "Loop did not converge"
#     return target


# Lets try to get pyqrack interpreter to run

from bloqade.pyqrack import PyQrack  # noqa: E402

theta = 0.1


@qasm2.extended
def recursion_main():
    target = qasm2.qreg(1)
    z_phase_gate_recursive(target[0], theta)
    return target


@qasm2.extended
def loop_main():
    target = qasm2.qreg(1)
    z_phase_gate_loop(target[0], theta, 5)
    return target


device = PyQrack()
qreg = device.run(recursion_main)
print(qreg)

# %% [markdown]
# Lets unwrap the postselection and loop versions of the gadget to see the qasm circuit.

from bloqade.qasm2.emit import QASM2  # noqa: E402
from bloqade.qasm2.parse import pprint  # noqa: E402

# qasm2 does not support parameterized circuits, so the entry point must be a function
# of zero arguments. We can use a closure to pass the parameters from outside the function.
# theta = 0.1


@qasm2.extended
def main():
    target = qasm2.qreg(1)
    return z_phase_gate_postselect(target[0], theta)


target = QASM2()
ast = target.emit(main)
pprint(ast)

# %% [markdown]
# And now the loop version, which first needs to be unwrapped:

pprint(target.emit(loop_main))
