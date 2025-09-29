# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: kirin-workspace (3.12.10)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GHZ State Preparation with Squin
#
# In this example, we will show (yet again) how to implement a program that prepares a GHZ state.
# We will do so with a simple linear algorithm and show how to manually insert noise.

# %% [markdown]
# The circuit we will implement is displayed below:
#
# ![GHZ linear circuit](../../ghz_linear_circuit.svg)

# %% [markdown]
# Since this circuit is rather simple, we can stick to the Squin standard library in order to implement it.
# The gates are defined in the `squin.gate` submodule.
#
# Let's start by importing Squin and writing our circuit for an arbitrary number of qubits.

# %%
from bloqade.pyqrack import StackMemorySimulator  # we'll need that later

from bloqade import squin


@squin.kernel
def ghz_linear(n: int):
    q = squin.qubit.new(n)
    squin.gate.h(q[0])
    for i in range(1, n):
        squin.gate.cx(q[i - 1], q[i])


ghz_linear.print()

# %% [markdown]
# As you can see, writing basic circuits in squin is rather straightforward.


# %% [markdown]
# ## Simulating the kernel
#
# You can simulate a kernel such as the above using bloqade's PyQrack backend.
#
# There are two basic simulators, that act like "devices" that you run your program on:
#
# * The `StackMemorySimulator`, which initializes its memory with a fixed number of qubits. The number is either set via the `min_qubits` argument or inferred automatically. Note, that automatic inference is not always possible in which case you will be required to set the argument accordingly.
# * The `DynamicMemorySimulator`, which, as the name suggests, allocates memory as required throughout the circuit. Generally, you should prefer the `StackMemorySimulator` over this one unless the number of qubits can only be known at runtime.
#
# Let's go ahead and use the `StackMemorySimulator` together with a fixed number of qubits to simulate our GHZ preparation program from above.

# %%
sim = StackMemorySimulator(min_qubits=2)
result = sim.run(ghz_linear, args=(2,))
print(result)

# %% [markdown]
# As you can see, the result of our simulation is `None`.
# That is because we are not returning anything from the kernel function.
#
# Note, how we're passing in the arguments of the kernel function as a separate tuple in the call to `run`.
# This signature is required since `ghz_linear(2)` would not return another kernel function, but rather attempt to run the kernel function as a Python function.
# As the function is written in squin rather than Python, this would fail.
# To provide a little more detail here: the `PyQrack` backend in bloqade-circuit actually has its own method table which tells it how to interpret the statements encountered in the squin kernel function.
#
# Since we are only simulating the circuit, however, we are able to fetch information that would otherwise not be attainable.
# For example, you can obtain the state vector from the simulator:

# %%
print(sim.state_vector(ghz_linear, args=(2,)))


# %% [markdown]
# Looking at the output, we can see that we indeed prepared a two-qubit GHZ state (up to a global phase).
#
# Note, that you can also add a return value to the kernel, which is then returned by `sim.run`.
# Again, this is not generally possible when running on hardware, but only during simulation.
#
# A realistic kernel function will return (a list of) measurement results.
# That is precisely what we will do in the following.

# %% [markdown]
# ## Inserting noise
#
# The above is rather basic, so let's try to do something that is a little more interesting.
# Let's write the same program as before, but now we assume that noise processes occur whenever a gate is applied.
#
# We will make use of Squin's noise submodule in order to do that.
#
# Our "noise model" will be quite simple:
# * Whenever a single-qubit gate is applied, that qubit undergoes a depolarization error with probability `p_single`.
# * Whenever a two-qubit (controlled) gate is applied, both qubits undergo a joint depolarization error with probability `p_paired`.
#
# Note, that a depolarization error with probability $p$ on a single qubit means that randomly chosen Pauli operators (one of $X, Y, Z$) is applied to the qubit with probability $p$.
# Similarly, a two-qubit depolarization error applies one of the 15 operators $IX, IY, IZ, XI, XX, ...$ with a given probability.


# %%
@squin.kernel
def noisy_linear_ghz(n: int, p_single: float, p_paired: float):
    q = squin.qubit.new(n)

    # define the noise operator for the single qubit
    single_qubit_noise = squin.noise.depolarize(p_single)

    squin.gate.h(q[0])
    squin.qubit.apply(single_qubit_noise, q[0])

    # pair qubit noise operator
    two_qubit_noise = squin.noise.depolarize2(p_paired)

    for i in range(1, n):
        squin.gate.cx(q[i - 1], q[i])
        squin.qubit.apply(two_qubit_noise, q[i - 1], q[i])

    return squin.qubit.measure(q)


# %%
noisy_linear_ghz.print()

# %% [markdown]
# <div class="admonition note">
# <p class="admonition-title">Noise operators</p>
# <p>
#     As opposed to standard gates, there is no standard library for noise statements as of now.
#     While we plan to add that in the future, also note how it can be convenient to separate the operator
#     from the gate application: we define the paired noise operator only once and apply it to different
#     pairs of qubits in the loop.
# </p>
# </div>

# %% [markdown]
# This kernel function can be simulated in the exact same way as before.
# The only difference is that we now need to provide additional arguments for the noise probabilities.

# %%
result = sim.run(noisy_linear_ghz, args=(2, 1e-2, 2e-2))
print(result)

# %% [markdown]
# Now that we actually return something, we also obtain a result from running the simulation.
# This result is just a list of measurement results (boolean values corresponding to 0 and 1).
# We can also obtain the bit string:

# %%
result_bitstring = [int(res) for res in result]
print(result_bitstring)

# %% [markdown]
# Ideally, the two values should always be correlated since we want to prepare a GHZ state.
# However, now that we've added noise, this is not always the case.
#
# We can actually use this fact to define a "fidelity" measure for the circuit: when repeatedly executing the circuit, uncorrelated results lower the fidelity.
#
# Mathematically, let's define the fidelity $F$ as
#
# $F = 1 - \sum_{i=1}^n \frac{\text{err}(i)}{n}$,
#
# where $n$ is the number of shots we take and
#
# $ \text{err}(i) = \begin{cases} 0 \text{ if run }i \text{ is correct} \\ 1 \text{ else } \end{cases}$
#
# In this case, "correct" means the measurement outcome is fully correlated.

# %%
n_shots = 1000
n_qubits = 4
p_single = 1e-2
p_paired = 2 * p_single
fidelity = 1.0
sim = StackMemorySimulator(min_qubits=n_qubits)
for _ in range(n_shots):
    result = sim.run(noisy_linear_ghz, args=(n_qubits, p_single, p_paired))
    measured_one_state = all(result)
    measured_zero_state = not any(result)
    is_correlated = measured_one_state or measured_zero_state
    if not is_correlated:
        fidelity -= 1 / n_shots

print(fidelity)

# %% [markdown]
# Note, that this is actually a poor measure for fidelity as it only counts fully correlated states and treats everything else as an equivalent error.
# If you have many qubits, you could argue that only flipping a single bit is a much lower error than flipping many, and that this should be weighed in here.
# Or, you can simply use the simulator to obtain the state vector and compute the overlap.
# Or, define whatever measure of fidelity you see fit here, but we'll end this tutorial here.
