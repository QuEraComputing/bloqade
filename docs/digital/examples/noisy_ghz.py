# %% [markdown]
# # GHZ State preparation and noise
#
# In this example, we will illustrate how to work with bloqade's noise models of the Gemini architecture in order to study its effects on a circuit that prepares a GHZ state.

# %% [markdown]
# ## Primer on Gemini noise models
#
# There are essentially two classes of heuristic noise models: one-zone models such as `GeminiOneZoneNoiseModel` and a two-zone model `GeminiTwoZoneNoiseModel`.
# These correspond to two distinct approaches getting a sense of the influence of noise on Gemini class hardware.
#
# On the one hand, the one-zone model assumes a single-zone layout where qubits remain in the gate zone throughout the computation.
# On the other hand, the two-zone model incorporates a storage zone and assumes that qubits are transported between gate and storage regions.
#
# Both models are informed by benchmark data on the device but are intentionally conservative.
# Specifically, they tend to overestimate noise due to the lack of knowledge about optimized move schedules, which leads to overestimating move-induced errors.
#
# At this stage, we recommend interpreting the two models as providing a range for expected noise levels on Gemini-class devices, rather than precise predictions. They are useful for gaining intuition about noise sensitivity and for benchmarking algorithmic robustness under realistic but  assumptions.
#
# Note, that there are actually two additional one-zone noise models, `GeminiOneZoneNoiseModelCorrelated` and `GeminiOneZoneNoiseModelConflictGraphMoves`.
# As the names suggest, the former also takes into account correlated noise, whereas the latter takes into account more realistic move schedules.
# In the following example, we will not be considering these two, but they are interchangeable with the used noise models (up to the fact, that the conflict graph moves require you to specify a grid using `cirq.GridQubit`s).

# %% [markdown]
# ## Noise model implementations
#
# For now, these noise models are implemented as [`cirq.NoiseModel`](https://quantumai.google/reference/python/cirq/NoiseModel) classes, so that you can use with any circuit you build using `cirq`.
# They are part of the `bloqade.cirq_utils` submodule.
#
# Support for using these models with e.g. [squin](../dialects_and_kernels.md) will follow in the future.
# However, you can already rely on [interoperability with cirq](../cirq_interop) in order to convert (noisy) circuits to squin kernels and use other parts of the compiler pipeline.

# %% [markdown]
# ## GHZ preparation and noise
#
# Now, let's get started with the actual example.
#
# As a first step, we will define a function that builds a GHZ circuit in cirq for a given number of qubits.
# The resulting circuit will grow linearly with the number of qubits.

# %%
import cirq
import numpy as np
import matplotlib.pyplot as plt
from bloqade.cirq_utils import noise, transpile

from bloqade import squin


def ghz_circuit(n: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(n)

    # Step 1: Hadamard on the first qubit
    circuit = cirq.Circuit(cirq.H(qubits[0]))

    # Step 2: CNOT chain from qubit i to i+1
    for i in range(n - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    return circuit


# %% [markdown]
# ### Closer look at a basic circuit

# %% [markdown]
# Here's what this circuit looks like for a number of qubits:

# %%
ghz_circuit_3 = ghz_circuit(3)
print(ghz_circuit_3)

# %% [markdown]
# So far so good.
# Now, we will convert the circuit above to a noisy one using bloqade's `cirq_utils` submodule.
#
# Specifically, we can use the `noise.transform_circuit` utility function with a noise model of our choice.

# %%
noise_model = noise.GeminiOneZoneNoiseModel()
noisy_ghz_circuit_3 = noise.transform_circuit(ghz_circuit_3, model=noise_model)
print(noisy_ghz_circuit_3)

# %% [markdown]
# As you can see, we have successfully added noise.
# However, the circuit also looks very different in terms of it's gates.
#
# This is because `noise.transform_circuit` does actually two things:
#
# 1. Since we want to consider a circuit that is compatible with the Gemini architecture, we need to transform it to the native gate set first. This set consists of (phased) X gates and CZ gates only.
# 2. Once we have a native circuit, noise is injected according to the used noise model.
#
# To clarify, here is how you would convert the circuit without using the `noise.transform_circuit` utility function:

# %%
native_ghz_3 = transpile(ghz_circuit_3)
print(native_ghz_3)

# %% [markdown]
# Note that `transpile` basically just wraps cirq's own `cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())`, with some additional benefits (such as filtering out empty moments).
#
# Using this native circuit, we can obtain the same noisy circuit as before by simply using cirq's `cirq.Circuit.with_noise` method.

# %%
noisy_ghz_circuit_3 = native_ghz_3.with_noise(noise_model)
print(noisy_ghz_circuit_3)

# %% [markdown]
# ### Studying the fidelity
#
# Now that we got the basics down, we can compute the fidelity of the circuit with noise present for different qubit numbers.
# By fidelity, we simply mean the difference from a noise-less version of the circuit.
#
# The corresponding density matrices are obtained using cirq's simulator.
#
# We will do so using two different noise models, the one-zone model from before and also the two-zone model.

# %% [markdown]
# <div class="admonition note">
# <p class="admonition-title">Fidelity calculation</p>
# <p>
#     In the following, we will simply use the expectation value of the noisy density matrix computed against the noiseless one as fidelity.
#     This is a suboptimal choice, but we wanted to keep the example simple.
#     Feel free to substitute the fidelity calculation by the fidelity of your choice (e.g. the Uhlmann fidelity)
# </p>
# </div>

# %%
qubits = range(3, 9)

one_zone_model = noise.GeminiOneZoneNoiseModel()
two_zone_model = noise.GeminiTwoZoneNoiseModel()
simulator = cirq.DensityMatrixSimulator()

fidelities_one_zone = []
fidelities_two_zone = []
for n in qubits:
    circuit = ghz_circuit(n)
    one_zone_circuit = noise.transform_circuit(circuit, model=one_zone_model)
    two_zone_circuit = noise.transform_circuit(circuit, model=two_zone_model)

    rho = simulator.simulate(circuit).final_density_matrix
    rho_one_zone = simulator.simulate(one_zone_circuit).final_density_matrix
    rho_two_zone = simulator.simulate(two_zone_circuit).final_density_matrix

    fidelity_one_zone = np.trace(rho @ rho_one_zone)
    fidelity_two_zone = np.trace(rho @ rho_two_zone)

    fidelities_one_zone.append(fidelity_one_zone)
    fidelities_two_zone.append(fidelity_two_zone)

# %% [markdown]
# Now, let's have a look at the results.

# %%
plt.plot(qubits, fidelities_one_zone, "o", label="one-zone model")
plt.plot(qubits, fidelities_two_zone, "x", label="two-zone model")
plt.xlabel("Number of qubits")
plt.ylabel("Fidelity")
plt.legend()

# %% [markdown]
# <div align="center">
# <picture>
#    <img src="../noisy_ghz_fidelity.svg" >
# </picture>
# </div>

# %% [markdown]
# We can see that in both cases the fidelity goes down when increasing the number of qubits.
#
# Interestingly, there is a cross-over point where the two-zone model starts to exhibit a better fidelity.
# This is because as the number of qubits grows, the error introduced on idle qubits inside the gate zone is larger in the one-zone model since all qubits are always inside the gate zone.
# Whereas, in the two-zone model, qubits are moved between the gate and storage zones.
#
# You could now think about how to optimize the circuits in order to reduce their sensitivity for noise.
# For example, you can [reduce the circuit depth](../ghz)

# %% [markdown]
# ## Interoperability with squin
#
# Finally, we want to point out that you can also use the generated circuit's to obtain a squin kernel function.
#
# This is useful if you want to use other features of the bloqade pipeline.
# For example, it would allow you to run the pyqrack simulator instead of cirq's own, which can be more efficient.

# %%
circuit = ghz_circuit(5)
noisy_circuit = noise.transform_circuit(circuit, model=noise.GeminiOneZoneNoiseModel())

# %%
kernel = squin.cirq.load_circuit(circuit, kernel_name="kernel")
noisy_kernel = squin.cirq.load_circuit(noisy_circuit, kernel_name="noisy_kernel")
kernel.print()
