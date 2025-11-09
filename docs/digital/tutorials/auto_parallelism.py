# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv (3.13.2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parallelism of Static Circuits
#
# This tutorial describes Bloqade's tools for converting sequential quantum circuits into parallel ones and for evaluating how parallelization affects performance using realistic noise models.
#
# Parallelism lets gates that act on disjoint qubits execute at the same time, reducing circuit depth and overall runtime. On neutral-atom quantum computers, many transversal operations (same gate type and parameters) can often be executed together in a single layer (moment).
#
# Reducing depth typically improves fidelity and increases the number of operations that can complete within the hardware's coherence time.
#
# Bloqade supports both automatic and manual parallelization. The examples below show both methods and compare fidelity using representative noise models.
#

# %% [markdown]
# ## Example 1: GHZ Circuit
#
# ### What is parallelism ?
# We take the GHZ state preparation circuit as an example:
#
# <div style="display: flex; justify-content: space-around; align-items: center;">
#   <div style="text-align: center;">
#     <img src="figures/ghz_linear_circuit.svg" alt="Linear GHZ circuit" height="300"/>
#     <p><b>Linear GHZ circuit</b></p>
#   </div>
#   <div style="text-align: center;">
#     <img src="figures/ghz_log_circuit.svg" alt="Log-depth GHZ circuit" height="300"/>
#     <p><b>Log-depth GHZ circuit</b></p>
#   </div>
# </div>
#
# The GHZ state can be prepared using a sequence of Hadamard and CNOT gates. In a linear (sequential) implementation, the CNOT gates are applied one after another, resulting in a circuit depth that grows linearly with the number of qubits. In contrast, a log-depth (parallel) implementation arranges the CNOT gates so that multiple gates acting on disjoint qubits can execute simultaneously, reducing the overall depth to logarithmic in the number of qubits.
# %%
import cirq
import matplotlib.pyplot as plt
import numpy as np
from bloqade import squin, cirq_utils
import bloqade.cirq_utils as utils
from cirq.contrib.svg import SVGCircuit


# %%
def build_linear_ghz(n_qubits: int) -> cirq.Circuit:
    """Build linear GHZ circuit using squin and convert to Cirq."""

    @squin.kernel
    def linear_ghz_kernel():
        q = squin.qalloc(n_qubits)
        squin.h(q[0])
        for i in range(n_qubits - 1):
            squin.cx(q[i], q[i + 1])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq_utils.emit_circuit(linear_ghz_kernel, circuit_qubits=qubits)
    return circuit


def build_log_ghz(n_qubits: int) -> cirq.Circuit:
    """Build logarithmic-depth GHZ circuit using squin and convert to Cirq."""
    import math

    max_iterations = math.ceil(math.log2(n_qubits)) if n_qubits > 1 else 1

    @squin.kernel
    def log_ghz_kernel():
        q = squin.qalloc(n_qubits)
        squin.h(q[0])

        for level in range(max_iterations):
            width = 2**level
            for i in range(n_qubits):
                if i < width:
                    target = i + width
                    if target < n_qubits:
                        squin.cx(q[i], q[target])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq_utils.emit_circuit(log_ghz_kernel, circuit_qubits=qubits)
    return circuit


# %% [markdown]
# ### The benefits of parallelism
# We'll run noise simulations for both circuits and compare their fidelities as we scale the number of qubits.
#
# See our blog post [Simulating noisy circuits for near-term quantum hardware](https://bloqade.quera.com/latest/blog/2025/07/30/simulating-noisy-circuits-for-near-term-quantum-hardware/) for detailed information about the noise model used here. The analysis workflow is:
#
# 1. Build a noiseless (ideal) circuit.
# 2. Choose a noise model (we use the Gemini noise model).
# 3. Apply the noise model to the circuit to produce a noisy circuit.
# 4. Simulate the noisy circuit to obtain the final density matrix.
# 5. Simulate the ideal circuit and compare its state to the noisy density matrix to compute fidelity.
#

# %%

# Initialize noise model (using Gemini one-zone architecture)
noise_model = utils.noise.GeminiOneZoneNoiseModel()
simulator = cirq.DensityMatrixSimulator()

# Initialize lists to store fidelities
fidelities_linear = []
fidelities_log = []

# %% [markdown]
# We run noise-model simulations for circuit sizes from 3 to 9 qubits and compute the fidelity (the higher is better). The ideal noiseless circuit has fidelity 1 by construction.

# %%
qubits = range(3, 9)
# Test both linear and log GHZ circuits with noise model
for n in qubits:
    # Linear GHZ circuit
    linear_circuit = build_linear_ghz(n)

    # Log GHZ circuit
    log_circuit = build_log_ghz(n)

    # Apply noise model
    linear_noisy_circuit = utils.noise.transform_circuit(
        linear_circuit, model=noise_model
    )
    log_noisy_circuit = utils.noise.transform_circuit(log_circuit, model=noise_model)

    # Simulate noiseless circuits
    rho_linear = simulator.simulate(linear_circuit).final_density_matrix
    rho_log = simulator.simulate(log_circuit).final_density_matrix

    # Simulate noisy circuits
    rho_linear_noisy = simulator.simulate(linear_noisy_circuit).final_density_matrix
    rho_log_noisy = simulator.simulate(log_noisy_circuit).final_density_matrix

    # Calculate fidelities
    fidelity_linear = np.trace(rho_linear @ rho_linear_noisy).real
    fidelity_log = np.trace(rho_log @ rho_log_noisy).real

    # Store results
    fidelities_linear.append(fidelity_linear)
    fidelities_log.append(fidelity_log)

    print(f"n={n}:")
    print(f"  Linear GHZ: {fidelity_linear:.4f}")
    print(f"  Log GHZ: {fidelity_log:.4f}")


# %% [markdown]
# Fidelity comparison plot:

# %%
# Create comparison plot
plt.figure(figsize=(10, 6))

plt.plot(
    qubits,
    fidelities_linear,
    "ro-",
    label="Linear GHZ",
    linewidth=2,
    markersize=8,
)
plt.plot(
    qubits,
    fidelities_log,
    "bo-",
    label="Log-depth GHZ",
    linewidth=2,
    markersize=8,
)

plt.xlabel("Number of Qubits", fontsize=14)
plt.ylabel("Fidelity", fontsize=14)
plt.title(
    "GHZ State Fidelity Comparison: Linear vs Log-Depth Circuits",
    fontsize=16,
)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(qubits)

# Add annotations for better understanding
plt.text(
    0.02,
    0.98,
    "Higher fidelity = Better performance",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(
    f"Linear GHZ: Mean = {np.mean(fidelities_linear):.4f}, Std = {np.std(fidelities_linear):.4f}"
)
print(
    f"Log-depth GHZ: Mean = {np.mean(fidelities_log):.4f}, Std = {np.std(fidelities_log):.4f}"
)


# %% [markdown]
# The GHZ results show that parallelizing gates increases fidelity compared with the sequential implementation. The log-depth circuit consistently outperforms the linear-depth circuit, with the advantage growing as we increase the number of qubits.

# %% [markdown]
# ## Automatic toolkits for circuit parallelization
#
# In the GHZ example the log-depth circuit was provided manually. Bloqade also includes automatic tools that compress a deep circuit into a more parallel form by casting the problem as an optimization (integer linear program).
#
# We first transpile the circuit into a standard gate set consisting of single-qubit gates and two-qubit CZ gates. CZ gates acting on disjoint qubits commute and therefore can be placed in the same moment.
#
# The optimization objective encourages gates that can execute in parallel to be assigned to nearby moments. Formally, we minimize an objective such as $\sum_{g}\sum_{o_p,o_q \in g} w_g \left|t_p-t_q\right|$, where each group $g$ contains operations with a shared tag. Within each group, an attraction force of strength $w_g$ is introduced. Here $t_p$ and $t_q$ are integer labels (its epoch) of operators $o_p$ and $o_q$. Each operation (gate) can have multiple tags.
#
# To preserve circuit equivalence, we only reorder operations that commute. Execution dependencies are represented as a directed acyclic graph (DAG); each vertex $i$ receives an integer label $t_i$ (its epoch) and ordering constraints are expressed as inequalities on these labels. These constraints and objective terms are encoded in an integer linear program (ILP).
#
# Because the ILP formulation has totally unimodular constraint structure in our encoding, the relaxed linear program yields integer solutions, which makes the optimization efficient in practice. Absolute-value terms are reformulated into equivalent linear forms during construction.
#
# The helper `bloqade.utils.auto_similarity` tags operations and assigns weights to build the linear objective. Users can also add manual annotations (tags/weights) to guide the parallelizer when they need fine-grained control.
#
#

# %% [markdown]
# ## Example 2: [7,1,3] Steane code circuit

# %% [markdown]
# We construct a sequential circuit for the [7,1,3] Steane code and then apply Bloqade's parallelization utilities (which use `auto_similarity`) to compress it.


# %% [code]
def build_steane_code_circuit():
    """Build the Steane code circuit for error correction using squin and convert to Cirq."""

    @squin.kernel
    def steane_kernel():
        q = squin.qalloc(7)

        # H gate on qubits 1, 2, 3
        squin.h(q[1])
        squin.h(q[2])
        squin.h(q[3])

        # Encode the logical qubit with CZ and H gates (equivalent to CNOT)
        squin.h(q[0])
        squin.cz(q[1], q[0])
        squin.cz(q[2], q[0])
        squin.h(q[4])
        squin.cz(q[2], q[4])
        squin.cz(q[6], q[4])
        squin.h(q[5])
        squin.cz(q[6], q[5])
        squin.cz(q[3], q[5])
        squin.cz(q[1], q[5])
        squin.h(q[5])
        squin.h(q[6])
        squin.cz(q[1], q[6])
        squin.cz(q[2], q[6])
        squin.h(q[6])
        squin.cz(q[3], q[4])
        squin.h(q[4])
        squin.cz(q[3], q[0])
        squin.h(q[0])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(7)
    circuit = cirq_utils.emit_circuit(steane_kernel, circuit_qubits=qubits)
    return circuit


# %%
# Build Steane circuits (reuse already defined noise models and simulator)
steane_original = build_steane_code_circuit()
steane_parallel = utils.parallelize(circuit=steane_original)
steane_parallel = utils.remove_tags(steane_parallel)

# Display original circuit - renders nicely in Jupyter, shows text in terminal
print("Original Steane Circuit:")
try:
    get_ipython()  # Check if we're in IPython/Jupyter
    # In Jupyter: display as SVG (prettier visualization)
    from IPython.display import display

    display(SVGCircuit(steane_original))
except NameError:
    # Not in Jupyter: print text representation
    print(steane_original)

print("\nParallelized Steane Circuit:")
try:
    get_ipython()
    from IPython.display import display

    display(SVGCircuit(steane_parallel))
except NameError:
    print(steane_parallel)
print("Original Steane circuit depth:", len(steane_original))
print("Parallelized Steane circuit depth:", len(steane_parallel))


# %% [markdown]
# We perform noise analysis on both the original and parallelized Steane circuits.

# %%
# Apply noise model to both circuits
steane_original_noisy = utils.noise.transform_circuit(
    steane_original, model=noise_model
)
steane_parallel_noisy = utils.noise.transform_circuit(
    steane_parallel, model=noise_model
)

# Simulate ideal circuits
rho_original_ideal = simulator.simulate(steane_original).final_density_matrix
rho_parallel_ideal = simulator.simulate(steane_parallel).final_density_matrix

# Simulate noisy circuits
rho_original_noisy = simulator.simulate(steane_original_noisy).final_density_matrix
rho_parallel_noisy = simulator.simulate(steane_parallel_noisy).final_density_matrix

# Calculate fidelities
fidelity_original = np.trace(rho_original_ideal @ rho_original_noisy).real
fidelity_parallel = np.trace(rho_parallel_ideal @ rho_parallel_noisy).real

# Print results
print("\n=== Steane Code Circuit Fidelity Comparison ===")
print(f"Original circuit: {fidelity_original:.4f}")
print(f"Parallelized circuit: {fidelity_parallel:.4f}")

# Calculate improvement
improvement = fidelity_parallel - fidelity_original

print(f"\nFidelity improvement with parallelization: {improvement:+.4f}")


# Summary analysis
depth_reduction = len(steane_original) - len(steane_parallel)
depth_reduction_pct = (
    (depth_reduction / len(steane_original) * 100) if len(steane_original) > 0 else 0
)

print("\n=== Analysis Summary ===")
print(
    f"Circuit depth reduction: {len(steane_original)} → {len(steane_parallel)} moments"
)
print(f"Depth reduction: {depth_reduction} moments ({depth_reduction_pct:.1f}%)")
if improvement > 0:
    print("✓ Parallelization improves fidelity")
else:
    print("✗ Parallelization does not improve fidelity")


# %% [markdown]
# ## Example 3: Linear chained circuit
#
# Here is another example of a circuit of CZ gates in a linear chain. The original circuit has a linearly growing number of moments with the number of qubits.


# %%
def build_circuit3():
    """Build a linear CZ chain circuit using squin and convert to Cirq."""
    n = 10

    @squin.kernel
    def cz_chain_kernel():
        q = squin.qalloc(n)
        for i in range(n - 1):
            squin.cz(q[i], q[i + 1])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(cz_chain_kernel, circuit_qubits=qubits)
    return circuit


# Build the linear CZ circuit
circuit3 = build_circuit3()

# Display original circuit - renders nicely in Jupyter, shows text in terminal
print("Original CZ chain circuit:")
try:
    get_ipython()  # Check if we're in IPython/Jupyter
    # In Jupyter: display as SVG (prettier visualization)
    from IPython.display import display

    display(SVGCircuit(circuit3))
except NameError:
    # Not in Jupyter: print text representation
    print(circuit3)

# Parallelize the circuit, `auto_similarity` is automatically applied inside `parallelize`
circuit3_parallel = utils.parallelize(circuit=circuit3)

# Remove any tags and print the parallelized circuit
circuit3_parallel = utils.remove_tags(circuit3_parallel)
print("Parallelized CZ chain circuit:")
try:
    get_ipython()
    from IPython.display import display

    display(SVGCircuit(circuit3_parallel))
except NameError:
    print(circuit3_parallel)

print("Original CZ chain circuit depth:", len(circuit3))
print("Parallelized CZ chain circuit depth:", len(circuit3_parallel))
# %% [markdown]
# ## Example 4: 6D Hypercube Circuit
#
# This example demonstrates parallelization on a 6-dimensional hypercube with 64 qubits arranged in a 4×16 array, which can be implemented on our Gemini machine with (5x17) architecture.
#
# The 6D hypercube (2x2x2x2x2x2) is reshaped into two dimenison (4x16). Each qubit's index maps to (x, y) coordinates:
# - Binary index: [y1][y0][x3][x2][x1][x0]
# - x-coordinate (0-15): lower 4 bits
# - y-coordinate (0-3): upper 2 bits
#
# Each of the 6 dimensions connects qubits that differ by exactly one bit, creating a highly connected quantum circuit with 192 CZ gates across 64 qubits.
#


# %%
def build_hypercube():
    """
    Build a 6D hypercube circuit with 64 qubits arranged in a 4x16 array.

    The hypercube structure follows the mapping from Task4 analysis:
    - Binary index: [y1][y0][x3][x2][x1][x0]
    - x-coordinate (0-15): lower 4 bits [x3][x2][x1][x0]
    - y-coordinate (0-3): upper 2 bits [y1][y0]

    Each of the 6 dimensions connects qubits that differ by exactly one bit:
    - Dimension 0: flip bit 0 (x ± 1)
    - Dimension 1: flip bit 1 (x ± 2)
    - Dimension 2: flip bit 2 (x ± 4)
    - Dimension 3: flip bit 3 (x ± 8)
    - Dimension 4: flip bit 4 (y ± 1)
    - Dimension 5: flip bit 5 (y ± 2)
    """
    # Create a 4x16 array of qubits (64 qubits total)
    num_rows = 4  # y-coordinates: 0-3
    num_cols = 16  # x-coordinates: 0-15

    # Create GridQubits for 4x16 array
    qubits = np.array(cirq.GridQubit.rect(num_rows, num_cols)).reshape(
        num_rows, num_cols
    )

    # Helper function to convert (x, y) to hypercube index
    def coord_to_index(x, y):
        return (y << 4) | x

    # Helper function to convert hypercube index to (x, y)
    def index_to_coord(idx):
        x = idx & 0b001111  # Lower 4 bits
        y = (idx >> 4) & 0b11  # Upper 2 bits
        return x, y

    # Build CZ gates for all 6 dimensions of the hypercube
    # Each dimension connects qubits that differ by exactly one bit
    gates = []

    for dim in range(6):
        # For each qubit, connect it to its neighbor along dimension `dim`
        for idx in range(64):
            # Get the neighbor by flipping bit `dim`
            neighbor_idx = idx ^ (1 << dim)

            # Only add each edge once (avoid duplicates)
            if neighbor_idx > idx:
                # Convert indices to coordinates
                x1, y1 = index_to_coord(idx)
                x2, y2 = index_to_coord(neighbor_idx)

                # Verify coordinates are valid
                if (
                    0 <= x1 < num_cols
                    and 0 <= y1 < num_rows
                    and 0 <= x2 < num_cols
                    and 0 <= y2 < num_rows
                ):
                    # Add CZ gate
                    gates.append(cirq.CZ(qubits[y1, x1], qubits[y2, x2]))

    # Shuffle the gates to make parallelization more challenging
    # Without shuffling, gates are already grouped by dimension which makes
    # parallelization trivial. Shuffling forces the optimizer to work harder.
    import random

    random.seed(0)  # Use seed=0 for optimal parallelization (depth 6)
    random.shuffle(gates)

    # Create circuit with all gates
    # Gates are added sequentially; parallelization will be done later
    circuit = cirq.Circuit(gates)
    return circuit


# %%
# Build and analyze the hypercube circuit
hypercube_circuit = build_hypercube()
print("Original hypercube circuit:")
print("Number of qubits: 64 (arranged in 4×16 array)")
print(f"Number of gates: {len(list(hypercube_circuit.all_operations()))}")
print(f"Circuit depth: {len(hypercube_circuit)} moments")

# Calculate number of edges per dimension
# Each dimension should have 32 edges (64 qubits / 2)
print("\nExpected edges per dimension: 32")
print("Total edges across 6 dimensions: 192")

# %% [markdown]
# The original hypercube circuit has 192 CZ gates (32 edges per dimension × 6 dimensions).
# The gates are shuffled randomly to make parallelization more challenging - without shuffling,
# gates would already be grouped by dimension, making the problem trivial. After shuffling,
# Cirq's default sequential insertion creates a deeper circuit, providing a better benchmark
# for the parallelization optimizer.
#
# Visualization is generated separately. The 4×16 grid layout shows:
# - Straight lines for nearest-neighbor interactions (Euclidean distance = 1)
# - Curved arcs for longer-range interactions (distance > 1) to avoid overlapping
#
# ![Original Hypercube Circuit](figures/hypercube_original.png)
#
# The links with same color belongs to the same moment are executed symmataniously within one layer.

# %% [markdown]
# Now we try to parallel the hypbercube circuit.

# %%
# Parallelize the hypercube circuit. There're some randomness in the optimizer which leads to different results in different runs.
print("\nParallelizing hypercube circuit...")
hypercube_parallel = utils.parallelize(circuit=hypercube_circuit)
hypercube_parallel = utils.remove_tags(hypercube_parallel)

print(f"\nOriginal hypercube circuit depth: {len(hypercube_circuit)}")
print(f"Parallelized hypercube circuit depth: {len(hypercube_parallel)}")
depth_reduction = len(hypercube_circuit) - len(hypercube_parallel)
depth_reduction_pct = depth_reduction / len(hypercube_circuit) * 100
print(f"Depth reduction: {depth_reduction} moments ({depth_reduction_pct:.1f}%)")

# %% [markdown]
# The parallelization significantly reduces the circuit depth by identifying gates that act on
# disjoint qubits and can execute simultaneously. The hypercube's regular structure allows many
# gates to be grouped into the same moment.
#
# Visualization of the parallelized circuit shows gates colored by their assigned moment:
#
# ![Parallelized Hypercube Circuit](figures/hypercube_parallel.png)
#
# The parallelized circuit demonstrates that quantum algorithms on hypercube connectivity can
# benefit substantially from automatic parallelization, potentially improving fidelity by
# reducing overall execution time and exposure to decoherence.
