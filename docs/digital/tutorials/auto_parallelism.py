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

linear_ghz = build_linear_ghz(8)
SVGCircuit(linear_ghz)
log_ghz = build_log_ghz(8)
SVGCircuit(log_ghz)

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

#%% [markdown]
# ### QAOA / graph state preparation
#
# As a final example, lets consider a more real-world example of a circuit for a graph-based algorithm:
# QAOA on MaxCut. The circuit is a variational ansatz that alternates between some entangling
# phasor gates that encodes the objective, and a single qubit mixer layer. A common choice of combinatorial
# problem is MaxCut, where the goal is to partition the nodes of a graph into two sets such that
# the number of edges between the sets is maximized. In this case, there is a qubit for each vertex,
# and the phasor consists of CZPhase gates for each edge in the graph. The ansatz is repeated p times,
# and in the p-> infty limit recovers the exact state.
#
# These sorts of graph-based circuits are inherently parallel, and serve as a good example for this tutorial.
# In particular, the CZPhase gates commute, and so an optimal parallelization can be found via an (approximate)
# edge coloring of the graph, where each color corresponds to a moment of the circuit.
# Additionally, a naive decomposition of CZPhase gates into CZ and single qubit rotations can lead to a lot of
# redundant single and multi-qubit gates that can be eliminated to improve fidelity.


#%%
import networkx as nx


def build_qaoa_circuit(graph: nx.Graph, gamma: list[float], beta:list[float]) -> cirq.Circuit:
    """Build a QAOA circuit for MaxCut on the given graph using squin"""
    n = len(graph.nodes)
    assert len(gamma) == len(beta), "Length of gamma and beta must be equal"

    # Prepare edge list for squin kernel
    edges = list(graph.edges)

    @squin.kernel
    def qaoa_kernel():
        q = squin.qalloc(n)

        # Initial Hadamard layer
        for i in range(n):
            squin.h(q[i])

        # QAOA layers
        for layer in range(len(gamma)):
            # Cost Hamiltonian: ZZ rotation for each edge
            # Using decomposition: exp(-i*gamma/2*Z⊗Z) = H → CZ → Rx(gamma) → CZ → H
            for edge in edges:
                u = edge[0]
                v = edge[1]
                squin.h(q[v])
                squin.cz(q[u], q[v])
                squin.rx(gamma[layer], q[v])
                squin.cz(q[u], q[v])
                squin.h(q[v])

            # Mixer Hamiltonian: Rx rotation on all qubits
            for i in range(n):
                squin.rx(2 * beta[layer], q[i])

    # Create LineQubits and emit circuit
    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(qaoa_kernel, circuit_qubits=qubits)

    # Convert to the native CZ gateset
    circuit2 = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    return circuit2


def build_qaoa_circuit_parallelized(graph: nx.Graph, gamma: list[float], beta:list[float]) -> cirq.Circuit:
    """Build and parallelize a QAOA circuit for MaxCut on the given graph using squin"""
    n = len(graph.nodes)
    assert len(gamma) == len(beta), "Length of gamma and beta must be equal"

    # A smarter implementation would use the Misra–Gries algorithm,
    # which gives a guaranteed Delta+1 coloring, consistent with
    # Vizing's theorem for edge coloring.
    # However, networkx does not have an implementation of this algorithm,
    # so we use greedy coloring as an approximation. This does not guarantee
    # optimal depth, but works reasonably well in practice.
    linegraph = nx.line_graph(graph)
    best = 1e99
    for strategy in ["largest_first", "random_sequential", "smallest_last", "independent_set",
                     "connected_sequential_bfs", "connected_sequential_dfs", "saturation_largest_first"]:
        coloring:dict = nx.coloring.greedy_color(linegraph, strategy=strategy)
        num_colors = len(set(coloring.values()))
        if num_colors < best:
            best = num_colors
            best_coloring = coloring
    coloring:dict = best_coloring
    colors = [[edge for edge, color in coloring.items() if color == c] for c in set(coloring.values())]

    print(len(colors))
    # For QAOA MaxCut, we need exp(i*gamma/2*Z⊗Z) per edge.
    # We decompose this using CZ and single-qubit rotations:
    #
    # exp(-i*gamma/2*Z⊗Z)  =  -------o----------o-------
    #                                 |          |
    #                         -----H--o--Rx(g)--o--H----
    #
    # where Rx(gamma) = X^(gamma/pi) in Cirq notation.

    # To cancel repeated Hadamards, we can select which qubit
    # of each gate pair to apply the Hadamards on. The minimum
    # number of Hadamards is equal to the size of the minimum vertex cover
    # of the graph. Finding the minimum vertex cover is NP-hard,
    # but we can use a greedy MIS heuristic instead.
    # The complement of the MIS is a minimum vertex cover.
    mis = nx.algorithms.approximation.maximum_independent_set(graph)
    hadamard_qubits = set(graph.nodes) - set(mis)

    # Prepare data structures for squin kernel
    # Flatten color groups and create parallel lists for indices
    all_edges = []
    h_qubits = []
    for color_group in colors:
        for edge in color_group:
            all_edges.append(edge)
            u, v = edge
            if u in hadamard_qubits:
                h_qubits.append(u)
            else:
                h_qubits.append(v)

    # Build the circuit using squin
    @squin.kernel
    def qaoa_parallel_kernel():
        q = squin.qalloc(n)

        # Initial Hadamard layer
        for i in range(n):
            squin.h(q[i])

        # QAOA layers
        for layer in range(len(gamma)):
            # Cost Hamiltonian: process edges in order
            edge_start = 0
            for color_group in colors:
                group_size = len(color_group)

                # First Hadamard layer
                for i in range(group_size):
                    h_qubit = h_qubits[edge_start + i]
                    squin.h(q[h_qubit])

                # First CZ layer
                for i in range(group_size):
                    edge = color_group[i]
                    u = edge[0]
                    v = edge[1]
                    squin.cz(q[u], q[v])

                # Rotation layer (Rx)
                for i in range(group_size):
                    h_qubit = h_qubits[edge_start + i]
                    squin.rx(gamma[layer], q[h_qubit])

                # Second CZ layer
                for i in range(group_size):
                    edge = color_group[i]
                    u = edge[0]
                    v = edge[1]
                    squin.cz(q[u], q[v])

                # Second Hadamard layer
                for i in range(group_size):
                    h_qubit = h_qubits[edge_start + i]
                    squin.h(q[h_qubit])

                edge_start = edge_start + group_size

            # Mixer Hamiltonian: Rx rotation on all qubits
            for i in range(n):
                squin.rx(2 * beta[layer], q[i])

    # Create LineQubits and emit circuit
    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(qaoa_parallel_kernel, circuit_qubits=qubits)

    # This circuit will have some redundant doubly-repeated Hadamards that can be removed.
    # Lets do that now by merging single qubit gates to phased XZ gates, which is the native
    # single-qubit gate on neutral atoms.
    circuit2 = cirq.merge_single_qubit_moments_to_phxz(circuit)
    # Do any last optimizing...
    circuit3 = cirq.optimize_for_target_gateset(circuit2, gateset=cirq.CZTargetGateset())

    return circuit3

#%% [markdown]
# Now we can compare the depth of the naive version that is not hardware aware,
# to the hand-tuned parallelized version.
#
# There is a third intermediate option between a fully naive and fully hand-tuned
# circuit, by using autoparallelism. This uses an integer linear program solver
# to minimize the average depth of the circuit by re-ordering commuting gates.
# For more details, see [this page]().
#%%
graph = nx.random_regular_graph(d=3, n=40, seed=42)
qaoa_naive = build_qaoa_circuit(graph, gamma=[np.pi/2], beta=[np.pi/4])
qaoa_autoparallel = utils.parallelize(qaoa_naive)
qaoa_parallel = build_qaoa_circuit_parallelized(graph, gamma=[np.pi/2], beta=[np.pi/4])
print("Naive QAOA circuit depth:    ", len(qaoa_naive))
print("Auto'd QAOA circuit depth:   ", len(qaoa_autoparallel))
print("Parallel QAOA circuit depth: ", len(qaoa_parallel))
#%% [markdown]
# We can see that the hand-tuned parallel version has the lowest depth and thus
# will have the best performance. The autoparallelized version is slightly better than the naive.
# Lets check out what the circuits look like, and simulate them on smaller graphs to compare fidelities.
#%%


def visualize_graph_with_edge_coloring(graph: nx.Graph, colors: list, title: str, hadamard_qubits: set, pos: dict):
    """Visualize graph with colored edges and arrows indicating control → target direction."""
    from matplotlib.patches import FancyArrowPatch

    plt.figure(figsize=(10, 10))

    # Draw all nodes with same color
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(graph, pos, font_size=14, font_weight='bold')

    # Define color palette for edge groups
    edge_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Draw edges with colors and arrows
    for color_idx, color_group in enumerate(colors):
        edge_color = edge_colors[color_idx % len(edge_colors)]
        for edge in color_group:
            u, v = edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Draw line
            plt.plot([x1, x2], [y1, y2], color=edge_color, linewidth=3, zorder=1)

            # Add arrow in the middle pointing from u (control) to v (target with H)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dx = x2 - x1
            dy = y2 - y1

            plt.arrow(mid_x - dx*0.05, mid_y - dy*0.05,
                     dx*0.1, dy*0.1,
                     head_width=0.04, head_length=0.03,
                     fc=edge_color, ec=edge_color,
                     linewidth=2, zorder=2)

    # Create legend
    legend_elements = [plt.Line2D([0], [0], color=edge_colors[i % len(edge_colors)],
                                 lw=4, label=f'{i}\'s Moment ({len(colors[i])} edges)')
                      for i in range(len(colors))]
    legend_elements.append(
        FancyArrowPatch((0, 0), (0.1, 0), arrowstyle='->', mutation_scale=20,
                       linewidth=2, color='black', label='Arrow: control → target')
    )
    plt.legend(handles=legend_elements, loc='upper left', fontsize=14, frameon=True, shadow=True)

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return pos


# Create test graph
graph = nx.random_regular_graph(d=3, n=10, seed=42)

# Build all three circuits
qaoa_naive = build_qaoa_circuit(graph, gamma=[np.pi/2], beta=[np.pi/4])
qaoa_parallel = build_qaoa_circuit_parallelized(graph, gamma=[np.pi/2], beta=[np.pi/4])
qaoa_autoparallel = utils.parallelize(qaoa_naive)

# Use consistent positions for all visualizations
pos = nx.spring_layout(graph, seed=42)

# Get edge coloring for hand-tuned parallelized circuit
linegraph = nx.line_graph(graph)
best = 1e99
best_coloring = None
for strategy in ["largest_first", "random_sequential", "smallest_last", "independent_set",
                 "connected_sequential_bfs", "connected_sequential_dfs", "saturation_largest_first"]:
    coloring = nx.coloring.greedy_color(linegraph, strategy=strategy)
    num_colors = len(set(coloring.values()))
    if num_colors < best:
        best = num_colors
        best_coloring = coloring
colors_parallel = [[edge for edge, color in best_coloring.items() if color == c] for c in set(best_coloring.values())]

# Extract edge coloring from auto-parallelized circuit
colors_autoparallel = []
for moment in qaoa_autoparallel:
    cz_edges = []
    for op in moment:
        if isinstance(op.gate, cirq.CZPowGate) or isinstance(op.gate, type(cirq.CZ)):
            qubits = op.qubits
            u = qubits[0].x
            v = qubits[1].x
            if graph.has_edge(u, v) or graph.has_edge(v, u):
                if graph.has_edge(u, v):
                    cz_edges.append((u, v))
                else:
                    cz_edges.append((v, u))
    if cz_edges:
        colors_autoparallel.append(cz_edges)

# Calculate Hadamard qubits (minimum vertex cover = complement of MIS)
mis = nx.algorithms.approximation.maximum_independent_set(graph)
hadamard_qubits = set(graph.nodes) - set(mis)

# Visualize hand-tuned parallelized circuit
print(f"Hand-tuned parallelized circuit: {len(colors_parallel)} color groups")
visualize_graph_with_edge_coloring(
    graph, colors_parallel,
    f"Hand-Tuned Parallelized QAOA\n{len(colors_parallel)} color groups for parallel execution",
    hadamard_qubits=hadamard_qubits,
    pos=pos
);

# Visualize auto-parallelized circuit
print(f"Auto-parallelized circuit: {len(colors_autoparallel)} color groups")
visualize_graph_with_edge_coloring(
    graph, colors_autoparallel,
    f"Auto-Parallelized QAOA\n{len(colors_autoparallel)} color groups for parallel execution",
    hadamard_qubits=hadamard_qubits,
    pos=pos
);
#%%
SVGCircuit(qaoa_naive)
#%%
SVGCircuit(qaoa_parallel)
# %%

# Apply noise model
qaoa_naive_noisy = utils.noise.transform_circuit(
    qaoa_naive, model=noise_model
)
qaoa_parallel_noisy = utils.noise.transform_circuit(qaoa_parallel, model=noise_model)
qaoa_autoparallel_noisy = utils.noise.transform_circuit(qaoa_autoparallel, model=noise_model)

# Simulate noiseless circuits
rho_naive = simulator.simulate(qaoa_naive).final_density_matrix
rho_parallel = simulator.simulate(qaoa_parallel).final_density_matrix
rho_autoparallel = simulator.simulate(qaoa_autoparallel).final_density_matrix

# Simulate noisy circuits
rho_naive_noisy = simulator.simulate(qaoa_naive_noisy).final_density_matrix
rho_parallel_noisy = simulator.simulate(qaoa_parallel_noisy).final_density_matrix
rho_autoparallel_noisy = simulator.simulate(qaoa_autoparallel_noisy).final_density_matrix

# Calculate fidelities
fidelity_naive = np.trace(rho_naive @ rho_naive_noisy).real
fidelity_parallel = np.trace(rho_parallel @ rho_parallel_noisy).real
fidelity_autoparallel = np.trace(rho_autoparallel @ rho_autoparallel_noisy).real
print(f"Naive QAOA circuit fidelity: {fidelity_naive:.4f}"
      )
print(f"Parallel QAOA circuit fidelity: {fidelity_parallel:.4f}"
      )
print(f"Auto-Parallel QAOA circuit fidelity: {fidelity_autoparallel:.4f}"
      )
# %%
