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
# We take the GHZ state preparation as an example. It prepares the state
#
# $\sqrt{2}|\psi\rangle = |000\cdots000\rangle + |111\cdots111\rangle$
#
# The GHZ state can be prepared using a sequence of Hadamard and CNOT gates. In a linear (sequential) implementation, the CNOT gates are applied one after another, resulting in a circuit depth that grows linearly with the number of qubits. In contrast, a log-depth (parallel) implementation arranges the CNOT gates so that multiple gates acting on disjoint qubits can execute simultaneously, reducing the overall depth to logarithmic in the number of qubits. This comes at the cost of requiring arbitrary connectivity, which is not native to all archetectures. However, it is perfect for reconfigurable neutral atom systems, which have a native "all to all" connectivity through mid-circuit atom shuttling.
# %%
import cirq
import matplotlib.pyplot as plt
import numpy as np
from bloqade import squin, cirq_utils
import bloqade.cirq_utils as utils
from cirq.contrib.svg import SVGCircuit

import warnings
warnings.filterwarnings("ignore")

# %%
def build_linear_ghz(n_qubits: int) -> cirq.Circuit:
    """
    Build a linear GHZ circuit using squin and convert to Cirq.
    Inputs:
    n_qubits: Number of qubits in the GHZ state.
    Returns:
    cirq.Circuit: The constructed linear GHZ circuit.
    """

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
    """
    Build logarithmic-depth GHZ circuit using squin and convert to Cirq.
    Inputs:
    n_qubits: Number of qubits in the GHZ state.
    Returns:
    cirq.Circuit: The constructed log-depth GHZ circuit.
    """

    max_iterations = int(np.ceil(np.log2(n_qubits))) if n_qubits > 1 else 1

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


linear_ghz = build_linear_ghz(12)
log_ghz = build_log_ghz(12)

# %%
SVGCircuit(linear_ghz)

# %%
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


# %% [markdown]
# We run noise-model simulations for circuit sizes from 3 to 9 qubits and compute the fidelity (the higher is better). The ideal noiseless circuit has fidelity 1 by construction.

# %%
# Scan a range of qubit numbers and compute fidelities
fidelities_linear = []
fidelities_log = []
num_qubits = list(range(2, 11))
# Test both linear and log GHZ circuits with noise model
for n in num_qubits:
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

# %% [markdown]
# Fidelity comparison plot:

# %%
# Create comparison plot
plt.figure(figsize=(10, 6))

plt.plot(
    num_qubits,
    fidelities_linear,
    "ro-",
    label="Linear GHZ",
    linewidth=2,
    markersize=8,
)
plt.plot(
    num_qubits,
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
plt.xticks(num_qubits)
plt.axis([1.5,10.5,0.6,1.0])
# Add annotations for better understanding
plt.text(
    0.15,
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
# The GHZ results show that parallelizing gates increases fidelity compared with the sequential implementation. The log-depth circuit consistently outperforms the linear-depth circuit, with the advantage growing as we increase the number of qubits. Observe that there is a jump in the fidelity at every power of two, corresponding to the addition of a new level in the log-depth circuit.

# %% [markdown]
# ## Automatic toolkits for circuit parallelization
#
# Bloqade provides automatic tools to compress circuits into more parallel forms:
#
# ```python
# import bloqade.cirq_utils as utils
#
# # Parallelize a circuit
# parallel_circuit = utils.parallelize(circuit)
#
# # Remove internal tags (for cleaner visualization)
# parallel_circuit = utils.remove_tags(parallel_circuit)
# ```
#
# The algorithm builds a DAG of gate dependencies (only commuting gates can be reordered), then solves an integer linear program (ILP) to assign gates to moments while minimizing circuit depth. Similar gates are attracted to the same moment via weighted objectives.
#

# %% [markdown]
# ## Example 2: [7,1,3] Steane code circuit
# Lets explore manual and automatic parallelism optimization on the Steane code, which is a prototypical quantum error correcting code that encodes one logical qubit into seven physical qubits, and can correct a single qubit error.

# We construct several versions of the [7,1,3] Steane code encoder circuit, based on three different initial circuits. The `seq` circuit is designed to be the "worst" possible version of the Steane code, with as much sequential operation as possible. The `11-CNOT` circuit is the textbook version of the Steane code, which uses 11 CNOT gates to perform the encoding. The `9-CZ` circuit is an optimized version that reduces the number of entangling gates to 9 CZ gates by using √Y and √Y† gates instead of Hadamards.
#
# | Version | Description | Parallelization |
# |---------|-------------|-----------------|
# | seq | Sequential circuit using CZ gates (native to neutral atoms) | Manual |
# | seq-auto | Auto-parallelized sequential circuit | Auto |
# | 11-CNOT | Textbook encoder using 11 CNOT gates | Manual |
# | 11-CNOT-auto | Auto-parallelized 11-CNOT circuit | Auto |
# | 9-CZ | Optimized encoder using only 9 CZ gates with √Y gates | Manual |
# | 9-CZ-auto | Auto-parallelized 9-CZ circuit | Auto |
#


# %%
def build_steane_code_circuit():
    """Build the Steane code circuit (version a) using CZ gates - native to neutral atoms, but designed to be as sequential as possible."""

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


def build_steane_11cnot()-> cirq.Circuit:
    """Build the Steane code encoder (version b) with 11 CNOT gates - textbook version.

    This is the standard Steane code encoder circuit where:
    - Qubit 6 is the data qubit |ψ⟩ to be encoded
    - Qubits 0-5 are ancillas initialized to |0⟩
    - H gates prepare superposition on control qubits
    - 11 CNOT gates create the encoded state
    """

    @squin.kernel
    def steane_11cnot_kernel():
        q = squin.qalloc(7)

        # H gates on qubits 1, 2, 3 (ancilla preparation)
        squin.h(q[1])
        squin.h(q[2])
        squin.h(q[3])

        # 11 CNOT gates following textbook Steane code structure
        # First layer of CNOTs
        squin.cx(q[6], q[5])
        squin.cx(q[1], q[0])
        squin.cx(q[2], q[4])
        squin.cx(q[2], q[0])
        squin.cx(q[3], q[5])
        squin.cx(q[1], q[5])
        squin.cx(q[6], q[4])
        squin.cx(q[2], q[6])
        squin.cx(q[3], q[4])
        squin.cx(q[3], q[0])
        squin.cx(q[1], q[6])

    qubits = cirq.LineQubit.range(7)
    circuit = cirq_utils.emit_circuit(steane_11cnot_kernel, circuit_qubits=qubits)
    return circuit


def build_steane_9cnot()-> cirq.Circuit:
    """Build the optimized Steane code encoder (version c) with only 9 CNOT gates.

    This optimized version uses √Y and √Y† gates instead of some Hadamards,
    reducing the CNOT count from 11 to 9 while maintaining circuit equivalence.

    The optimization exploits the structure of the Steane code to eliminate
    redundant entangling operations.
    """

    @squin.kernel
    def steane_9cnot_kernel():
        q = squin.qalloc(7)

        # Initial √Y† layer on ancilla qubits (replaces H gates)
        # √Y† = Ry(-π/2)
        squin.ry(-np.pi / 2, q[0])
        squin.ry(-np.pi / 2, q[1])
        squin.ry(-np.pi / 2, q[2])
        squin.ry(-np.pi / 2, q[3])
        squin.ry(-np.pi / 2, q[4])
        squin.ry(-np.pi / 2, q[5])

        # First CZ layer (parallel)
        squin.cz(q[1], q[2])
        squin.cz(q[3], q[4])
        squin.cz(q[5], q[6])

        # √Y layer
        squin.ry(np.pi / 2, q[6])

        # Second CZ layer (parallel)
        squin.cz(q[0], q[3])
        squin.cz(q[2], q[5])
        squin.cz(q[4], q[6])

        # √Y layer from 2 to 6
        squin.ry(np.pi / 2, q[2])
        squin.ry(np.pi / 2, q[3])
        squin.ry(np.pi / 2, q[4])
        squin.ry(np.pi / 2, q[5])
        squin.ry(np.pi / 2, q[6])

        # Third CZ layer (parallel)
        squin.cz(q[0], q[1])
        squin.cz(q[2], q[3])
        squin.cz(q[4], q[5])

        # Final √Y layer
        squin.ry(np.pi / 2, q[1])
        squin.ry(np.pi / 2, q[2])
        squin.ry(np.pi / 2, q[4])

    qubits = cirq.LineQubit.range(7)
    circuit = cirq_utils.emit_circuit(steane_9cnot_kernel, circuit_qubits=qubits)
    return circuit


# %%
# Build all Steane circuit versions (reuse already defined noise models and simulator)
steane_seq = build_steane_code_circuit()  # Sequential CZ-based
steane_seq_auto = utils.parallelize(circuit=steane_seq)  # Auto-parallelized
steane_seq_auto = utils.remove_tags(steane_seq_auto)

steane_11cnot = build_steane_11cnot()  # 11 CNOT textbook
steane_11cnot_auto = utils.parallelize(circuit=steane_11cnot)  # Auto-parallelized
steane_11cnot_auto = utils.remove_tags(steane_11cnot_auto)

steane_9cz = build_steane_9cnot()  # 9 CZ optimized
steane_9cz_auto = utils.parallelize(circuit=steane_9cz)  # Auto-parallelized
steane_9cz_auto = utils.remove_tags(steane_9cz_auto)

# %% [markdown]
# ### seq: Sequential CZ-based Steane Circuit

# %%
SVGCircuit(steane_seq)

# %% [markdown]
# ### seq-auto: Auto-Parallelized Sequential Circuit

# %%
SVGCircuit(steane_seq_auto)

# %% [markdown]
# ### 11-CNOT: Textbook Steane Encoder

# %%
SVGCircuit(steane_11cnot)

# %% [markdown]
# ### 11-CNOT-auto: Auto-Parallelized 11-CNOT Circuit

# %%
SVGCircuit(steane_11cnot_auto)

# %% [markdown]
# ### 9-CZ: Optimized Steane Encoder

# %%
SVGCircuit(steane_9cz)

# %% [markdown]
# ### 9-CZ-auto: Auto-Parallelized 9-CZ Circuit

# %%
SVGCircuit(steane_9cz_auto)

# %% [markdown]
# ### Circuit Depths
# A lower depth is heuristically better than higher depth due to spectator errors on idle qubits.

# %%
print(f"seq:          {len(steane_seq)} moments")
print(f"seq-auto:     {len(steane_seq_auto)} moments")
print(f"11-CNOT:      {len(steane_11cnot)} moments")
print(f"11-CNOT-auto: {len(steane_11cnot_auto)} moments")
print(f"9-CZ:         {len(steane_9cz)} moments")
print(f"9-CZ-auto:    {len(steane_9cz_auto)} moments")


# %% [markdown]
# ### Noise Analysis

# %%
# Compute fidelities for all circuit versions
steane_circuits = {
    "seq": steane_seq,
    "seq-auto": steane_seq_auto,
    "11-CNOT": steane_11cnot,
    "11-CNOT-auto": steane_11cnot_auto,
    "9-CZ": steane_9cz,
    "9-CZ-auto": steane_9cz_auto,
}

steane_fidelities = {}
for name, circuit in steane_circuits.items():
    noisy = utils.noise.transform_circuit(circuit, model=noise_model)
    rho_ideal = simulator.simulate(circuit).final_density_matrix
    rho_noisy = simulator.simulate(noisy).final_density_matrix
    steane_fidelities[name] = np.trace(rho_ideal @ rho_noisy).real

# Print summary
print(f"{'Version':<15} {'Depth':<10} {'Fidelity':<10}")
print("-" * 35)
for name, circuit in steane_circuits.items():
    print(f"{name:<15} {len(circuit):<10} {steane_fidelities[name]:.4f}")

best_version = max(steane_fidelities, key=steane_fidelities.get)
print(f"\nBest: {best_version} ({steane_fidelities[best_version]:.4f})")

# %% [markdown]
# Fidelity comparison plot for all Steane code versions:

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

labels = ["seq", "seq\nauto", "11-CNOT", "11-CNOT\nauto", "9-CZ", "9-CZ\nauto"]
fidelity_vals = list(steane_fidelities.values())
depth_vals = [len(c) for c in steane_circuits.values()]
colors = ["#c0392b", "#e74c3c", "#d68910", "#f4d03f", "#1e8449", "#58d68d"]

for ax, vals, ylabel, title in [
    (ax1, fidelity_vals, "Fidelity", "Steane Code: Fidelity"),
    (ax2, depth_vals, "Circuit Depth", "Steane Code: Depth"),
]:
    bars = ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.3f}" if isinstance(v, float) else str(v)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            fmt,
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
ax1.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# As expected, the manual and optimized circuits do better than their naively optimized counterparts. The "worst case" sequential circuit has the lowest fidelity, while the auto-optimized 9 CZ circuit has the highest fidelity. However, this also comes with a point of warning: the noise model is not a perfect representation of real hardware. In practice, the hand optimized 9-CZ circuit was implemented as part of QuEra's [magic state distillation paper](https://arxiv.org/abs/2412.15165), which suggests that the noise model is not aligned with hardware. The next steps after manual and automatic optimization should be implementation and tuning on real hardware.

# %% [markdown]
# ## Example 4: QAOA / graph state preparation
#
# As a final example, let's consider a circuit for a graph-based algorithm:
# QAOA on MaxCut. The circuit is a variational ansatz that alternates between entangling
# phasor gates (encoding the objective) and a single-qubit mixer layer.
#
# For MaxCut, there is a qubit for each vertex, and the phasor consists of CZPhase gates
# for each edge. The ansatz is repeated $p$ times, and in the $p \to \infty$ limit recovers the exact state.
#
# These graph-based circuits are inherently parallel: CZPhase gates commute, so optimal parallelization
# can be found via edge coloring of the graph, where each color corresponds to a circuit moment.
# This lets us pull some tricks in manually optimizing the circuit depth in a way that automatic parallelism and transpilation cannot easily do. Lets construct three different versions of the QAOA circuit:
# - **Naive**: Sequential circuit without optimization
# - **Auto-parallel**: Using `utils.parallelize()` for automatic optimization
# - **Hand-tuned**: Manual parallelization via edge coloring

# %%
import networkx as nx


def build_qaoa_circuit(
    graph: nx.Graph, gamma: list[float], beta: list[float]
) -> cirq.Circuit:
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


def build_qaoa_circuit_parallelized(
    graph: nx.Graph, gamma: list[float], beta: list[float]
) -> cirq.Circuit:
    """Build and parallelize a QAOA circuit for MaxCut on the given graph using squin"""
    n = len(graph.nodes)
    assert len(gamma) == len(beta), "Length of gamma and beta must be equal"

    # A smarter implementation would use the Misra–Gries algorithm,
    # which gives a guaranteed Δ+1 coloring, consistent with
    # Vizing's theorem for edge coloring.
    # However, networkx does not have an implementation of this algorithm,
    # so we use greedy coloring as an approximation. This does not guarantee
    # optimal depth, but works reasonably well in practice.
    linegraph = nx.line_graph(graph)
    best = 1e99
    for strategy in [
        "largest_first",
        "random_sequential",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "saturation_largest_first",
    ]:
        coloring: dict = nx.coloring.greedy_color(linegraph, strategy=strategy)
        num_colors = len(set(coloring.values()))
        if num_colors < best:
            best = num_colors
            best_coloring = coloring
    coloring: dict = best_coloring
    colors = [
        [edge for edge, color in coloring.items() if color == c]
        for c in set(coloring.values())
    ]

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
    circuit3 = cirq.optimize_for_target_gateset(
        circuit2, gateset=cirq.CZTargetGateset()
    )

    return circuit3



# %%
# Build circuits on a small graph for visualization and fidelity comparison
graph = nx.random_regular_graph(d=3, n=10, seed=42)

qaoa_naive = build_qaoa_circuit(graph, gamma=[np.pi / 2], beta=[np.pi / 4])
qaoa_parallel = build_qaoa_circuit_parallelized(
    graph, gamma=[np.pi / 2], beta=[np.pi / 4]
)
qaoa_autoparallel = utils.parallelize(qaoa_naive)

print(f"Naive circuit depth:       {len(qaoa_naive)}")
print(f"Auto-parallel depth:       {len(qaoa_autoparallel)}")
print(f"Hand-tuned parallel depth: {len(qaoa_parallel)}")

# %% [markdown]
# The depth of the hand-tuned circuit is much lower, and in fact ends up being constant in the degree of the graph consistent with Vizing's theorem.

# ### Circuit Visualization

# %%
SVGCircuit(qaoa_naive)

# %%
SVGCircuit(qaoa_parallel)

# %% [markdown]
# ### Edge Coloring Visualization
#
# The parallelization can be visualized as an edge coloring of the graph,
# where edges of the same color can be executed in the same moment.


# %%
def visualize_graph_with_edge_coloring(
    graph: nx.Graph, colors: list, title: str, pos: dict, hadamard_qubits: set
):
    """Visualize graph with colored edges and arrows indicating control -> target direction.

    Arrow points from control to target, where target is the qubit receiving the Hadamard gate.
    Following the convention in build_qaoa_circuit_parallelized:
    - If u in hadamard_qubits: target=u, control=v
    - Else: target=v, control=u
    """
    fig,ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")

    edge_colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    for color_idx, color_group in enumerate(colors):
        edge_color = edge_colors[color_idx % len(edge_colors)]
        for edge in color_group:
            u, v = edge
            # Match the convention in build_qaoa_circuit_parallelized
            if u in hadamard_qubits:
                ctrl, tgt = v, u  # u gets H, so u is target
            else:
                ctrl, tgt = u, v  # v gets H, so v is target
            x1, y1 = pos[ctrl]
            x2, y2 = pos[tgt]

            # Draw edge line
            plt.plot([x1, x2], [y1, y2], color=edge_color, linewidth=2.5,zorder=-1)

            # Draw arrow at midpoint pointing from control to target
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            plt.arrow(
                mid_x - dx * 0.08,
                mid_y - dy * 0.08,
                dx * 0.16,
                dy * 0.16,
                head_width=0.04,
                head_length=0.02,
                fc=edge_color,
                ec=edge_color,
                linewidth=1.5,
            )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=edge_colors[i % len(edge_colors)],
            lw=4,
            label=f"Moment {i} ({len(colors[i])} edges)",
        )
        for i in range(len(colors))
    ]
    # Add arrow explanation to legend
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            color="black",
            lw=0,
            marker=">",
            markersize=10,
            label="Arrow: ctrl → tgt (H gate)",
        )
    )
    plt.legend(handles=legend_elements, loc="upper left", fontsize=10)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# Get edge coloring for hand-tuned circuit
pos = nx.kamada_kawai_layout(graph)
linegraph = nx.line_graph(graph)
best_coloring = min(
    [
        nx.coloring.greedy_color(linegraph, strategy=s)
        for s in ["largest_first", "smallest_last", "saturation_largest_first"]
    ],
    key=lambda c: len(set(c.values())),
)
colors_parallel = [
    [e for e, c in best_coloring.items() if c == i] for i in set(best_coloring.values())
]

# Calculate Hadamard qubits (target qubits = complement of MIS)
mis = nx.algorithms.approximation.maximum_independent_set(graph)
hadamard_qubits = set(graph.nodes) - set(mis)

visualize_graph_with_edge_coloring(
    graph,
    colors_parallel,
    f"Hand-Tuned Parallelization: {len(colors_parallel)} moments",
    pos=pos,
    hadamard_qubits=hadamard_qubits,
)

# %% [markdown]
# ### Fidelity Comparison

# %%
qaoa_circuits = {
    "Naive": qaoa_naive,
    "Auto-parallel": qaoa_autoparallel,
    "Hand-tuned": qaoa_parallel,
}
qaoa_fidelities = {}
for name, circuit in qaoa_circuits.items():
    noisy = utils.noise.transform_circuit(circuit, model=noise_model)
    rho_ideal = simulator.simulate(circuit).final_density_matrix
    rho_noisy = simulator.simulate(noisy).final_density_matrix
    qaoa_fidelities[name] = np.trace(rho_ideal @ rho_noisy).real

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
labels = list(qaoa_circuits.keys())
fidelity_vals = list(qaoa_fidelities.values())
depth_vals = [len(c) for c in qaoa_circuits.values()]
colors = ["#c0392b", "#d68910", "#1e8449"]

for ax, vals, ylabel, title in [
    (ax1, fidelity_vals, "Fidelity", "QAOA: Fidelity"),
    (ax2, depth_vals, "Circuit Depth", "QAOA: Depth"),
]:
    bars = ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.3f}" if isinstance(v, float) else str(v)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            fmt,
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
ax1.set_ylim(0, 1)
plt.tight_layout()
plt.show()
# %% [markdown]
# From the results, we can see that the manually parallelized circuit achieves the best fidelity,
# followed closely by the auto-parallelized version. Both parallelized circuits outperform
# the naive sequential implementation, demonstrating the effectiveness of parallelization and gate
# optimization techniques in improving circuit performance under realistic noise models.
