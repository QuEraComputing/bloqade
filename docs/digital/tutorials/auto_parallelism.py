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
# Parallelism lets gates that act on disjoint qubits execute at the same time, reducing circuit depth and overall runtime. On neutral-atom processors, many transversal operations (same gate type and parameters) can often be executed together in a single layer.
#
# Reducing depth typically improves fidelity and increases the number of operations that can complete within the hardware's coherence time.
#
# Bloqade supports both automatic and manual parallelization. The examples below show both methods and compare fidelity using representative noise models.
#

# %% [markdown]
# ## Example 1: GHZ Circuit
#
# We start with a simple example: GHZ state preparation implemented in a linear (sequential) form and in a log-depth parallel form. We'll run noise simulations for both circuits and compare their fidelities as we scale the number of qubits.
#
# See our blog post [Simulating noisy circuits for near-term quantum hardware](https://bloqade.quera.com/latest/blog/2025/07/30/simulating-noisy-circuits-for-near-term-quantum-hardware/) for detailed information about the noise model used here. The analysis workflow is:
#
# 1. Build a noiseless (ideal) circuit.
# 2. Choose a noise model (we use both one-zone and two-zone Gemini models).
# 3. Apply the noise model to the circuit to produce a noisy circuit.
# 4. Simulate the noisy circuit to obtain the final density matrix.
# 5. Simulate the ideal circuit and compare its state to the noisy density matrix to compute fidelity.
#
#

# %%
import bloqade.cirq_utils as utils
import cirq
import matplotlib.pyplot as plt
import numpy as np


# %%
def build_linear_ghz(n_qubits: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


def build_log_ghz(n_qubits: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    width = 1
    while width < n_qubits:
        for i in range(0, width, 1):
            if i + width > n_qubits - 1:
                break
            circuit.append(cirq.CNOT(qubits[i], qubits[i + width]))
        width *= 2
    return circuit


print(build_log_ghz(7))

# Initialize noise models
one_zone_model = utils.noise.GeminiOneZoneNoiseModel()
two_zone_model = utils.noise.GeminiTwoZoneNoiseModel()
simulator = cirq.DensityMatrixSimulator()

# Initialize lists to store fidelities
fidelities_linear_one_zone = []
fidelities_linear_two_zone = []
fidelities_log_one_zone = []
fidelities_log_two_zone = []


# %% [markdown]
# We run noise-model simulations for circuit sizes from 3 to 9 qubits and compute the fidelity (higher is better). The ideal noiseless circuit has fidelity 1 by construction.

# %%
qubits = range(3, 9)
# Test both linear and log GHZ circuits with both noise models
for n in qubits:
    # Linear GHZ circuit
    linear_circuit = build_linear_ghz(n)

    # Log GHZ circuit
    log_circuit = build_log_ghz(n)

    # Apply noise models
    linear_one_zone_circuit = utils.noise.transform_circuit(
        linear_circuit, model=one_zone_model
    )
    linear_two_zone_circuit = utils.noise.transform_circuit(
        linear_circuit, model=two_zone_model
    )
    log_one_zone_circuit = utils.noise.transform_circuit(
        log_circuit, model=one_zone_model
    )
    log_two_zone_circuit = utils.noise.transform_circuit(
        log_circuit, model=two_zone_model
    )

    # Simulate noiseless circuits
    rho_linear = simulator.simulate(linear_circuit).final_density_matrix
    rho_log = simulator.simulate(log_circuit).final_density_matrix

    # Simulate noisy circuits
    rho_linear_one_zone = simulator.simulate(
        linear_one_zone_circuit
    ).final_density_matrix
    rho_linear_two_zone = simulator.simulate(
        linear_two_zone_circuit
    ).final_density_matrix
    rho_log_one_zone = simulator.simulate(log_one_zone_circuit).final_density_matrix
    rho_log_two_zone = simulator.simulate(log_two_zone_circuit).final_density_matrix

    # Calculate fidelities
    fidelity_linear_one_zone = np.trace(rho_linear @ rho_linear_one_zone).real
    fidelity_linear_two_zone = np.trace(rho_linear @ rho_linear_two_zone).real
    fidelity_log_one_zone = np.trace(rho_log @ rho_log_one_zone).real
    fidelity_log_two_zone = np.trace(rho_log @ rho_log_two_zone).real

    # Store results
    fidelities_linear_one_zone.append(fidelity_linear_one_zone)
    fidelities_linear_two_zone.append(fidelity_linear_two_zone)
    fidelities_log_one_zone.append(fidelity_log_one_zone)
    fidelities_log_two_zone.append(fidelity_log_two_zone)

    print(f"n={n}:")
    print(
        f"  Linear GHZ - One Zone: {fidelity_linear_one_zone:.4f}, Two Zone: {fidelity_linear_two_zone:.4f}"
    )
    print(
        f"  Log GHZ - One Zone: {fidelity_log_one_zone:.4f}, Two Zone: {fidelity_log_two_zone:.4f}"
    )


# %% [markdown]
# Fidelity comparison plot:

# %%
# Create comparison plot
plt.figure(figsize=(12, 8))

plt.plot(
    qubits,
    fidelities_linear_one_zone,
    "ro-",
    label="Linear GHZ - One Zone",
    linewidth=2,
    markersize=8,
)
plt.plot(
    qubits,
    fidelities_linear_two_zone,
    "r^--",
    label="Linear GHZ - Two Zone",
    linewidth=2,
    markersize=8,
)
plt.plot(
    qubits,
    fidelities_log_one_zone,
    "bo-",
    label="Log GHZ - One Zone",
    linewidth=2,
    markersize=8,
)
plt.plot(
    qubits,
    fidelities_log_two_zone,
    "b^--",
    label="Log GHZ - Two Zone",
    linewidth=2,
    markersize=8,
)

plt.xlabel("Number of Qubits", fontsize=14)
plt.ylabel("Fidelity", fontsize=14)
plt.title(
    "GHZ State Fidelity Comparison: Linear vs Log Depth with Different Noise Models",
    fontsize=16,
)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(qubits)
# plt.ylim(0, 1.05)

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
    f"Linear GHZ - One Zone: Mean = {np.mean(fidelities_linear_one_zone):.4f}, Std = {np.std(fidelities_linear_one_zone):.4f}"
)
print(
    f"Linear GHZ - Two Zone: Mean = {np.mean(fidelities_linear_two_zone):.4f}, Std = {np.std(fidelities_linear_two_zone):.4f}"
)
print(
    f"Log GHZ - One Zone: Mean = {np.mean(fidelities_log_one_zone):.4f}, Std = {np.std(fidelities_log_two_zone):.4f}"
)
print(
    f"Log GHZ - Two Zone: Mean = {np.mean(fidelities_log_two_zone):.4f}, Std = {np.std(fidelities_log_two_zone):.4f}"
)


# %% [markdown]
# The GHZ results show that parallelizing gates increases fidelity compared with the sequential implementation. The fidelity improvement is consistent across the one-zone and two-zone architectures and generally grows as we increase the number of qubits.

# %% [markdown]
# ## Automatic toolkits for circuit parallelization
#
# In the GHZ example the log-depth circuit was provided manually. Bloqade also includes automatic tools that compress a deep circuit into a more parallel form by casting the problem as an optimization (integer linear program).
#
# We first transpile the circuit into a standard gate set consisting of single-qubit gates and two-qubit CZ gates. CZ gates acting on disjoint qubits commute and therefore can be placed in the same moment.
#
# The optimization objective encourages gates that can execute in parallel to be assigned to nearby moments. Formally, we minimize an objective such as $um_{g}um_{o_p,o_qn g} w_g|athrm{moment}(o_p)-athrm{moment}(o_q)|$, where each group g collects operations that should be close and the weight $w_g$ controls their attraction. Operations may have multiple tags (groups) when they can be parallelized with different sets of gates.
#
# To preserve program semantics we only reorder operations that commute. Execution dependencies are represented as a directed acyclic graph (DAG); each vertex i receives an integer label $t_i$ (its epoch) and ordering constraints are expressed as inequalities on these labels. These constraints and objective terms are encoded in an integer linear program (ILP).
#
# Because the ILP formulation has totally unimodular constraint structure in our encoding, the relaxed linear program yields integer solutions, which makes the optimization efficient in practice. Absolute-value terms are reformulated into equivalent linear forms during construction.
#
# The helper `bloqade.cirq_utils.auto_similarity` tags operations and assigns weights to build the linear objective. Users can also add manual annotations (tags/weights) to guide the parallelizer when they need fine-grained control.
#
#

# %% [markdown]
# ## Example 2: [7,1,3] Steane code circuit

# %% [markdown]
# We construct a sequential circuit for the [7,1,3] Steane code and then apply Bloqade's parallelization utilities (which use `auto_similarity`) to compress it.


# %% [code]
def build_steane_code_circuit():
    """
    Build the Steane code circuit for error correction.
    """
    # Create qubits for the 7-qubit Steane code
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.Circuit()

    # Define the gate sequence as a list of operations
    operations = [
        # H gate on qubits 1, 2, 3
        cirq.H(qubits[1]),
        cirq.H(qubits[2]),
        cirq.H(qubits[3]),
        # Encode the logical qubit with CZ and H gates (equivalent to CNOT)
        cirq.H(qubits[0]),
        cirq.CZ(qubits[1], qubits[0]),
        cirq.CZ(qubits[2], qubits[0]),
        cirq.H(qubits[4]),
        cirq.CZ(qubits[2], qubits[4]),
        cirq.CZ(qubits[6], qubits[4]),
        cirq.H(qubits[5]),
        cirq.CZ(qubits[6], qubits[5]),
        cirq.CZ(qubits[3], qubits[5]),
        cirq.CZ(qubits[1], qubits[5]),
        cirq.H(qubits[5]),
        cirq.H(qubits[6]),
        cirq.CZ(qubits[1], qubits[6]),
        cirq.CZ(qubits[2], qubits[6]),
        cirq.H(qubits[6]),
        cirq.CZ(qubits[3], qubits[4]),
        cirq.H(qubits[4]),
        cirq.CZ(qubits[3], qubits[0]),
        cirq.H(qubits[0]),
    ]

    # Append all operations to the circuit
    circuit.append(operations)

    return circuit


# %%
# Build Steane circuits (reuse already defined noise models and simulator)
steane_original = build_steane_code_circuit()
steane_parallel = utils.parallelize(circuit=steane_original)
steane_parallel = utils.no_similarity(steane_parallel)
print("Original Steane Circuit:")
print(steane_original)
print("\nParallelized Steane Circuit:")
print(steane_parallel)
print("Original Steane circuit depth:", len(steane_original))
print("Parallelized Steane circuit depth:", len(steane_parallel))


# %% [markdown]
# We perform noise analysis on both the original and parallelized Steane circuits using our one-zone and two-zone noise models.

# %%
# Apply noise models to both circuits
steane_original_one_zone = utils.noise.transform_circuit(
    steane_original, model=one_zone_model
)
steane_original_two_zone = utils.noise.transform_circuit(
    steane_original, model=two_zone_model
)
steane_parallel_one_zone = utils.noise.transform_circuit(
    steane_parallel, model=one_zone_model
)
steane_parallel_two_zone = utils.noise.transform_circuit(
    steane_parallel, model=two_zone_model
)

# Simulate ideal circuits
rho_original_ideal = simulator.simulate(steane_original).final_density_matrix
rho_parallel_ideal = simulator.simulate(steane_parallel).final_density_matrix

# Simulate noisy circuits
rho_original_one_zone = simulator.simulate(
    steane_original_one_zone
).final_density_matrix
rho_original_two_zone = simulator.simulate(
    steane_original_two_zone
).final_density_matrix
rho_parallel_one_zone = simulator.simulate(
    steane_parallel_one_zone
).final_density_matrix
rho_parallel_two_zone = simulator.simulate(
    steane_parallel_two_zone
).final_density_matrix

# Calculate fidelities
fidelity_original_one_zone = np.trace(rho_original_ideal @ rho_original_one_zone).real
fidelity_original_two_zone = np.trace(rho_original_ideal @ rho_original_two_zone).real
fidelity_parallel_one_zone = np.trace(rho_parallel_ideal @ rho_parallel_one_zone).real
fidelity_parallel_two_zone = np.trace(rho_parallel_ideal @ rho_parallel_two_zone).real

# Print results
print("\n=== Steane Code Circuit Fidelity Comparison ===")
print(f"Original circuit - One Zone: {fidelity_original_one_zone:.4f}")
print(f"Original circuit - Two Zone: {fidelity_original_two_zone:.4f}")
print(f"Parallelized circuit - One Zone: {fidelity_parallel_one_zone:.4f}")
print(f"Parallelized circuit - Two Zone: {fidelity_parallel_two_zone:.4f}")

# Calculate improvements
improvement_one_zone = fidelity_parallel_one_zone - fidelity_original_one_zone
improvement_two_zone = fidelity_parallel_two_zone - fidelity_original_two_zone

print("\nFidelity improvement with parallelization:")
print(f"One Zone: {improvement_one_zone:+.4f}")
print(f"Two Zone: {improvement_two_zone:+.4f}")


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
if improvement_one_zone > 0 and improvement_two_zone > 0:
    print("✓ Parallelization improves fidelity under both noise models")
elif improvement_one_zone > 0 or improvement_two_zone > 0:
    print("~ Parallelization shows mixed results across noise models")
else:
    print("✗ Parallelization does not improve fidelity under these noise models")


# %% [markdown]
# ## Example 3: Linear chained circuit
#
# Here is another example of a circuit of CZ gates in a linear chain. The orginal circuit has a linear growing number of moments as qubit number.


# %%
def build_circuit3():
    # Build a simple circuit with CZ gates in a linear chain
    n = 10
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    for i in range(n - 1):
        circuit.append(cirq.CZ(qubits[i], qubits[(i + 1)]))
    return circuit


# Build the linear CZ circuit
circuit3 = build_circuit3()

print("Original CZ chain circuit:")
print(circuit3)

# Parallelize the circuit, `auto_similarity` is automatically applied inside `parallelize`
circuit3_parallel = utils.parallelize(circuit=circuit3)

# Remove any tags and print the parallelized circuit
circuit3_parallel = utils.no_similarity(circuit3_parallel)
print("Parallelized CZ chain circuit:")
print(circuit3_parallel)

print("Original CZ chain circuit depth:", len(circuit3))
print("Parallelized CZ chain circuit depth:", len(circuit3_parallel))
# %% [markdown]
# ## Example 4: 2D CZ circuit
#
# We extend the linear chain example to a 4-by-4 lattice of qubits with periodic boundary conditions along both x and y directions. The circuit contains CZ operations acting only on neighboring qubits in a 2D grid.
#


# %%
def build_circuit4():
    L = 4
    # Create GridQubits and reshape into 2D array for qubits[i,j] indexing
    qubits = np.array(cirq.GridQubit.rect(L, L)).reshape(L, L)

    # Method 1: Use explicit Moments to fix gate positions
    moments = []

    # Add horizontal nearest-neighbor interactions in separate moments
    for row in range(L):
        for col in range(L):
            moments.append(
                cirq.Moment([cirq.CZ(qubits[row, col], qubits[row, (col + 1) % L])])
            )

    # Add vertical nearest-neighbor interactions in separate moments
    for row in range(L):
        for col in range(L):
            moments.append(
                cirq.Moment([cirq.CZ(qubits[row, col], qubits[(row + 1) % L, col])])
            )
    circuit = cirq.Circuit(moments)
    return circuit


circuit4 = build_circuit4()

print("Original 2D grid circuit:")
print(circuit4)
# %% [markdown]
# The helper `visualize_grid_interactions` renders qubit interactions on a 2D grid. It draws connections for two-qubit gates and colors each moment differently to show execution order.


# %%
def visualize_grid_interactions(circuit, L=4):
    """
    Visualize qubit interactions on a grid with different colors for each moment.
    Uses arcs for non-nearest-neighbor interactions to avoid overlapping.

    Args:
        circuit: cirq.Circuit with GridQubit gates
        L: Grid size (L x L)
    """
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 12))

    # Get a colormap for different moments with MAXIMUM contrast
    num_moments = len(circuit)

    # Define highly contrasting colors manually for maximum visibility
    high_contrast_colors = [
        "#FF0000",  # Bright Red
        "#0000FF",  # Bright Blue
        "#00FF00",  # Bright Green
        "#FF00FF",  # Magenta
        "#FFA500",  # Orange
        "#00FFFF",  # Cyan
        "#FFFF00",  # Yellow
        "#FF1493",  # Deep Pink
        "#8B00FF",  # Violet
        "#00FF7F",  # Spring Green
        "#FF4500",  # Orange Red
        "#1E90FF",  # Dodger Blue
        "#FF69B4",  # Hot Pink
        "#32CD32",  # Lime Green
        "#DC143C",  # Crimson
        "#00CED1",  # Dark Turquoise
        "#FFD700",  # Gold
        "#9400D3",  # Dark Violet
        "#FF8C00",  # Dark Orange
        "#00FA9A",  # Medium Spring Green
    ]

    # Use high contrast colors or repeat if we have more moments
    if num_moments <= len(high_contrast_colors):
        colors = [high_contrast_colors[i] for i in range(num_moments)]
    else:
        colors = [
            high_contrast_colors[i % len(high_contrast_colors)]
            for i in range(num_moments)
        ]

    # Function to check if interaction is nearest neighbor
    def is_nearest_neighbor(r1, c1, r2, c2, L):
        # Nearest neighbor: both x and y differences are at most 1 (non-periodic)
        # Exclude same qubit (r1==r2 and c1==c2)
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 and not (r1 == r2 and c1 == c2)

    def is_periodic_boundary(r1, c1, r2, c2, L):
        # Check if this is a periodic boundary connection
        # Horizontal periodic: same row, column distance is L-1
        h_periodic = (r1 == r2) and abs(c1 - c2) == L - 1
        # Vertical periodic: same column, row distance is L-1
        v_periodic = (c1 == c2) and abs(r1 - r2) == L - 1
        return h_periodic or v_periodic

    # Draw interactions from each moment with different colors
    from matplotlib.patches import FancyBboxPatch, ConnectionPatch

    legend_elements = []
    for moment_idx, moment in enumerate(circuit):
        for op in moment:
            if len(op.qubits) == 2:
                q1, q2 = op.qubits
                r1, c1 = q1.row, q1.col
                r2, c2 = q2.row, q2.col

                if is_nearest_neighbor(r1, c1, r2, c2, L):
                    # Draw straight line for nearest neighbors
                    ax.plot(
                        [c1, c2],
                        [r1, r2],
                        "-",
                        color=colors[moment_idx],
                        linewidth=4,
                        alpha=0.9,
                        zorder=1,
                    )
                elif is_periodic_boundary(r1, c1, r2, c2, L):
                    # Draw curved connection for periodic boundary using ConnectionPatch
                    # Use smaller radius values (larger effective radius) for gentler curves

                    # For horizontal periodic boundary (same row)
                    if r1 == r2:
                        # Always curve in the same direction regardless of order
                        # For consistency, always curve upward (toward row 0)
                        # Ensure we draw from left to right (smaller c to larger c)
                        if c1 < c2:
                            # This is the normal case (e.g., (0,row) to (3,row))
                            # Draw from right to left to make the arc go upward
                            start, end = (c2, r2), (c1, r1)
                            connectionstyle = "arc3,rad=-.15"
                        else:
                            # This is when c1 > c2 (e.g., (3,row) to (0,row))
                            start, end = (c1, r1), (c2, r2)
                            connectionstyle = "arc3,rad=-.15"

                        con = ConnectionPatch(
                            start,
                            end,
                            "data",
                            "data",
                            arrowstyle="-",
                            shrinkA=0,
                            shrinkB=0,
                            mutation_scale=20,
                            fc=colors[moment_idx],
                            connectionstyle=connectionstyle,
                            color=colors[moment_idx],
                            linewidth=4,
                            alpha=0.9,
                            zorder=1,
                        )
                        ax.add_artist(con)

                    # For vertical periodic boundary (same column)
                    else:  # c1 == c2
                        # Always curve in the same direction regardless of order
                        # For consistency, always curve leftward (toward column 0)
                        # Ensure we draw from top to bottom (smaller r to larger r)
                        if r1 < r2:
                            # This is the normal case (e.g., (col,0) to (col,3))
                            # Draw from bottom to top to make the arc go leftward
                            start, end = (c2, r2), (c1, r1)
                            connectionstyle = "arc3,rad=-.15"
                        else:
                            # This is when r1 > r2 (e.g., (col,3) to (col,0))
                            start, end = (c1, r1), (c2, r2)
                            connectionstyle = "arc3,rad=-.15"

                        con = ConnectionPatch(
                            start,
                            end,
                            "data",
                            "data",
                            arrowstyle="-",
                            shrinkA=0,
                            shrinkB=0,
                            mutation_scale=20,
                            fc=colors[moment_idx],
                            connectionstyle=connectionstyle,
                            color=colors[moment_idx],
                            linewidth=4,
                            alpha=0.9,
                            zorder=1,
                        )
                        ax.add_artist(con)
                else:
                    # Draw curved connection for other non-nearest neighbors
                    con = ConnectionPatch(
                        (c1, r1),
                        (c2, r2),
                        "data",
                        "data",
                        arrowstyle="-",
                        shrinkA=0,
                        shrinkB=0,
                        mutation_scale=20,
                        fc=colors[moment_idx],
                        connectionstyle="arc3,rad=.3",
                        color=colors[moment_idx],
                        linewidth=4,
                        alpha=0.9,
                        zorder=1,
                    )
                    ax.add_artist(con)

        # Add legend entry for this moment
        legend_elements.append(
            mpatches.Patch(color=colors[moment_idx], label=f"Moment {moment_idx}")
        )

    # Draw grid points (qubits) - extra large size for best visibility
    for row in range(L):
        for col in range(L):
            ax.plot(
                col, row, "ko", markersize=75, zorder=3
            )  # Outer black circle (50% larger)
            ax.plot(
                col, row, "wo", markersize=68, zorder=4
            )  # Inner white circle (50% larger)
            ax.text(
                col,
                row,
                f"{row},{col}",
                ha="center",
                va="center",
                fontsize=20,
                color="black",
                weight="bold",
                zorder=5,
            )  # Proportionally larger font

    # Set up the plot
    ax.set_xlim(-0.7, L - 0.3)
    ax.set_ylim(-0.7, L - 0.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Column", fontsize=14, weight="bold")
    ax.set_ylabel("Row", fontsize=14, weight="bold")
    ax.set_title(
        f"Qubit Interactions on {L}x{L} Grid by Moment\n"
        f"({num_moments} moments, colored by execution order)",
        fontsize=16,
        weight="bold",
    )
    ax.invert_yaxis()  # Make (0,0) appear at top-left

    # Add legend
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=11,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.show()

    print(f"Circuit depth (number of moments): {num_moments}")


# %% [markdown]
# This function can visualize the circuit on 2D grid. All operations within the same moment will be ploted with the same color.

# %%
# Visualize the circuits in 2D grid format
print("Original circuit:")
visualize_grid_interactions(circuit4, L=4)

# %% [markdown]
# Now we parallelize the circuit and print basic statistics (circuit text and moment counts).

# %%
circuit4_parallel = utils.parallelize(circuit=circuit4)
circuit4_parallel = utils.no_similarity(circuit4_parallel)
print(circuit4_parallel)
# Print circuit depths
print("Original 2D CZ circuit depth:", len(circuit4))
print("Parallelized 2D CZ circuit depth:", len(circuit4_parallel))

# %% [markdown]
# The `parallelism` compresses the circuit down to four moments. Note that this solution is degenerate: multiple equivalent moment assignments exist, so the specific packing depends on tie-breaking in the optimizer.

# %% [markdown]
# Here is visualization of the parallel circuit on a 2D grid. Only four colors (moments) are used to color all interactions.

# %%
print("\nParallelized circuit:")
visualize_grid_interactions(circuit4_parallel, L=4)
