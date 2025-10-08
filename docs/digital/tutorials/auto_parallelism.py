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
# This tutorial is an overview of how to use Bloqade's parallelism tools to optimize your circuit for parallel execution, as well as give you an intuition for what makes a "good" program via analysis with our noise models.

# %% [markdown]
# The parallel circuits will execute several gates simultaneously. This capability will:
# 1. Greatly simplify the circuits and reduce the number of depths (moments in Cirq).
# 2. Correspondingly, the parallel circuits in general can reach higher fidelities.

# %% [markdown]
# ## Example 1: GHZ Circuit
# Let's kick things off with a simple example: a linear and log depth GHZ state preparation circuit.

# %%
import bloqade.cirq_utils as utils
import cirq
import matplotlib.pyplot as plt
import numpy as np

# %%
def build_linear_ghz(n_qubits:int)->cirq.Circuit:
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(n_qubits-1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    return circuit

def build_log_ghz(n_qubits:int)->cirq.Circuit:
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    width = 1
    while width < n_qubits:
        for i in range(0,width,1):
            if i+width > n_qubits-1:
                break
            circuit.append(cirq.CNOT(qubits[i], qubits[i+width]))
        width *= 2
    return circuit
print(build_log_ghz(7))


# %% [markdown]
# Create the noise model for one zone and two zone architecture. 

# %%
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
# We run the noise model simulations from 3 qubits to 9 qubits: 

# %%
qubits = range(3, 9)
# Test both linear and log GHZ circuits with both noise models
for n in qubits:
    # Linear GHZ circuit
    linear_circuit = build_linear_ghz(n)
    
    # Log GHZ circuit
    log_circuit = build_log_ghz(n)

    # Apply noise models
    linear_one_zone_circuit = utils.noise.transform_circuit(linear_circuit, model=one_zone_model)
    linear_two_zone_circuit = utils.noise.transform_circuit(linear_circuit, model=two_zone_model)
    log_one_zone_circuit = utils.noise.transform_circuit(log_circuit, model=one_zone_model)
    log_two_zone_circuit = utils.noise.transform_circuit(log_circuit, model=two_zone_model)
    
    # Simulate ideal circuits
    rho_linear = simulator.simulate(linear_circuit).final_density_matrix
    rho_log = simulator.simulate(log_circuit).final_density_matrix
    
    # Simulate noisy circuits
    rho_linear_one_zone = simulator.simulate(linear_one_zone_circuit).final_density_matrix
    rho_linear_two_zone = simulator.simulate(linear_two_zone_circuit).final_density_matrix
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
    print(f"  Linear GHZ - One Zone: {fidelity_linear_one_zone:.4f}, Two Zone: {fidelity_linear_two_zone:.4f}")
    print(f"  Log GHZ - One Zone: {fidelity_log_one_zone:.4f}, Two Zone: {fidelity_log_two_zone:.4f}")


# %% [markdown]
# The fidelity plot: 

# %%
# Create comparison plot
plt.figure(figsize=(12, 8))

plt.plot(qubits, fidelities_linear_one_zone, 'ro-', label='Linear GHZ - One Zone', linewidth=2, markersize=8)
plt.plot(qubits, fidelities_linear_two_zone, 'r^--', label='Linear GHZ - Two Zone', linewidth=2, markersize=8)
plt.plot(qubits, fidelities_log_one_zone, 'bo-', label='Log GHZ - One Zone', linewidth=2, markersize=8)
plt.plot(qubits, fidelities_log_two_zone, 'b^--', label='Log GHZ - Two Zone', linewidth=2, markersize=8)

plt.xlabel('Number of Qubits', fontsize=14)
plt.ylabel('Fidelity', fontsize=14)
plt.title('GHZ State Fidelity Comparison: Linear vs Log Depth with Different Noise Models', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(qubits)
#plt.ylim(0, 1.05)

# Add annotations for better understanding
plt.text(0.02, 0.98, 'Higher fidelity = Better performance', 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Linear GHZ - One Zone: Mean = {np.mean(fidelities_linear_one_zone):.4f}, Std = {np.std(fidelities_linear_one_zone):.4f}")
print(f"Linear GHZ - Two Zone: Mean = {np.mean(fidelities_linear_two_zone):.4f}, Std = {np.std(fidelities_linear_two_zone):.4f}")
print(f"Log GHZ - One Zone: Mean = {np.mean(fidelities_log_one_zone):.4f}, Std = {np.std(fidelities_log_two_zone):.4f}")
print(f"Log GHZ - Two Zone: Mean = {np.mean(fidelities_log_two_zone):.4f}, Std = {np.std(fidelities_log_two_zone):.4f}")

# %% [markdown]
# From the GHZ example results, we can clearly see the advantages of parallel gates in terms of fidelity for different zone architectures. And the advantages grow as we increase the number of qubits. 

# %% [markdown]
# ## Automatic Toolkits to Parallelize Circuits
#
# In the previous GHZ example, the log GHZ circuit is given by the user.
# Bloqade provides parallel toolkits to automatically simplify a deep circuit into a parallel circuit. This is implemented by optimizing the depth of the circuit while allowing shuffling of CZ gates.
#
# 1. Rewrite the circuits into 1-qubit gates and 2-qubit CZ gates.
# 2. The circuit can be considered as a Directed Acyclic Graph (DAG), which clearly keeps track of the execution dependence, i.e., one specific gate should come before another.
# 3. Internally, DAGs can be written as a series of inequalities, with each vertex $i$ given an integer label $t_i$ of its epoch. These would be the constraints in the integer programming.
# 4. The constraints of the ILP are totally unimodular, so the solutions are always integers and thus efficient.
# 5. Minimizing the total depth of the circuit is the same as minimizing the total time via a linear objective $\sum_i \lambda_i t_i$
#
# The 2-qubit CZ gates commute with each other.
#
# In order to manually have some control of the process, Bloqade provides annotation similarity by tags, to be ingested by the parallelism functions.
#
# The gates with the same tag would tend to come close with a constant force. This is implemented by adding $weight \times |i-j|$ to the ILP cost function, where $i$ and $j$ are the epoch ids.
#
# `bloqade.cirq_utils` enables the user to add tags to different blocks or moments. 

# %%
utils.auto_similarity   # Similarity if two gates are commuting and identical
utils.block_similarity  # Give all gates in the circuit the same similarity tag (useful for concatenating circuits together)
utils.moment_similarity # Give all gates in the same moment the same simiarlity tag
utils.no_similarity     # Remove all tags
#utils.parallelize(circuit=circuit, auto_tag = False) # auto_tag = False if you want to manually tag similarity


# %% [markdown]
# ## Example 2: [7,1,3] Steane Code Circuit

# %% [markdown]
# We will build a sequential circuit for [7,1,3] Steane code, then use the `auto_similarity` tool to simplify the circuit.  

# %% [code]
def build_steane_code_circuit():
    """
    Build the Steane code circuit for error correction.
    """
    # Create qubits for the 7-qubit Steane code
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.Circuit()

    # H gate on qubits 1, 2, 3
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.H(qubits[2]))
    circuit.append(cirq.H(qubits[3]))
    
    # Encode the logical qubit with CZ and H gates (equivalent to CNOT)

    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CZ(qubits[1], qubits[0]))

    circuit.append(cirq.CZ(qubits[2], qubits[0]))
    
    circuit.append(cirq.H(qubits[4]))
    circuit.append(cirq.CZ(qubits[2], qubits[4]))

    circuit.append(cirq.CZ(qubits[6], qubits[4]))
    
    circuit.append(cirq.H(qubits[5]))
    circuit.append(cirq.CZ(qubits[6], qubits[5]))

    circuit.append(cirq.CZ(qubits[3], qubits[5]))

    circuit.append(cirq.CZ(qubits[1], qubits[5]))
    circuit.append(cirq.H(qubits[5]))
    
    circuit.append(cirq.H(qubits[6]))
    circuit.append(cirq.CZ(qubits[1], qubits[6]))

    circuit.append(cirq.CZ(qubits[2], qubits[6]))
    circuit.append(cirq.H(qubits[6]))
    
    circuit.append(cirq.CZ(qubits[3], qubits[4]))
    circuit.append(cirq.H(qubits[4]))
    
    circuit.append(cirq.CZ(qubits[3], qubits[0]))
    circuit.append(cirq.H(qubits[0]))

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
# Noise analysis for two versions of Steane code for both one-zone and two-zone architecture. 

# %%
# Apply noise models to both circuits
steane_original_one_zone = utils.noise.transform_circuit(steane_original, model=one_zone_model)
steane_original_two_zone = utils.noise.transform_circuit(steane_original, model=two_zone_model)
steane_parallel_one_zone = utils.noise.transform_circuit(steane_parallel, model=one_zone_model)
steane_parallel_two_zone = utils.noise.transform_circuit(steane_parallel, model=two_zone_model)

# Simulate ideal circuits
rho_original_ideal = simulator.simulate(steane_original).final_density_matrix
rho_parallel_ideal = simulator.simulate(steane_parallel).final_density_matrix

# Simulate noisy circuits
rho_original_one_zone = simulator.simulate(steane_original_one_zone).final_density_matrix
rho_original_two_zone = simulator.simulate(steane_original_two_zone).final_density_matrix
rho_parallel_one_zone = simulator.simulate(steane_parallel_one_zone).final_density_matrix
rho_parallel_two_zone = simulator.simulate(steane_parallel_two_zone).final_density_matrix

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

print(f"\nFidelity improvement with parallelization:")
print(f"One Zone: {improvement_one_zone:+.4f}")
print(f"Two Zone: {improvement_two_zone:+.4f}")


# Summary analysis
depth_reduction = len(steane_original) - len(steane_parallel)
depth_reduction_pct = (depth_reduction/len(steane_original)*100) if len(steane_original) > 0 else 0

print(f"\n=== Analysis Summary ===")
print(f"Circuit depth reduction: {len(steane_original)} → {len(steane_parallel)} moments")
print(f"Depth reduction: {depth_reduction} moments ({depth_reduction_pct:.1f}%)")
if improvement_one_zone > 0 and improvement_two_zone > 0:
    print("✓ Parallelization improves fidelity under both noise models")
elif improvement_one_zone > 0 or improvement_two_zone > 0:
    print("~ Parallelization shows mixed results across noise models")
else:
    print("✗ Parallelization does not improve fidelity under these noise models")


# %% [markdown]
# ### Example 2 
#

# %%
def build_circuit1():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.Circuit()
    for i in range(10):
        circuit.append(cirq.CZ(qubits[i], qubits[(i+1)%2 ]))
    return circuit

circuit1 = build_circuit1()

print(circuit1)


# %%
def build_circuit2():
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.Circuit()
    for i in range(5):
        circuit.append(cirq.CZ(qubits[2*i], qubits[(2*i+1)%2] ))
    for i in range(5):
        circuit.append(cirq.CZ(qubits[2*i+1], qubits[(2*i+2)%2] ))
    return circuit

circuit2 = build_circuit2()
print(circuit2)

# %%
print(circuit1 )

# %%
def add_tag_to_operation(op: cirq.Operation, target_ops: set, tag: str):
    """
    Add a tag to operations matching any operation in target_ops.

    Args:
        op: cirq.Operation - operation to check
        target_ops: set of cirq.Operation - reference operations to match
        tag: str - tag to add
    """
    for op2 in target_ops:
        if isinstance(op.gate, type(op2.gate)) and set(op.qubits) == set(op2.qubits):
            return op.with_tags(tag)
    return op

# Example: Add tag to CZ(3,5) and CZ(1,6) in steane_original
target_ops = {
    cirq.CZ(cirq.LineQubit(3), cirq.LineQubit(5)),
    cirq.CZ(cirq.LineQubit(2), cirq.LineQubit(6))
}
steane_original_tagged = steane_original.map_operations(
    lambda op: add_tag_to_operation(op, target_ops, 'abc')
)
print("Tagged circuit:")
print(steane_original_tagged)
transpiled_circuit = utils.transpile(steane_original_tagged)
print("Transpiled circuit:")
print(transpiled_circuit)
transpiled_circuit, group_weights = utils.auto_similarity(
    transpiled_circuit,
    weight_1q=1.0,
    weight_2q=2.0,
)
print("Transpiled circuit before parallelization:")
print(transpiled_circuit)
group_weights.update({'abc': 1000.0})  # Increase the weight for the 'abc' tag
steane_parallel_tagged = utils.parallelize(circuit=transpiled_circuit, auto_tag=False, group_weights=group_weights)

steane_parallel_tagged = utils.no_similarity(steane_parallel_tagged)
print("Parallelized tagged circuit:")
print(steane_parallel_tagged)


