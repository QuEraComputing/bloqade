# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bloqade (3.13.12)
#     language: python
#     name: python3
# ---

# %%
from bloqade.gemini.physical.kernels import (
    build_iqp_circuit,
    get_iqp_layout,
    compose_layout_circuit,
)
from bloqade.gemini.physical import GeminiPhysicalDeviceEmulator
from bloqade.core.device import TaskBuilder

# %%
# Use 4 for a quick local emulator run.
n_qubits = 64
iqp_circ = build_iqp_circuit(n_qubits=n_qubits, seed=42)

# %%
iqp_circ_layout = get_iqp_layout(n_qubits=n_qubits, layout_type="hypercube")

# %%
iqp_circ_with_layout = compose_layout_circuit(iqp_circ_layout, iqp_circ)

# %%
task = TaskBuilder()

# %%
task.add_subtask(iqp_circ_with_layout, num_shots=100)

# %%
device = GeminiPhysicalDeviceEmulator()

# %%
future = device.run_async(task)

# %%
result = future.result()

# %%
measurements = result.measurements

# %%
