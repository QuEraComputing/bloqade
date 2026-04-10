# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: kirin-workspace (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulator Demo for Gemini Logical
# Gemini Logical is a set of dialects and compilation tools in Bloqade for logical-kernel workflows on Gemini-style architectures.
# In this tutorial, we build a simple logical kernel, inspect the generated physical and noisy circuits, and analyze detector and observable outputs using the `GeminiLogicalSimulator`.

# %%
# Builtins
import math
from collections import Counter
import numpy as np

# Types
from typing import Any
from kirin.dialects import ilist
from kirin.ir.method import Method

# Functions and methods
from bloqade.decoders import BpLsdDecoder
from bloqade.lanes import GeminiLogicalSimulator

# Dialect groups
from bloqade.gemini import logical
from bloqade import qubit, squin

# %%

import matplotlib.pyplot as plt


def render_steane_code_qubit(
    ax: plt.Axes | None = None, center: tuple[float, float] = (0, 0)
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim([-2 + center[0], 2 + center[0]])
        ax.set_ylim([-2 + center[1], 2 + center[1]])
        ax.axis("off")
    RED = "#EF2F55"
    PURPLE = "#670EFF"
    GREEN = "#57BC13"

    pos_center = np.zeros([2, 7])
    pos_center[:, 1::] = np.array(
        [
            np.cos(np.linspace(0, 2 * np.pi, 7)[0:6]) * 1.5,
            1.5 * np.sin(np.linspace(0, 2 * np.pi, 7)[0:6]),
        ]
    )
    pos_center += np.array(center).reshape(2, 1)

    ax.scatter(
        pos_center[0], pos_center[1], color="w", s=800, zorder=100, edgecolors="k"
    )
    indexing = [2, 0, 3, 6, 4, 5, 1]
    for xi, yi, label in zip(pos_center[0], pos_center[1], indexing):
        ax.text(xi, yi, str(label), ha="center", va="center", zorder=200)

    ax.fill(
        [pos_center[0, x] for x in [0, 2, 3, 4]],
        [pos_center[1, x] for x in [0, 2, 3, 4]],
        color=RED,
    )
    ax.fill(
        [pos_center[0, x] for x in [0, 4, 5, 6]],
        [pos_center[1, x] for x in [0, 4, 5, 6]],
        color=GREEN,
    )
    ax.fill(
        [pos_center[0, x] for x in [0, 6, 1, 2]],
        [pos_center[1, x] for x in [0, 6, 1, 2]],
        color=PURPLE,
    )
    logical_label = [indexing.index(5), indexing.index(1), indexing.index(0)]
    ax.plot(
        pos_center[0, logical_label],
        pos_center[1, logical_label],
        color="k",
        ls="-",
        linewidth=5,
        zorder=50,
    )

    return ax


# A minimal kernel that prepares a single qubit in an arbitrary state,
# so that it can be shown by the tsim renderer.
@logical.kernel(aggressive_unroll=True, verify=True)
def main():
    reg = qubit.qalloc(1)
    squin.u3(0.1, 0.2, 0.3, reg[0])
    return logical.terminal_measure(reg)


task = GeminiLogicalSimulator().task(main)

# %% [markdown]
# ### Some prototype stdutils functions: detectors and observables
# We break abstraction a bit between physical and logical qubits. Each logical measurement is a batch of 7 physical measurements as indexed by the following diagram.
#
# In order to correct the errors from a Steane code, we need to inform the decoder and detector error model of the decoding steps. This can be done by defining the **detectors** and the **observables**.
#
# For the Steane code, the detectors are four-qubit parity patches corresponding to the three plaquettes of the code; in the following render the default detectors are red/green/purple patches.
#
# For the Steane code, the obsevables are three-qubit parity lines corresponding to edges of the code; in the following render is default observable are the black line.

# %%
render_steane_code_qubit()
task.tsim_circuit.diagram(width=400)


# %% [markdown]
# Lets define some default functions which use the `squin.set_detector` and `squin.set_observable` functions, which annotate the program for later analysis to generate the detector error model.

# %% [markdown]
# For the purposes of our demonstration, lets prepare a simple GHZ state. Note that the decorator is `@logical.kernel` instead of `@squin.kernel`.


# %%
@logical.kernel(aggressive_unroll=True, verify=True)
def main():
    reg = qubit.qalloc(3)
    squin.h(reg[0])
    squin.cx(reg[0], reg[1])

    return logical.default_post_processing(reg)  # Return the physical measurements


task = GeminiLogicalSimulator().task(main)

# %% [markdown]
# The `task` has several attributes. The key attributes are:
# | Attribute         | Description                          |
# |-------------------|--------------------------------------|
# | `task.run`        | Run the task, sampling bitstrings from the noisy distribution |
# | `task.noiseless_tsim_circuit`             | The underlying physical circuit without noise        |
# | `task.tsim_circuit`             | The underlying physical circuit including noise             |
# | `task.detector_error_model`           | The DEM associated with the noisy circuit       |
# | `task.visualize`     | Render an interactive atom move. Does not work in jupyter notebooks =(     |
#
# ## Rendering of the noiseless circuit:

# %%
task.noiseless_tsim_circuit.diagram(height=task.noiseless_tsim_circuit.num_qubits * 25)

# %% [markdown]
# ## Rendering of the noisy circuit:
# Includes 1 and 2 qubit gate error, spectator errors, and move errors. It does not include state preparation errors.

# %%
task.tsim_circuit.diagram(height=task.tsim_circuit.num_qubits * 25)

# %% [markdown]
# ## Running the task
# the `task.run` attribute compiles the task to tsim and then samples from it. Note that the majority of the time is spent compiling the task; the sampler is very fast.

# %%
result = task.run(1, with_noise=True)
result_wo_noise = task.run(1, with_noise=False)

# %%
# After recompilation, the task runs very quickly.
result = task.run(10000, with_noise=True)
result_wo_noise = task.run(10000, with_noise=False)

# %% [markdown]
# The `result` object has several meaningful attributes that are useful for analysis:
# | Attribute         | Description                          |
# |-------------------|--------------------------------------|
# | `result.return_values`        | The values returned from the kernel |
# | `result.detectors` | The parity values of the annotated detectors |
# | `result.observables` | The parity values of the annotated observables |
# | `result.physical` | The physical qubit measurements |
#
# For each value, the zeroth dimension is the shot index.
# - `detectors` are a flattened list of `[ [detectors of qubit 0 ], [detectors of qubit 1] [ ... ] ]`
# - `observables` are a list of `[ obsevable of qubit 0, observable of qubit 1, ... ]`
# - `physical` is a nested list of `[[7 physical measurements of qubit 0], [7 physical measurements of qubit 1], ...]`
#
# Indexing is in the same ordering of however the qubits were measured in the `logical.terminal_measure` statement.

# %%
return_values = result.return_values
detectors = np.asarray(result.detectors)
observables = np.asarray(result.observables)
physical = np.asarray(result.measurements)
observables_without_noise = np.asarray(result_wo_noise.observables)

print(detectors.shape)
print(observables.shape)
print(physical.shape)

# %% [markdown]
# ## Decoding and post-selection
# Decoders can be inherited from elsewhere but follow a common pattern. Given the detector error model, flips to the logical qubits can be decoded based on the detector triggers. Because the code is linear, the corrected code is simply the XOR of the flips and the observables.
#
# Alternatively, one may postselect on having no errors, or all detectors being zero.

# %%
# Correct
flips = BpLsdDecoder(task.detector_error_model).decode(detectors)
observables_corrected = observables ^ flips
print("Average bits flipped:", np.average(flips))

# Postselect
post_selection = np.all(detectors == 0, axis=1)
observables_postselected = observables[post_selection, :]
print("Postselection rate:  ", len(observables_postselected) / len(observables))

# %% [markdown]
# ## Analysis 1: parity
# For the GHZ state, we have the convenience of the final state being uniformly sampled from 00 or 11, with 01 or 10 indicating an error outside of the distribution. Thus, computing the parity of the observables can serve as a proxy of the fidelity of the distribution: parity 0 means no error, parity 1 means error, and the average parity is the error rate. Postselection and correction decreases the parity, meaning the final error is better!

# %%
print(
    "Average parity (before correction):",
    np.average(observables[:, 0] ^ observables[:, 1]),
)
print(
    "Average parity (after correction):",
    np.average(observables_corrected[:, 0] ^ observables_corrected[:, 1]),
)
print(
    "Average parity (after postselection):",
    np.average(observables_postselected[:, 0] ^ observables_postselected[:, 1]),
)


# %% [markdown]
# ### Some helper functions and standard utilities to analyze statistical divergence


# %%
# helper functions to analyze statistical distribution of logical measurements
def get_hist(obs_array: np.ndarray):
    return Counter(map(lambda x: tuple(map(int, x)), obs_array[:]))


def kl_divergence(p_hist: Counter, q_hist: Counter) -> float:
    """Compute the KL divergence D_KL(P || Q) between two histograms."""
    total_p = sum(p_hist.values())
    total_q = sum(q_hist.values())
    if total_p == 0 or total_q == 0:
        return float("inf")  # Infinite divergence if one distribution is empty
    divergence = 0.0
    for key in p_hist:
        p_prob = p_hist[key] / total_p
        q_prob = q_hist.get(key, 0) / total_q
        if q_prob > 0:
            divergence += p_prob * math.log(p_prob / q_prob)
        else:
            divergence += p_prob * math.log(p_prob / (1e-10))  # Avoid log(0)
    return divergence


# %% [markdown]
# The Kullback-Leibler divergence $D_{KL}(P||Q)$ measures the dissimilarity between two probability distributions. When the KL divergence is zero, there is no loss when the noisy distribution (Q) is used to represent the perfect distribution (P). Similar to the parity measurement above, we find that the divergence is lower for corrected and postselected distributions. Note that the distribution is approximated from finite sampling (a simple frequentist bootstrap) so the KL divergence is an upper bound on the true distribution.

# %%
observables_hist = get_hist(observables)
observables_decoded_hist = get_hist(observables_corrected)
observables_postselected_hist = get_hist(observables_postselected)
observables_wo_noise_hist = get_hist(observables_without_noise)

# compute and print the KL divergence between the histograms
print(
    "KL divergence between noiseless and raw observables:",
    kl_divergence(observables_wo_noise_hist, observables_hist),
)
print(
    "KL divergence between noiseless and decoded observables:",
    kl_divergence(observables_wo_noise_hist, observables_decoded_hist),
)
print(
    "KL divergence between noiseless and post-selected observables:",
    kl_divergence(observables_wo_noise_hist, observables_postselected_hist),
)


# %% [markdown]
# ### Dos and do nots for kernels
# A valid kernel for Gemini must:
# 1. Have less than 10 qubits
# 2. Only have a single non-Clifford gate per qubit, acting as a single-qubit gate as the first gate on each qubit
# 3. Measurement is in Z basis only.
#
# ![image.svg](../image.svg)

# %% [markdown]
# Too many qubits

# %%
try:

    @logical.kernel(aggressive_unroll=True, verify=True)
    def main():
        reg = qubit.qalloc(12)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])

        return logical.default_post_processing(reg)

    task = GeminiLogicalSimulator().task(main)
except BaseException as e:
    print("Error during kernel definition or task creation:", e)

# %% [markdown]
# Repeated non-Clifford rotations

# %%
try:

    @logical.kernel(aggressive_unroll=True, verify=True)
    def main():
        reg = qubit.qalloc(12)
        squin.t(reg[0])
        squin.t(reg[0])
        squin.cx(reg[0], reg[1])

        return logical.default_post_processing(reg)

    task = GeminiLogicalSimulator().task(main)
except BaseException as e:
    print("Error during kernel definition or task creation:", e)

# %% [markdown]
# Non-Clifford rotation not as the first gate (This is the same validation error)

# %%
try:

    @logical.kernel(aggressive_unroll=True, verify=True)
    def main():
        reg = qubit.qalloc(12)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        squin.t(reg[0])

        return logical.default_post_processing(reg)

    task = GeminiLogicalSimulator().task(main)
except BaseException as e:
    print("Error during kernel definition or task creation:", e)


# %% [markdown]
# ## Explicitly annotate parallelism
# If parallelism is not annotated, the program will implement each two qubit gate sequentially. We currently do not have any auto-parallelization passes.


# %%
def terminal_main_wrapper(kernel: Method[[], ilist.IList[qubit.Qubit, Any]]) -> Method:
    """
    A helper function that wraps a kernel that returns a qubit register that has
    had some computation performed on it and transforms it into a logical kernel.
    """

    @logical.kernel(aggressive_unroll=True, verify=True)
    def terminal_main():
        reg = kernel()

        logical.default_post_processing(reg)

    return terminal_main


# %%
@squin.kernel
def unparallelized_main() -> ilist.IList[qubit.Qubit, Any]:
    """
    A kernel that annotates no parallelism, even though they exist
    """
    reg = qubit.qalloc(4)
    squin.cx(reg[0], reg[1])
    squin.cx(reg[2], reg[3])
    return reg


@squin.kernel
def parallelized_main() -> ilist.IList[qubit.Qubit, Any]:
    """
    An equivalent kernel to the above, but with parallelism annotated via broadcast operations.
    """
    reg = qubit.qalloc(4)
    squin.broadcast.cx([reg[0], reg[2]], [reg[1], reg[3]])
    return reg


@squin.kernel
def conflicted_parallelized_main() -> ilist.IList[qubit.Qubit, Any]:
    """
    A kernel where parallelism is annotated, but the moves cannot be done all at once due to AOD constraints.
    """
    reg = qubit.qalloc(4)
    squin.broadcast.cx([reg[0], reg[1]], [reg[3], reg[2]])
    return reg


unparallelized_main = terminal_main_wrapper(
    unparallelized_main
)  # hashtag METAPROGRAMMING
parallelized_main = terminal_main_wrapper(parallelized_main)
conflicted_parallelized_main = terminal_main_wrapper(conflicted_parallelized_main)

task_unparallelized = GeminiLogicalSimulator().task(unparallelized_main)
task_parallelized = GeminiLogicalSimulator().task(parallelized_main)
task_conflicted_parallelized = GeminiLogicalSimulator().task(
    conflicted_parallelized_main
)

# %% [markdown]
# The unparallelized circuit sequentially does the two gates with two sets of moves

# %%
_, f = task_unparallelized.fidelity_bounds()
print("Fidelity bounds for unparallelized main:", f)
task_unparallelized.tsim_circuit.diagram(
    height=task_unparallelized.tsim_circuit.num_qubits * 10
)

# %% [markdown]
# The parallelized circuit does a parallel move and implements both gates at the same time.

# %%
_, f = task_parallelized.fidelity_bounds()
print("Fidelity bounds for parallelized main:", f)
task_parallelized.tsim_circuit.diagram(
    height=task_parallelized.tsim_circuit.num_qubits * 10
)

# %% [markdown]
# The parallelized but conflicted circuit implements two sequential moves and then does both gates at the same time.

# %%
_, f = task_conflicted_parallelized.fidelity_bounds()
print("Fidelity bounds for conflicted parallelized main:", f)
task_conflicted_parallelized.tsim_circuit.diagram(
    height=task_conflicted_parallelized.tsim_circuit.num_qubits * 10
)
