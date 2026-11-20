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

# %% [markdown]
# # Running A Simple Postselection Experiment

# %% [markdown]
# ## Without Postselection Wizard

# %%
# from bloqade.gemini.logical import algorithms

# from bloqade.gemini.logical.device import GeminiLogicalDeviceEmulator
# from bloqade.gemini.logical.device import GeminiLogicalDevice

from bloqade.gemini.logical.simulation import GeminiLogicalSimulator

# TODO in the future, for physical:
# from bloqade.gemini.physical.device import GeminiPhysicalDeviceEmulator
# from bloqade.gemini.physical.device import GeminiPhysicalDevice

# from bloqade.gemini.physical.simulation import GeminiPhysicalSimulator

from bloqade.gemini.logical.kernels import (
    build_msd_kernels,
    build_tomography_kernels,
    compose_tomography_kernels,
)
from bloqade.gemini.logical.utils import MSD_TARGET_BLOCH, bloch_to_u3
from bloqade.analysis.tomography import TomographyResult

# from numpy.typing import NDArray
# import math
import numpy as np

from bloqade.decoders import GurobiDecoder

# nonclifford
# clifford
# tomography
# postselection
# decoding

# NOTE: decided NOT to do this:
# from bloqade.gemini_logical import ..
# from bloqade.gemini_physical import ..

# vs

# from bloqade.gemini.logical import ..
# from bloqade.gemini.physical import ..

decoder_cls = GurobiDecoder

# %%
# NOTE: this should be moved to a stdlib in bloqade
# def bloch_to_u3(bloch: NDArray) -> tuple[float, float, float]:
#     x, y, z = bloch[0], bloch[1], bloch[2]
#     theta = math.atan2(math.hypot(x, y), z)
#     phi = math.atan2(y, x)
#     lam = 0.0
#     return (theta, phi, lam)

# %%
# TODO: linter things
NUM_LOGICAL_QUBITS = 5
NUM_SHOTS = 100_000

theta, phi, lam = bloch_to_u3(MSD_TARGET_BLOCH)
primitive_kernels = build_msd_kernels(theta, phi, lam)
tomography_kernels = build_tomography_kernels(output_qubit=0)
experiment_tomography_kernels = compose_tomography_kernels(
    primitive_kernels, NUM_LOGICAL_QUBITS, tomography_kernels
)
simulator = GeminiLogicalSimulator()
compiled_circuits = {
    basis: simulator.task(kernel)
    for basis, kernel in experiment_tomography_kernels.items()
}
simulated_shots = {
    basis: task.run(NUM_SHOTS) for basis, task in compiled_circuits.items()
}

# %%
# DEM's are not quite right here; technically we'll ignore errors that only affect logical observables w/ completely random logical observables. But
# it should be good enough as an approximation
compiled_dems = {
    basis: task.detector_error_model for basis, task in compiled_circuits.items()
}
# Is not quite correct because you are jointly decoding here over ancilla qubits and output observables
compiled_decoders = {basis: decoder_cls(dem) for basis, dem in compiled_dems.items()}
obs_corrections = {
    basis: compiled_decoders[basis].decode(np.array(simulated_shots[basis].detectors))
    for basis in simulated_shots
}

# %%
decoded_obs = {
    basis: obs_corrections[basis] ^ np.array(simulated_shots[basis].observables)
    for basis in obs_corrections
}

# %%
tomography_res = TomographyResult(decoded_obs)

# %% [markdown]
# ## With Postselection Wizard

# %%
import numpy as np  # noqa: E402

from bloqade.gemini.logical.simulation import GeminiLogicalSimulator  # noqa: E402
from bloqade.gemini.logical.kernels import (  # noqa: E402
    build_msd_kernels,
    build_tomography_kernels,
)
from bloqade.gemini.logical.experiments import PostSelectionExperiment  # noqa: E402
from bloqade.gemini.logical.utils import (  # noqa: E402
    plot_decoder_curves,  # noqa: E402
    msd_postselection_obs,  # noqa: E402
    MSD_TARGET_BLOCH,  # noqa: E402
    bloch_to_u3,
)
from bloqade.simulation import TsimSimulatorBackend  # noqa: E402

from bloqade.decoders import TableDecoder  # , MILPDecoder # noqa: E402

NUM_LOGICAL_QUBITS = 5
NUM_SHOTS = 100_000

# %%
theta, phi, lam = bloch_to_u3(MSD_TARGET_BLOCH)
primitive_kernels = build_msd_kernels(theta, phi, lam)
nonclifford_prefix, clifford_circuit = (
    primitive_kernels.state_injection_circuit,
    primitive_kernels.logical_circuit,
)

# %%
tomography_kernels = build_tomography_kernels(output_qubit=0)

# %%
# GeminiLogicalDevice
tsim_simulator = GeminiLogicalSimulator(backend=TsimSimulatorBackend(seed=10))

# %%
msd_mld_exp = PostSelectionExperiment(
    nonclifford_prefix,
    clifford_circuit,
    tomography_kernels,
    TableDecoder,
    decoder_init_args={"seed": 10, "num_shots": NUM_SHOTS},
    device=tsim_simulator,
)

# %%
msd_mld_exp.kernels(num_logical_qubits=NUM_LOGICAL_QUBITS)

# %%
msd_mld_exp.dem_circuits()
msd_mld_exp.dems()
msd_mld_exp.initialize_decoders()

# %%
# TODO: put device as an argument to the experiment
# TODO: can have strategy here for interleaving shots
msd_mld_exp.make_tasks()
msd_mld_exp.get_samples(num_shots=NUM_SHOTS)

# %%
msd_mld_exp.decode_and_postselect(msd_postselection_obs)

# %%
tomo_result = msd_mld_exp.tomography_result(
    accepted_fraction=0.50,
)
tomo_result.fidelity_bloch(MSD_TARGET_BLOCH)

# %%
msd_mld_curve = msd_mld_exp.analysis_f_vs_fraction(
    target_bloch=MSD_TARGET_BLOCH,
)

# %%
fig_mld, ax_mld = msd_mld_exp.analysis_visualization(
    min_accepted_fraction=0.04,
    title="Distilled MSD with MLD postselection",
)

# %%
curves = {"Distilled (MLD)": msd_mld_curve}

# %%
fig, ax = plot_decoder_curves(
    curves,
    min_accepted_fraction=0.04,
    title="PostselectionExperiment MSD workflow",
)
ax.set_xscale("linear")
ax.set_xlabel("Total accepted fraction")
ax.set_ylabel("Magic state fidelity")
ax.legend()

# %%
