from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast
from collections.abc import Mapping, Callable

import stim
import numpy as np
from kirin import ir
from bloqade.decoders import BaseDecoder

from bloqade.gemini.device import (
    SimulatorResult,
    TsimSimulatorBackend,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
)
from bloqade.gemini.decoding.dem import _sub_detector_error_model
from bloqade.gemini.decoding.msd import (
    _build_msd_primitives,
    _build_decoder_kernel_bundle,
)
from bloqade.gemini.decoding.layout import _DEFAULT_SYNDROME_LAYOUT
from bloqade.gemini.decoding.kernels import (
    _DecoderPrimitiveSet,
    _LogicalTomographyReturn,
    _build_tomography_primitives,
)
from bloqade.gemini.decoding.sampling import _BasisDataset
from bloqade.gemini.decoding.workflow import _plot_decoder_curves
from bloqade.gemini.decoding.confidence import ConfidenceDecoder
from bloqade.gemini.decoding.tomography import _DEFAULT_TARGET_BLOCH, TomographyResult
from bloqade.gemini.decoding.postselection import (
    PostselectionCurveData,
    _DecodedPostselectionResult,
    _shots_at_accepted_fraction,
    _build_generic_threshold_tables,
    _evaluate_cached_threshold_curve,
)
from bloqade.gemini.decoding.special_tasks import _apply_special_tsim_circuit_strategy

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]


def magic_state_dist_steane(
    *,
    theta_offset: float = 0.0,
    phi_offset: float = 0.0,
    lam_offset: float = 0.0,
) -> tuple[ir.Method[..., None], ir.Method[..., None]]:
    """
    Returns the nonclifford prefix circuit and clifford logical circuit
    used for magic state distillation. Note that the kernels take in the qubit register 'reg' as input.
    This means that the kernels are technically agnostic to the number of input qubits.

    Args:
        theta_offset (float): An optional offset to theta to manually introduce rotation error in the nonclifford prefix. Defaults to 0.0.
        phi_offset (float): An optional offset to phi to manually introduce rotation error in the nonclifford prefix. Defaults to 0.0.
        lan_offset (float): An optional offset to lam to manually introduce rotation error in the nonclifford prefix. Defaults to 0.0.

    Returns:
        tuple[ir.Method[..., None], ir.Method[..., None]]: The (nonclifford_prefix, logical_circuit) pair.
    """
    ideal_theta = math.acos(1 / math.sqrt(3))
    ideal_phi = 0.25 * math.pi
    ideal_lam = 0.0

    theta = ideal_theta + theta_offset
    phi = ideal_phi + phi_offset
    lam = ideal_lam + lam_offset

    primitive_set = _build_msd_primitives(theta, phi, lam)

    return primitive_set.state_injection_circuit, primitive_set.logical_circuit


def single_qubit_state_tomography() -> dict[str, ir.Method[..., None]]:
    """
    Returns X, Y, and Z basis kernels used to perform single-qubit tomography. Note that the kernels take in the register as the input.

    Returns:
        dict[str, ir.Method[..., None]]: A dictionary mapping "X", "Y", and "Z" to the respective tomography kernels.
    """
    # return a list?
    # should return (X, Y, Z) in order, but can check this.
    return _build_tomography_primitives(output_qubit=0)


def empty_logical_circuit() -> ir.Method[..., None]:
    """Return a no-op logical Squin kernel for injected-state tomography."""

    from bloqade import squin

    @squin.kernel
    def empty_logical(reg):
        return

    return empty_logical


# TODO: make this "cache" class abstract as well?
class _PostSelectionExperimentCache:
    dem_kernels: dict[str, ir.Method[..., _LogicalTomographyReturn]] | None
    dem_circuits: Mapping[str, tsim_backend.Circuit] | None
    dems: Mapping[str, stim.DetectorErrorModel] | None
    decoders_with_confidence: Mapping[str, tuple[ConfidenceDecoder, BaseDecoder]] | None
    raw_results: Mapping[str, _BasisDataset] | None
    decoded_results: Mapping[str, _DecodedPostselectionResult] | None
    thresholded_data: PostselectionCurveData | None
    hardware_tasks: (
        Mapping[str, GeminiLogicalSimulatorTask[_LogicalTomographyReturn]] | None
    )

    def __init__(self):
        self.dem_kernels = None
        self.dem_circuits = None
        self.dems = None
        self.decoders_with_confidence = None
        self.raw_results = None
        self.decoded_results = None
        self.thresholded_data = None
        self.hardware_tasks = None


def _basis_dataset_from_task_result(
    result: _BasisDataset | SimulatorResult[_LogicalTomographyReturn],
) -> _BasisDataset:
    if isinstance(result, _BasisDataset):
        return result
    return _BasisDataset(
        detectors=np.asarray(result.detectors, dtype=np.uint8),
        observables=np.asarray(result.observables, dtype=np.uint8),
    )


# TODO: should PostSelectionExperiment inherit from some "abstract" experiment workflow class?
# ^ What methods should this "abstract" experiment workflow class have?
class PostSelectionExperiment:
    """
    A "wizard" class that orchestrates the steps of running a experiment to do tomography which obtains samples from the hardware
    and runs decoding with postselection on your ancilla qubits.

    It defines methods for creating noisy tomography circuits, creating decoders from the detector error models from those tomography circuits,
    sampling from the hardware, and doing some analysis based on the confidence associated with each shot.

    Attributes:
        nonclifford_prefix (ir.Method[..., None]): A SQuIN kernel consisting of a single layer of single-qubit gates applied to the physical qubits before the state-preparation circuit is applied.
        clifford_circuit (ir.Method[..., None]): A SQuIN kernel consisting of a Clifford circuit applied to the logical qubits (after logical encoding).
        tomography_circuits (ir.Method[..., None]): A mapping of basis strings to SQuIN kernels consisting of a clifford circuit applied to your logical qubits.
        These kernels will be appended to your circuits in the following fashion, for each `basis_label in tomography_circuits`:
            `nonclifford_prefix + clifford_circuit + tomography_circuits[basis_label]`
            From these circuits in each basis, a DEM will be extracted to initialize decoders in each basis.
        decoder (type[ConfidenceDecoder]): A type of ConfidenceDecoder used to initialize decoders.
        decoder_init_args (Mapping[str, object] | None): Optional arguments that can be passed in to initialize the decoder. Defaults to None.
    """

    def __init__(
        self,
        nonclifford_prefix: ir.Method[..., None],
        clifford_circuit: ir.Method[..., None],
        tomography_circuits: Mapping[str, ir.Method[..., None]],
        decoder: type[ConfidenceDecoder],
        decoder_init_args: Mapping[str, object] | None = None,
    ):
        self.nonclifford_prefix = nonclifford_prefix
        self.clifford_circuit = clifford_circuit
        self.tomography_circuits = dict(tomography_circuits)
        self.decoder = decoder
        self.decoder_init_args = (
            {} if decoder_init_args is None else dict(decoder_init_args)
        )

        self._postselection_exp_cache = _PostSelectionExperimentCache()
        # NOTE: hardcoding this for now (I guess) to support having some
        # interface for adding noise to the circuit and compiling it down.
        # This simulator object is used for compiling down and adding noise to a circuit
        # to construct a DEM for initializing the decoder.
        self._simulator = GeminiLogicalSimulator()

    # TODO: We have to specify the number of logical qubits and number of output qubits here because, given our kernels that take reg as an input, we
    # don't know how many logical qubits that kernel has. -- we can try to implement a compiler pass to infer both the number of qubits and the
    # output qubit from a kernel, however.
    def kernels(
        self,
        num_logical_qubits: int = 5,
    ) -> dict[str, ir.Method[..., _LogicalTomographyReturn]]:
        """
        Composes the nonclifford, clifford, and tomography kernels into a dictionary mapping each basis label to
        the respective nonclifford + clifford + tomography kernel.

        Args:
            num_logical_qubits (int): An integer corresponding to the number of logical qubits allocated in the kernels.

        Returns:
            dict[str, ir.Method[..., _LogicalTomographyReturn]]: A dictionary mapping the tomography basis labels to the respective kernel used for tomography.
        """
        # TODO: change the name of _DecoderPrimitiveSet --> whole_circuit
        decoder_primitive_set = _DecoderPrimitiveSet(
            state_injection_circuit=self.nonclifford_prefix,
            logical_circuit=self.clifford_circuit,
        )
        dem_kernels = _build_decoder_kernel_bundle(
            decoder_primitive_set,
            num_logical_qubits,
            tomography_kernels=self.tomography_circuits,
        )
        self._postselection_exp_cache.dem_kernels = dem_kernels
        return dem_kernels

    def dem_circuits(self) -> dict[str, tsim_backend.Circuit]:
        """
        Constructs TSIM circuits annotated with error channels for each basis from the tomography kernels, which will later
        be used to construct detector error models.

        Note: this function depends on .kernels() being invoked.
        Note: in the implementation of this function, to obtain a noiseless reference observable to later construct a DEM,
        we take our whole circuit, strip away the noise, and prepend it to our circuit so that the noiseless behavior is U^dagU = I.

        Returns:
            dict[str, tsim_backend.Circuit]: A dictionary mapping each basis label to a compiled TSIM circuit with noise channels, where the observables
            in noiseless simulation are all 0.
        """
        dem_kernels = self._postselection_exp_cache.dem_kernels
        if dem_kernels is None:
            raise RuntimeError("kernels must be called before dem_circuits.")
        dem_simulator = GeminiLogicalSimulator(
            noise_model=self._simulator.noise_model,
            backend=TsimSimulatorBackend(),
        )
        dem_tasks = {
            basis: dem_simulator.task(kernel.similar())
            for basis, kernel in dem_kernels.items()
        }
        dem_tasks = _apply_special_tsim_circuit_strategy(dem_tasks)
        special_tsim_circuits: dict[str, tsim_backend.Circuit] = {
            basis: task.tsim_circuit for basis, task in dem_tasks.items()
        }
        self._postselection_exp_cache.dem_circuits = special_tsim_circuits
        return special_tsim_circuits

    def dems(self) -> dict[str, stim.DetectorErrorModel]:
        """
        Constructs detector error models from the noisy circuits in each basis. Defaults to approximating disjoint errors.

        Note: this function depends on dem_circuits() being called.

        Returns:
            dict[str, stim.DetectorErrorModel]: A dictionary mapping each basis label to the detector error model for the tomography
            circuit in that basis.
        """
        dem_circuits = self._postselection_exp_cache.dem_circuits
        if dem_circuits is None:
            raise RuntimeError("dem_circuits must be called before dems.")
        dems = {
            basis: circ.detector_error_model(approximate_disjoint_errors=True)  # type: ignore[attr-defined]
            for basis, circ in dem_circuits.items()
        }
        self._postselection_exp_cache.dems = dems
        return dems

    def initialize_decoders(self) -> dict[str, tuple[ConfidenceDecoder, BaseDecoder]]:
        """
        Initializes the decoders for the tomography circuits in each basis from the detector error models.
        This function utilizes correlated decoding (https://arxiv.org/abs/2403.03272).
        Because of this, this function initializes two decoders: one for the decoding the ancilla qubits, and one for decoding the output qubit.
        The decoder that decodes the ancilla qubits decodes them jointly, taking in (num_ancillas * 3) detectors and outputting corrections for (num_ancillas) observables.
            This decoder does not additionally take in the output qubit's detectors for the case that the output qubit might be used later in the computation.
        The decoder that decodes the output qubit decodes it jointly with the ancilla qubits, taking in ((num_ancillas + 1) * 3) detectors and outputting corrections for 1 observable (on the output qubit).
        This function assumes that the first qubit is the output qubit, and also assumes that we are using the [[7, 1, 3]] Steane code, which has three detectors per logical qubit.

        Note: this function depends on dems() being called.

        Returns:
            dict[str, tuple[ConfidenceDecoder, BaseDecoder]]: A dictionary mapping each basis label to (factory_decoder, full_decoder), where factory_decoder
            decodes the ancilla qubits jointly and full_decoder decodes the output qubit jointly with the information on all logical qubits.
        """
        dems_bases = self._postselection_exp_cache.dems
        if dems_bases is None:
            raise RuntimeError("dems must be called before initialize_decoders.")
        decoders_bases: dict[str, tuple[ConfidenceDecoder, BaseDecoder]] = {}
        # NOTE: there is a question of if we want to support multi-qubit
        # tomography in this experiment. For now, probably not; if we did, we
        # would have to specify the number of output qubits and their locations
        # and use that information to construct a custom layout.
        layout = _DEFAULT_SYNDROME_LAYOUT
        for basis_label, dem_base in dems_bases.items():
            # We subset the detector error models for the "full" and "factory" decoders to
            # extract only the detectors and observables that are used by the respective decoders.
            full_dem = _sub_detector_error_model(
                dem_base,
                detector_indices=range(dem_base.num_detectors),
                observable_indices=range(layout.output_observable_count),
            )
            factory_dem = _sub_detector_error_model(
                dem_base,
                detector_indices=range(
                    layout.output_detector_count, dem_base.num_detectors
                ),
                observable_indices=range(
                    layout.output_observable_count,
                    dem_base.num_observables,
                ),
            )
            # NOTE: We are initializing the factory and full decoders twice. For the TableDecoder, this means that we are sampling two times for each basis
            # to build the 2 decoders. It might be more efficient if, from the same sampled data, we constructed the factory and full decoders (so we just
            # have to sample once per basis). But, the current TableDecoder does not support that API (as we assume that training happens during
            # initialization of the TableDecoder).
            # TODO: this is because the current API does NOT support decoder(dem, **kwargs) for the constructor so we have to
            # do a cast. We would want to decide on the decoder __init__ API to be able to avoid such a cast (e.g., to have **kwargs in the __init__ function
            # in the abstract class)
            # The reason why the factory decoder is a ConfidenceDecoder is that technically, we only need confidence scores when decoding the ancilla qubits.
            decoder_constructor = cast(Callable[..., ConfidenceDecoder], self.decoder)
            full_decoder = cast(
                BaseDecoder,
                decoder_constructor(full_dem, **self.decoder_init_args),
            )
            factory_decoder = decoder_constructor(factory_dem, **self.decoder_init_args)
            decoders_bases[basis_label] = (factory_decoder, full_decoder)
        self._postselection_exp_cache.decoders_with_confidence = decoders_bases
        return decoders_bases

    def make_tasks(
        self,
        # TODO: Ideally, we don't want to make the type of the device GeminiLogicalSimulator (this should be flexible to support the type representing
        # the hardware). However, because GeminiLogicalSimulator currently doesn't obey any interface/inheritance for a generic hardware object,
        # we can't really replace this with a more specific type without changing the design of GeminiLogicalSimulator.
        device: GeminiLogicalSimulator,
        # TODO: the return type of make_tasks of what we want to run on hardware should not be GeminiLogicalSimulatorTask. It should be
        # TaskABC[GeminiLogicalFuture]. However, currently, GeminiLogicalSimulatorTask does not inherit from the abstract "task" types in
        # bloqade-core.
    ) -> dict[str, GeminiLogicalSimulatorTask[_LogicalTomographyReturn]]:
        """Prepares tasks for submission to the hardware device."""
        dem_kernels = self._postselection_exp_cache.dem_kernels
        if dem_kernels is None:
            raise RuntimeError("kernels must be called before make_tasks.")
        actual_tasks: dict[
            str, GeminiLogicalSimulatorTask[_LogicalTomographyReturn]
        ] = {
            basis: device.task(kernel.similar())
            for basis, kernel in dem_kernels.items()
        }
        self._postselection_exp_cache.hardware_tasks = actual_tasks
        return actual_tasks

    # NOTE: This is NOT idempotent. Calling it multiple times WILL give you DIFFERENT sample data, because you will re-sample from the hardware.
    # NOTE: Each experiment evaluates on a DIFFERENT set of 'hardware' samples. This might not be what we want (is expensive).
    # ^ We might want a way to reuse hardware samples across different "experiments".
    # NOTE: In the future, we may want to define a "get_samples_async" method that allows
    # the user to get futures and later request to actually get the samples on their own time (a "non-blocking" implementation).
    def get_samples(
        self,
        num_shots: int,
    ) -> dict[str, _BasisDataset]:
        """For each basis, samples num_shots from the hardware. Returns the detector and observable information for each task."""
        actual_tasks = self._postselection_exp_cache.hardware_tasks
        if actual_tasks is None:
            raise RuntimeError("make_tasks must be called before get_samples.")
        # In this implementation, we request samples for each basis in parallel, and then iteratively block
        # on X, Y, and then Z being finished.
        futures = {
            basis: task.run_async(num_shots) for basis, task in actual_tasks.items()
        }
        actual_data = {
            basis: _basis_dataset_from_task_result(future.result())
            for basis, future in futures.items()
        }
        self._postselection_exp_cache.raw_results = actual_data
        return actual_data

    def decode_and_postselect(
        self,
        postselection_condition: np.ndarray,
        progress_label: str | bool = False,
    ) -> dict[str, _DecodedPostselectionResult]:
        """
        With the resulting shot data from the hardware samples, runs the following steps:
        1. Decoding and correction on the ancilla qubits
        2. Filtering shots whose corrected ancilla observable is a valid pattern in postselection_condition
        3. Out of these filtered shots, decode and correct the output qubit
        4. Returns the corrected observables as well as the confidence score associated with each observable.

        Args:
            postselection_condition (np.ndarray): A 2D numpy array of shape (num_conditions, num_ancillae) representing the valid ancilla patterns to postselect on.
                Example: If postselection_condition = np.array([[1, 0, 1, 1]]), then we will only accept shots where the first ancilla is 1, the second ancilla is 0, the third ancilla is 1, and the fourth ancilla is 1
                after those ancilla have been corrected by the decoder.
            progress_label (str | bool): If False, no progress bar is displayed. If True,
                the decoder class name is used as the progress-bar label. If a string, that
                string is used as the progress-bar label. Defaults to False.

        Returns:
            dict[str, _DecodedPostselectionResult]: A dictionary that maps each basis to postselected observables and confidence scores per shot.
        """
        actual_data = self._postselection_exp_cache.raw_results
        decoder_map = self._postselection_exp_cache.decoders_with_confidence
        if actual_data is None:
            raise RuntimeError("get_samples must be called before decoding.")
        if decoder_map is None:
            raise RuntimeError("initialize_decoders must be called before decoding.")
        targets = np.asarray(postselection_condition, dtype=np.uint8)
        if targets.ndim != 2:
            raise ValueError("postselection_condition must be a 2D array.")
        basis_labels = list(decoder_map.keys())
        decoded_results = _build_generic_threshold_tables(
            actual_data,
            decoder_map,
            targets=targets,
            basis_labels=basis_labels,
            progress_label=(
                self.decoder.__name__ if progress_label is True else progress_label
            ),
        )
        self._postselection_exp_cache.decoded_results = decoded_results
        return decoded_results

    def analysis_f_vs_fraction(
        self,
        *,
        target_bloch: np.ndarray = _DEFAULT_TARGET_BLOCH,
        threshold_points: int = 64,
        min_accepted_per_basis: int = 50,
    ) -> PostselectionCurveData:
        """
        Analyzes the shot data to produce arrays for the fidelity to some target state as a function of the fraction of total accepted shots.
        Produces various thresholds of accepted shots by thresholding on the confidence score associated with each shot.

        Args:
            target_bloch (np.ndarray): The bloch vector to which fidelity is computed. Defaults to np.array([1.0, 1.0, 1.0]) / np.sqrt(3).
            threshold_points (int): The number of thresholds that we would like to compute. We get thresholds at evenly spaced quantiles on the array of
            confidence scores across all shots in each basis. Defaults to 64.
            min_accepted_per_basis (int): The minimum number of shots that we require per basis in order to do tomography. This is to prevent very
            inaccurate estimates for the tomography due to shot count in a basis being very low. Defaults to 50.

        Returns:
            PostselectionCurveData: The accepted fractions and point fidelities
            for the thresholded curve.
        """
        decoded_results = self._postselection_exp_cache.decoded_results
        actual_data = self._postselection_exp_cache.raw_results
        if decoded_results is None:
            raise RuntimeError("decode_and_postselect must be called before analysis.")
        if actual_data is None:
            raise RuntimeError("get_samples must be called before analysis.")
        basis_labels = tuple(decoded_results.keys())
        total_shots = sum(len(dataset.observables) for dataset in actual_data.values())
        # NOTE: In the current implementation, it is possible to get less than "threshold" points if many of your confidence scores end up being the same.
        # Exploring alternative implementations to identify confidence thresholds can be done in a future PR.
        threshold_curve = _evaluate_cached_threshold_curve(
            decoded_results,
            threshold_points=threshold_points,
            target_bloch=target_bloch,
            basis_labels=basis_labels,
            min_accepted_per_basis=min_accepted_per_basis,
            total_shots=total_shots,
        )
        self._postselection_exp_cache.thresholded_data = threshold_curve
        return threshold_curve

    # NOTE: The maximum number of shots that you can "accept" is equal to the
    # number of shots that pass decoding of the ancilla after all observables on ancilla
    # are 0.
    def tomography_result(
        self,
        accepted_fraction: float,
    ) -> TomographyResult:
        """
        Returns a TomographyResult after decoding and postselection with the option to manually specify the fraction
        of shots accepted.

        The implementation will find the minimum number of shots to accept where the fraction of accepted shots is >= accepted_fraction,
        sorted by highest to lowest confidence.

        Args:
            accepted_fraction (float): The fraction of shots to accept out of shots that passed postselection. Note that if accepted_fraction == 1.0, this means
            accepting all shots that passed postselection, not all shots used to perform the experiment.

        Returns:
            TomographyResult: The resulting shots in each basis based on the accepted_fraction provided.
        """

        decoded_results = self._postselection_exp_cache.decoded_results
        if decoded_results is None:
            raise RuntimeError(
                "decode_and_postselect must be called before tomography_result."
            )

        basis_labels = tuple(decoded_results.keys())
        shots_by_basis = _shots_at_accepted_fraction(
            decoded_results,
            accepted_fraction,
            basis_labels=basis_labels,
        )
        return TomographyResult(shots_by_basis)

    def analysis_visualization(
        self, min_accepted_fraction: float = 0.04, title: str | None = None
    ):
        """Plots the curve of the fidelity vs. accepted fraction, with a cutoff of min_accepted_fraction as well as a supplied title."""
        if self._postselection_exp_cache.thresholded_data is None:
            raise RuntimeError(
                "analysis_f_vs_fraction must be called before visualization."
            )
        return _plot_decoder_curves(
            {"decoder": self._postselection_exp_cache.thresholded_data},
            min_accepted_fraction=min_accepted_fraction,
            title=title,
        )
