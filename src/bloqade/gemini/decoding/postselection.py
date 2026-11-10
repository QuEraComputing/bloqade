from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from bloqade.decoders import BaseDecoder

from bloqade.gemini.decoding.layout import _DEFAULT_SYNDROME_LAYOUT, _split_factory_bits
from bloqade.gemini.decoding.sampling import _BasisDataset
from bloqade.gemini.decoding.confidence import ConfidenceDecoder
from bloqade.gemini.decoding.tomography import _DEFAULT_TARGET_BLOCH, TomographyResult

_DEFAULT_BASIS_LABELS = ("X", "Y", "Z")
_ProgressLabel = str | bool


@dataclass(frozen=True)
class _DecodedPostselectionResult:
    """Decoded output observables and one factory confidence per accepted shot."""

    observables: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class PostselectionCurveData:
    """Point-estimate fidelity curve from thresholded postselection."""

    accepted_fraction: np.ndarray
    fidelity: np.ndarray
    point_fidelity: np.ndarray


def _pack_boolean_array(arr: np.ndarray) -> np.ndarray:
    """Pack each bit row into a little-endian integer label."""

    bits = np.asarray(arr, dtype=np.uint64)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    if bits.ndim != 2:
        raise ValueError("arr must be a 1D or 2D bit array.")
    if bits.shape[1] > 64:
        raise ValueError("Cannot pack more than 64 bits into uint64 values.")
    return np.sum(bits << np.arange(bits.shape[1], dtype=np.uint64), axis=1)


def _decode_detector_batch(
    decoder: BaseDecoder, detector_bits: np.ndarray
) -> np.ndarray:
    """Decode unique detector rows, falling back to row-wise decoding if needed."""

    detector_bits = np.asarray(detector_bits, dtype=np.bool_)
    if detector_bits.ndim != 2:
        raise ValueError("detector_bits must have shape (shots, detectors).")
    if detector_bits.shape[0] == 0:
        return np.zeros(
            (0, int(getattr(decoder, "num_observables", 0))),
            dtype=np.uint8,
        )

    decoded = np.asarray(decoder.decode(detector_bits), dtype=np.uint8)
    if decoded.ndim == 2 and decoded.shape[0] == detector_bits.shape[0]:
        return decoded
    if decoded.ndim == 1 and detector_bits.shape[0] == 1:
        return decoded.reshape(1, -1)

    # Some simple decoder test doubles and non-batched decoders interpret a 2D
    # input as one object. Fall back to explicit row-wise calls in that case.
    rows = [
        np.asarray(decoder.decode(row), dtype=np.uint8).reshape(-1)
        for row in detector_bits
    ]
    return np.stack(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.uint8)


def _decode_confidence_batch(
    decoder: ConfidenceDecoder,
    detector_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode unique factory detector rows and return corrections/confidences."""

    detector_bits = np.asarray(detector_bits, dtype=np.bool_)
    if detector_bits.ndim != 2:
        raise ValueError("detector_bits must have shape (shots, detectors).")
    if detector_bits.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros(0, dtype=np.float64)

    rows = [decoder.decode_with_confidence(row) for row in detector_bits]
    corrections = [
        np.asarray(correction, dtype=np.uint8).reshape(-1) for correction, _ in rows
    ]
    confidences = [float(np.float64(confidence)) for _, confidence in rows]
    return (
        (
            np.stack(corrections, axis=0)
            if corrections
            else np.zeros((0, 0), dtype=np.uint8)
        ),
        np.asarray(confidences, dtype=np.float64),
    )


def _resolve_progress_label(
    progress_label: _ProgressLabel,
    decoder_map: Mapping[str, tuple[ConfidenceDecoder, BaseDecoder]],
) -> str | None:
    if progress_label is False:
        return None
    if progress_label is True:
        first_decoder_pair = next(iter(decoder_map.values()), None)
        if first_decoder_pair is None:
            return "decoder"
        return type(first_decoder_pair[0]).__name__
    return progress_label


def _build_generic_threshold_tables(
    actual_data: Mapping[str, _BasisDataset],
    decoder_map: Mapping[str, tuple[ConfidenceDecoder, BaseDecoder]],
    *,
    targets: np.ndarray,
    basis_labels: Sequence[str],
    progress_label: _ProgressLabel = False,
) -> dict[str, _DecodedPostselectionResult]:
    """Decode, factory-postselect, and keep output bits with confidence.

    Each basis result stores decoded output observable shots with shape
    ``(shots, output_observables)`` and a separate confidence vector with shape
    ``(shots,)``.
    """

    packed_targets = {
        int(x) for x in _pack_boolean_array(np.asarray(targets, dtype=np.uint8))
    }
    decoded_results: dict[str, _DecodedPostselectionResult] = {}
    progress_bars = {}
    resolved_progress_label = _resolve_progress_label(progress_label, decoder_map)

    if resolved_progress_label is not None:
        from tqdm.auto import tqdm

        progress_bars = {
            basis: tqdm(
                total=len(actual_data[basis].observables),
                desc=f"{resolved_progress_label} {basis}: decoded",
                unit="shots",
                position=index,
                leave=True,
            )
            for index, basis in enumerate(basis_labels)
        }

    try:
        for basis in basis_labels:
            dataset = actual_data[basis]
            factory_decoder, full_decoder = decoder_map[basis]
            anc_det, anc_obs = _split_factory_bits(
                dataset.detectors,
                dataset.observables,
                layout=_DEFAULT_SYNDROME_LAYOUT,
            )
            progress_bar = progress_bars.get(basis)

            _unique_anc_packed, unique_anc_indices, inverse_anc = np.unique(
                _pack_boolean_array(anc_det),
                return_index=True,
                return_inverse=True,
            )
            unique_anc_det = anc_det[unique_anc_indices]
            unique_anc_corrections, unique_confidence = _decode_confidence_batch(
                factory_decoder,
                unique_anc_det,
            )

            packed_anc_obs = _pack_boolean_array(anc_obs)
            packed_anc_correction = _pack_boolean_array(unique_anc_corrections)
            corrected_factory = packed_anc_obs ^ packed_anc_correction[inverse_anc]
            accepted_mask = np.isfinite(unique_confidence[inverse_anc]) & np.isin(
                corrected_factory,
                list(packed_targets),
            )

            if not np.any(accepted_mask):
                decoded_results[basis] = _DecodedPostselectionResult(
                    observables=np.zeros((0, 1), dtype=np.uint8),
                    confidence=np.zeros(0, dtype=np.float64),
                )
                if progress_bar is not None:
                    progress_bar.update(dataset.detectors.shape[0])
                continue

            accepted_detectors = dataset.detectors[accepted_mask]
            _unique_full_packed, unique_full_indices, inverse_full = np.unique(
                _pack_boolean_array(accepted_detectors),
                return_index=True,
                return_inverse=True,
            )
            unique_full_detectors = accepted_detectors[unique_full_indices]
            unique_full_corrections = _decode_detector_batch(
                full_decoder,
                unique_full_detectors,
            )

            output_bits = np.asarray(
                dataset.observables[accepted_mask, :1], dtype=np.uint8
            )
            correction_bits = np.asarray(
                unique_full_corrections[inverse_full, : output_bits.shape[1]],
                dtype=np.uint8,
            )
            decoded_output = output_bits ^ correction_bits
            confidence_rows = unique_confidence[inverse_anc][accepted_mask]

            decoded_results[basis] = _DecodedPostselectionResult(
                observables=decoded_output.astype(np.uint8, copy=False),
                confidence=np.asarray(confidence_rows, dtype=np.float64),
            )
            if progress_bar is not None:
                progress_bar.update(dataset.detectors.shape[0])
    finally:
        for progress_bar in progress_bars.values():
            progress_bar.close()

    return decoded_results


def _shots_at_accepted_fraction(
    decoded_results: Mapping[str, _DecodedPostselectionResult],
    accepted_fraction: float,
    *,
    basis_labels: Sequence[str] = _DEFAULT_BASIS_LABELS,
) -> dict[str, np.ndarray]:
    """Return decoded observable shots after a global confidence top-k cut."""

    if not 0.0 <= accepted_fraction <= 1.0:
        raise ValueError("accepted_fraction must be between 0 and 1.")

    total_postselected = sum(
        int(decoded_results[basis].observables.shape[0]) for basis in basis_labels
    )
    empty = {
        basis: np.zeros(
            (0, decoded_results[basis].observables.shape[1]), dtype=np.uint8
        )
        for basis in basis_labels
    }
    if total_postselected == 0 or accepted_fraction == 0.0:
        return empty

    target_count = min(
        total_postselected,
        max(1, int(np.ceil(float(accepted_fraction) * total_postselected))),
    )
    confidences: list[np.ndarray] = []
    basis_ids: list[np.ndarray] = []
    shot_ids: list[np.ndarray] = []
    for basis_index, basis in enumerate(basis_labels):
        basis_results = decoded_results[basis]
        count = basis_results.observables.shape[0]
        confidences.append(basis_results.confidence)
        basis_ids.append(np.full(count, basis_index, dtype=np.int64))
        shot_ids.append(np.arange(count, dtype=np.int64))

    all_confidences = np.concatenate(confidences)
    all_basis_ids = np.concatenate(basis_ids)
    all_shot_ids = np.concatenate(shot_ids)
    order = np.argsort(-all_confidences, kind="stable")[:target_count]

    selected: dict[str, list[np.ndarray]] = {basis: [] for basis in basis_labels}
    for flat_index in order:
        basis = basis_labels[int(all_basis_ids[flat_index])]
        shot_index = int(all_shot_ids[flat_index])
        selected[basis].append(decoded_results[basis].observables[shot_index])

    return {
        basis: (
            np.asarray(selected[basis], dtype=np.uint8)
            if selected[basis]
            else np.zeros(
                (0, decoded_results[basis].observables.shape[1]), dtype=np.uint8
            )
        )
        for basis in basis_labels
    }


# NOTE: We can consider moving this function to bloqade-core if it is "generic" enough.
def _evaluate_cached_threshold_curve(
    decoded_results: Mapping[str, _DecodedPostselectionResult],
    *,
    total_shots: int,
    threshold_points: int = 64,
    target_bloch: np.ndarray = _DEFAULT_TARGET_BLOCH,
    basis_labels: Sequence[str] = _DEFAULT_BASIS_LABELS,
    min_accepted_per_basis: int = 50,
) -> PostselectionCurveData:
    """Evaluate a point-fidelity curve from decoded shots and confidences."""

    confidence_arrays = [
        decoded_results[basis].confidence
        for basis in basis_labels
        if decoded_results[basis].observables.shape[0] > 0
    ]
    if not confidence_arrays:
        raise RuntimeError("No factory-accepted shots found for threshold sweep.")
    confidences = np.sort(np.concatenate(confidence_arrays))
    threshold_indices = np.rint(
        np.linspace(0, len(confidences) - 1, threshold_points)
    ).astype(np.int64)
    thresholds = np.unique(confidences[threshold_indices])

    accepted_fractions: list[float] = []
    fidelities: list[float] = []
    for threshold in thresholds:
        shots_by_basis: dict[str, np.ndarray] = {}
        accepted_count = 0
        for basis in basis_labels:
            basis_results = decoded_results[basis]
            keep = basis_results.confidence >= float(threshold)
            shots = basis_results.observables[keep].astype(np.uint8, copy=False)
            shots_by_basis[basis] = shots
            accepted_count += int(shots.shape[0])

        if (
            min(shots_by_basis[basis].shape[0] for basis in basis_labels)
            < min_accepted_per_basis
        ):
            continue

        point = TomographyResult(shots_by_basis).fidelity_bloch(target_bloch)
        accepted_fractions.append(accepted_count / total_shots)
        fidelities.append(point)

    order = np.argsort(accepted_fractions)
    accepted = np.asarray(accepted_fractions, dtype=np.float64)[order]
    fidelity = np.asarray(fidelities, dtype=np.float64)[order]
    return PostselectionCurveData(
        accepted_fraction=accepted,
        fidelity=fidelity,
        point_fidelity=fidelity.copy(),
    )
