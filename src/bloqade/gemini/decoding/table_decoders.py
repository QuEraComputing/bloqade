"""Table decoder used by the MSD postselection notebook."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import stim
import numpy as np

from bloqade.gemini.decoding.confidence import (
    ConfidenceDecoder,
    _validate_detector_bits,
)

# NOTE: We will plan to move this code to bloqade-decoders.

_TABLE_DECODER_MISSING_MSG = (
    "Table confidence decoding requires the optional table decoder. Install it "
    "with `bloqade[decoding]`."
)

if TYPE_CHECKING:
    from bloqade.decoders import TableDecoder
    from bloqade.decoders._decoders.mld.utils import (
        shots_to_counts,
        pack_boolean_array,
    )
else:
    try:
        from bloqade.decoders import TableDecoder
        from bloqade.decoders._decoders.mld.utils import (
            shots_to_counts,
            pack_boolean_array,
        )
    except ImportError:

        class TableDecoder:  # type: ignore[no-redef]
            """Runtime stub used when the optional TableDecoder is absent."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError(_TABLE_DECODER_MISSING_MSG)

        def _missing_table_decoder(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(_TABLE_DECODER_MISSING_MSG)

        pack_boolean_array = _missing_table_decoder
        shots_to_counts = _missing_table_decoder

logger = logging.getLogger(__name__)

_COUNT_DTYPE = np.uint32
_COUNT_MAX = np.iinfo(_COUNT_DTYPE).max


def _as_uint32_count_table(counts: np.ndarray) -> np.ndarray:
    arr = np.asarray(counts)
    if np.any(arr < 0):
        raise ValueError("det_obs_counts cannot contain negative counts.")
    if np.any(arr > _COUNT_MAX):
        raise OverflowError(
            f"det_obs_counts contains a value larger than uint32 max ({_COUNT_MAX})."
        )
    return arr.astype(_COUNT_DTYPE, copy=False)


# NOTE: When we migrate TableDecoderWithConfidence, we will add a shim import
# from bloqade.decoders import TableDecoderWithConfidence
# as well as a deprecation warning saying that this is not the "preferred" import path and that users should import from bloqade.decoders.
# NOTE (breaking change): for the TableDecoder constructor in bloqade.decoders, we expect a breaking change to not feed in det_obs_counts in the __init__ in TableDecoder.
# ^ For the public API for decoders, we just want Decoder(dem).
# NOTE (breaking change): we expect to convert the datatype from int64 to uint32 in TableDecoder for space savings.
class TableDecoderWithConfidence(TableDecoder, ConfidenceDecoder):
    # make private most stuff that other decoders don't have.
    # TODO: only expose decode_with_confidence.
    """Dense table decoder with empirical per-syndrome confidence."""

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        num_shots: int = 10**7,
        seed: int | None = None,
        step_size: int | None = None,
    ) -> None:
        data_len = dem.num_detectors + dem.num_observables
        if data_len > 64:
            raise ValueError(
                f"Total data length {data_len} (detectors + observables) "
                "exceeds 64 bits and cannot be packed into int64."
            )

        counts_table = np.zeros(2**data_len, dtype=_COUNT_DTYPE)

        super().__init__(dem=dem, det_obs_counts=counts_table)
        self._det_obs_counts = _as_uint32_count_table(self._det_obs_counts)
        self._correction_confidence: np.ndarray | None = None

        if data_len == 0:
            self._det_obs_counts[0] = max(1, min(int(num_shots), _COUNT_MAX))
            return
        self._train_from_dem(num_shots=num_shots, seed=seed, step_size=step_size)

    def _train_from_dem(
        self,
        *,
        num_shots: int,
        seed: int | None,
        step_size: int | None,
    ) -> None:
        if num_shots < 0:
            raise ValueError("num_shots must be non-negative.")
        if num_shots == 0:
            return
        if step_size is None:
            step_size = num_shots
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        from tqdm import tqdm

        sampler = self._dem.compile_sampler(seed=seed)
        progress_bar_steps = ((num_shots - 1) // step_size) + 1
        total_sampled = 0

        logger.info("Building decoder from detector error model...")
        for _ in tqdm(range(progress_bar_steps)):
            next_shots = min(step_size, num_shots - total_sampled)
            total_sampled += next_shots
            det_samples, obs_samples = sampler.sample(
                shots=next_shots,
                bit_packed=False,
            )[:2]
            self._update_det_obs_counts(
                np.concatenate([det_samples, obs_samples], axis=1)
            )

    def _update_det_obs_counts(self, det_obs_shots: np.ndarray) -> None:
        shots = np.asarray(det_obs_shots, dtype=np.uint8)
        expected_width = self.num_detectors + self.num_observables
        if shots.ndim != 2 or shots.shape[1] != expected_width:
            raise ValueError(
                f"Expected det_obs_shots with shape (N, {expected_width}), "
                f"got {shots.shape}."
            )

        step_counts = shots_to_counts(shots)
        if np.any(step_counts > _COUNT_MAX - self._det_obs_counts):
            raise OverflowError(
                f"TableDecoder count table would exceed uint32 max ({_COUNT_MAX})."
            )
        self._det_obs_counts += step_counts.astype(_COUNT_DTYPE, copy=False)
        self._is_cached_df = False
        self._is_cached_correction = False
        self._correction_confidence = None

    def decode(self, detector_bits: np.ndarray) -> np.ndarray:
        """Decode detector bits after validating the detector-shot width."""

        validated_bits = _validate_detector_bits(
            detector_bits,
            self.num_detectors,
            allow_batch=True,
            method_name="decode",
        )
        return super().decode(validated_bits)

    def cache_correction(self) -> None:
        # TableDecoder.decode dispatches to this method name, so keep the
        # override while routing the implementation through the private helper.
        self._cache_correction()

    def _cache_correction(self) -> None:
        """Cache maximum-likelihood corrections and their empirical confidence."""

        if self._is_cached_correction and self._correction_confidence is not None:
            return

        # NOTE: this current implementation with cache both the correction AND the confidence
        # even if you don't use it (simplifies the implementation a bit). In the future,
        # if you don't want to use the confidence, maybe you can specify as a flag in the
        # decoder constructor to not cache the confidence.
        obs_counts = self._det_obs_counts.reshape(
            2**self.num_observables,
            2**self.num_detectors,
        )
        self._maximum_likelihood_correction = np.argmax(obs_counts, axis=0).reshape(-1)

        max_counts = np.max(obs_counts, axis=0).astype(np.float64, copy=False)
        total_counts = np.sum(obs_counts, axis=0, dtype=np.uint64)
        confidence = np.full(total_counts.shape, np.nan, dtype=np.float64)
        np.divide(
            max_counts,
            total_counts.astype(np.float64, copy=False),
            out=confidence,
            where=total_counts > 0,
        )
        self._correction_confidence = confidence
        self._is_cached_correction = True

    def decode_with_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        """Decode one detector syndrome and return its empirical confidence."""

        validated_bits = _validate_detector_bits(
            detector_bits,
            self.num_detectors,
            allow_batch=False,
            method_name="decode_with_confidence",
        )
        correction = np.asarray(self.decode(validated_bits), dtype=np.bool_)
        assert self._correction_confidence is not None
        packed = int(
            pack_boolean_array(
                np.asarray(validated_bits, dtype=np.uint8).reshape(1, -1)
            )[0]
        )
        return correction, np.float64(self._correction_confidence[packed])

    # TODO: can implement batch decoding on a batch of det_obs_counts here with confidence (
    # calls the method decode_det_obs_counts in TableDecoder and also returns the batch of confidence scores)


__all__ = ["TableDecoderWithConfidence"]
