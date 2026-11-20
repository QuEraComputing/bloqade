from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _SyndromeLayout:
    """Column layout describing output versus factory syndrome bits.

    Attributes:
        output_detector_count: Number of leading detector columns that belong
            to the output logical qubit.
        output_observable_count: Number of leading observable columns that
            belong to the output logical qubit.
    """

    output_detector_count: int = 3
    output_observable_count: int = 1


_DEFAULT_SYNDROME_LAYOUT = _SyndromeLayout()


def _split_factory_bits(
    detectors: np.ndarray,
    observables: np.ndarray,
    *,
    layout: _SyndromeLayout = _DEFAULT_SYNDROME_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return factory detector and observable bits after output columns.

    Args:
        detectors: Detector sample matrix.
        observables: Observable sample matrix.
        layout: Syndrome layout specifying how many leading columns are output
            bits.

    Returns:
        A pair ``(factory_detectors, factory_observables)``.
    """

    return (
        detectors[:, layout.output_detector_count :],
        observables[:, layout.output_observable_count :],
    )
