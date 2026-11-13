from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _BasisDataset:
    """Detector and observable samples for one tomography basis."""

    detectors: np.ndarray
    observables: np.ndarray
