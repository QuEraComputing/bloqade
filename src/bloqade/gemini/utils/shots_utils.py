"""Dense histograms for binary measurement shots."""

import numpy as np


def get_histogram(shots: np.ndarray) -> np.ndarray:
    """Count binary outcomes in a two-dimensional array of measurement shots.

    Args:
        shots: Binary array of shape ``(n_shots, n_bits)``. The first column
            is the most significant bit.

    Returns:
        Integer counts of shape ``(2**n_bits,)``. Bin ``i`` counts the shots
        whose binary representation equals ``i``, including zero-count bins.
        With zero bit columns, the single bin counts all empty outcomes.
    """
    n = shots.shape[1]
    weights = 1 << np.arange(n - 1, -1, -1)
    indices = shots.astype(np.int64) @ weights
    return np.bincount(indices, minlength=1 << n)
