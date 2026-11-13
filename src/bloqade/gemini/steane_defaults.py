"""Default Steane [[7,1,3]] detector and observable annotation matrices.

The Steane code encodes one logical qubit into seven physical qubits.
These matrices define the parity checks (detectors) and logical
observables for use with the Gemini logical simulator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag

# Steane [[7,1,3]] X-stabilizer parity check matrix (3 detectors per logical qubit)
STEANE7_DETECTOR_MATRIX: NDArray[np.int_] = np.array(
    [
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1],
    ]
)

# Steane [[7,1,3]] logical X observable (1 observable per logical qubit)
STEANE7_OBSERVABLE_MATRIX: NDArray[np.int_] = np.array([[1, 1, 0, 0, 0, 1, 0]])

# Physical qubits per logical qubit. Read off the parity-check matrix rather
# than written as a literal so it cannot drift from the matrices above.
STEANE7_PHYSICAL_QUBITS: int = STEANE7_DETECTOR_MATRIX.shape[1]


def steane7_m2dets(num_qubits: int) -> list[list[int]]:
    """Build the measurement-to-detector matrix for ``num_qubits`` Steane-encoded qubits.

    Args:
        num_qubits (int): Number of logical qubits.

    Returns:
        list[list[int]]: Binary matrix of shape ``(7 * num_qubits, 3 * num_qubits)``
            mapping physical measurements to detectors.

    """
    result = np.asarray(block_diag(*[STEANE7_DETECTOR_MATRIX.T] * num_qubits))
    return result.tolist()


def steane7_m2obs(num_qubits: int) -> list[list[int]]:
    """Build the measurement-to-observable matrix for ``num_qubits`` Steane-encoded qubits.

    Args:
        num_qubits (int): Number of logical qubits.

    Returns:
        list[list[int]]: Binary matrix of shape ``(7 * num_qubits, num_qubits)``
            mapping physical measurements to observables.

    """
    result = np.asarray(block_diag(*[STEANE7_OBSERVABLE_MATRIX.T] * num_qubits))
    return result.tolist()
