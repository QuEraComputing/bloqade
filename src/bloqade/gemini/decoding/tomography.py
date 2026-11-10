"""Minimal single-qubit tomography helpers for MSD postselection demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

_DEFAULT_TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)


def _density_matrix_from_bloch(bloch: Mapping[str, float]) -> np.ndarray:
    if len(bloch) != 3:
        raise ValueError(
            f"Bloch vectors with {len(bloch)} keys are not supported; "
            "single-qubit tomography requires X, Y, and Z keys."
        )
    required_keys = {"X", "Y", "Z"}
    if set(bloch) != required_keys:
        raise ValueError("Single-qubit tomography requires X, Y, and Z keys.")

    x = float(bloch["X"])
    y = float(bloch["Y"])
    z = float(bloch["Z"])
    return 0.5 * np.array(
        [[1.0 + z, x - 1j * y], [x + 1j * y, 1.0 - z]],
        dtype=np.complex128,
    )


def _bloch_mapping_from_sequence(
    bloch: np.ndarray | Sequence[float],
) -> dict[str, float]:
    bloch_arr = np.asarray(bloch, dtype=np.float64)
    if bloch_arr.shape != (3,):
        raise ValueError("bloch must be a length-3 vector.")
    return {
        "X": float(bloch_arr[0]),
        "Y": float(bloch_arr[1]),
        "Z": float(bloch_arr[2]),
    }


def _validate_single_qubit_bloch_vector(
    bloch: np.ndarray | Sequence[float],
    *,
    tol: float,
) -> dict[str, float]:
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    bloch_arr = np.asarray(bloch, dtype=np.float64)
    if bloch_arr.shape != (3,):
        raise ValueError("bloch must be a length-3 vector.")
    bloch_norm_squared = float(np.dot(bloch_arr, bloch_arr))
    if bloch_norm_squared > 1.0 + tol:
        raise ValueError("Single-qubit Bloch vector must have squared norm <= 1.")
    return _bloch_mapping_from_sequence(bloch_arr)


def _single_qubit_fidelity(
    density_matrix: np.ndarray,
    target_density_matrix: np.ndarray,
) -> float:
    overlap = float(np.real(np.trace(density_matrix @ target_density_matrix)))
    det_product = float(
        np.real(np.linalg.det(density_matrix))
        * np.real(np.linalg.det(target_density_matrix))
    )
    return overlap + 2.0 * math.sqrt(max(det_product, 0.0))


@dataclass(frozen=True, init=False)
class TomographyResult:
    """Point-estimate single-qubit tomography result."""

    density_matrix: np.ndarray

    def __init__(
        self,
        shots_by_basis: Mapping[str, np.ndarray],
    ) -> None:
        """
        Create a tomography result by computing the density matrix from the shots per basis.

        Args:
            shots_by_basis (Mapping[str, np.ndarray]): A mapping of each basis to an array of shots (0/1's) in each basis.
        """
        zero_counts: dict[str, int] = {}
        one_counts: dict[str, int] = {}
        totals: dict[str, int] = {}
        bloch: dict[str, float] = {}
        for basis in shots_by_basis:
            shots = np.asarray(shots_by_basis[basis], dtype=np.uint8)
            if shots.ndim != 2 or shots.shape[1] != 1:
                raise ValueError(
                    "TomographyResult expects each basis to have shape (shots, 1)."
                )
            zero_counts[basis] = int(np.count_nonzero(shots[:, 0] == 0))
            one_counts[basis] = int(np.count_nonzero(shots[:, 0] == 1))
            totals[basis] = max(zero_counts[basis] + one_counts[basis], 1)
            bloch[basis] = (zero_counts[basis] - one_counts[basis]) / totals[basis]

        object.__setattr__(self, "density_matrix", _density_matrix_from_bloch(bloch))

    # NOTE: if you want to add more generic methods for fidelity, to density matrices, just define a new method "fidelity_to_density_mat".
    def fidelity_bloch(
        self,
        target_bloch: np.ndarray | Sequence[float],
        tol: float = 1e-10,
    ) -> float:
        """Return the fidelity to a target state from its Bloch vector."""

        target_density_matrix = _density_matrix_from_bloch(
            _validate_single_qubit_bloch_vector(target_bloch, tol=tol)
        )
        return _single_qubit_fidelity(self.density_matrix, target_density_matrix)


__all__ = [
    "TomographyResult",
]
