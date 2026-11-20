"""Tests for dense histograms of binary measurement shots."""

import numpy as np

from bloqade.gemini.utils import get_histogram


def test_get_histogram_counts_shots_in_binary_order() -> None:
    shots = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    original = shots.copy()

    histogram = get_histogram(shots)

    # Bins run from 000 to 111; the first column is the most significant bit.
    np.testing.assert_array_equal(histogram, [1, 1, 0, 0, 2, 0, 0, 1])
    np.testing.assert_array_equal(shots, original)


def test_get_histogram_accepts_boolean_shots() -> None:
    shots = np.array([[False, True], [True, False], [False, True]])

    np.testing.assert_array_equal(get_histogram(shots), [0, 2, 1, 0])


def test_get_histogram_includes_all_bins_for_empty_shots() -> None:
    shots = np.empty((0, 3), dtype=bool)

    np.testing.assert_array_equal(get_histogram(shots), np.zeros(8, dtype=int))


def test_get_histogram_with_zero_bit_columns_counts_empty_outcomes() -> None:
    shots = np.empty((3, 0), dtype=bool)

    np.testing.assert_array_equal(get_histogram(shots), [3])
