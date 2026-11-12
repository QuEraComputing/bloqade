from __future__ import annotations

import stim
import numpy as np
import pytest
from bloqade.decoders import BaseDecoder
from bloqade.decoders._decoders.mld.utils import shots_to_counts, pack_boolean_array

from bloqade.gemini.decoding.dem import _sub_detector_error_model
from bloqade.gemini.decoding.confidence import (
    GurobiDecoder,
)
from bloqade.gemini.decoding.tomography import TomographyResult
from bloqade.gemini.decoding.table_decoders import TableDecoder


def test_tomography_result_builds_density_matrix_and_point_fidelity():
    result = TomographyResult(
        {
            "X": np.array([0, 0], dtype=np.uint8),
            "Y": np.array([0, 1], dtype=np.uint8),
            "Z": np.array([0, 1], dtype=np.uint8),
        }
    )

    assert result.density_matrix.shape == (2, 2)
    assert result.fidelity_bloch(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_tomography_result_rejects_non_physical_target_bloch_vector():
    result = TomographyResult(
        {
            "X": np.array([0, 1], dtype=np.uint8),
            "Y": np.array([0, 1], dtype=np.uint8),
            "Z": np.array([0, 1], dtype=np.uint8),
        }
    )

    with pytest.raises(ValueError, match="squared norm <= 1"):
        result.fidelity_bloch(np.array([2.0, 0.0, 0.0]))


def test_bit_packing_helpers_pack_little_endian_bits():
    bits = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

    packed = pack_boolean_array(bits)

    assert packed.tolist() == [0b101, 0b110]


def test_bit_packing_helpers_support_zero_width_rows():
    bits = np.zeros((3, 0), dtype=np.uint8)

    packed = pack_boolean_array(bits)

    np.testing.assert_array_equal(packed, np.zeros(3, dtype=np.uint64))


def test_sub_detector_error_model_composes_duplicate_projected_errors():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0 L0
        error(0.3) D1
        """)

    projected = _sub_detector_error_model(
        dem,
        detector_indices=[0],
        observable_indices=[0],
    )
    error_instructions = [
        instruction
        for instruction in projected.flattened()
        if isinstance(instruction, stim.DemInstruction) and instruction.type == "error"
    ]

    assert len(error_instructions) == 1
    assert error_instructions[0].args_copy()[0] == pytest.approx(0.26)
    assert [str(target) for target in error_instructions[0].targets_copy()] == [
        "D0",
        "L0",
    ]


def test_table_decoder_decodes_and_scores_detector_patterns():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    shots = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )
    decoder = TableDecoder(dem, num_shots=0)
    decoder.update_det_obs_counts(shots)

    correction0, confidence0 = decoder.decode_confidence(np.array([0], dtype=np.bool_))
    correction1, confidence1 = decoder.decode_confidence(np.array([1], dtype=np.bool_))

    assert isinstance(decoder, BaseDecoder)
    np.testing.assert_array_equal(correction0, np.array([False]))
    np.testing.assert_array_equal(correction1, np.array([True]))
    assert confidence0 == pytest.approx(2.0 / 3.0)
    assert confidence1 == pytest.approx(2.0 / 3.0)

    corrections, confidence = decoder.decode_confidence(
        np.array([[0], [1]], dtype=np.bool_)
    )
    np.testing.assert_array_equal(corrections, [[False], [True]])
    np.testing.assert_allclose(confidence, [2 / 3, 2 / 3])

    corrections, confidence = decoder.decode_confidence(
        np.empty((0, 1), dtype=np.bool_)
    )
    assert corrections.shape == (0, 1)
    assert np.asarray(confidence).shape == (0,)

    # Training through the BaseDecoder lifecycle must invalidate cached results.
    decoder.train(num_shots=0)
    decoder.update_det_obs_counts(np.array([[0, 1]], dtype=np.uint8))
    correction, confidence = decoder.decode_confidence(np.array([0], dtype=np.bool_))
    np.testing.assert_array_equal(correction, [True])
    assert confidence == 1.0


def test_table_decoder_reports_zero_confidence_for_unseen_syndromes():
    decoder = TableDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"), num_shots=0)
    correction, confidence = decoder.decode_confidence(np.array([True]))
    np.testing.assert_array_equal(correction, [False])
    assert confidence == 0.0


@pytest.mark.parametrize("untrained", [False, True])
def test_gurobi_decoder_reports_normalized_logical_gap(untrained: bool):
    pytest.importorskip("gurobipy")
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.01) D0 L0
        """)
    decoder = GurobiDecoder.instantiate(dem) if untrained else GurobiDecoder(dem)
    assert decoder._solver_name == "GUROBI"

    correction, logical_gap = decoder.decode_confidence(np.array([1], dtype=np.bool_))

    np.testing.assert_array_equal(correction, np.array([False]))
    expected_gap = np.log(0.1 / 0.9) - np.log(0.01 / 0.99)
    assert logical_gap == pytest.approx(np.tanh(expected_gap / 2))

    corrections, confidences = decoder.decode_confidence(
        np.ones((2, 1), dtype=np.bool_)
    )
    np.testing.assert_array_equal(corrections, [[False], [False]])
    np.testing.assert_allclose(confidences, np.tanh(expected_gap / 2))

    corrections, confidences = decoder.decode_confidence(
        np.empty((0, 1), dtype=np.bool_)
    )
    assert corrections.shape == (0, 1)
    assert np.asarray(confidences).shape == (0,)


@pytest.mark.parametrize(
    "detector_bits",
    [
        np.zeros(0, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
    ],
    ids=["too_short", "too_long"],
)
def test_gurobi_decoder_validates_decode_confidence_width(
    detector_bits: np.ndarray,
):
    pytest.importorskip("gurobipy")
    decoder = GurobiDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"))

    with pytest.raises(ValueError, match="detector bits"):
        decoder.decode_confidence(detector_bits)


@pytest.mark.parametrize(
    "detector_bits",
    [
        np.zeros(0, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        np.zeros((2, 0), dtype=np.bool_),
        np.zeros((2, 2), dtype=np.bool_),
    ],
    ids=["single_too_short", "single_too_long", "batch_too_short", "batch_too_long"],
)
def test_gurobi_decoder_validates_decode_width(
    detector_bits: np.ndarray,
):
    pytest.importorskip("gurobipy")
    decoder = GurobiDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"))

    with pytest.raises(ValueError, match="detector bits"):
        decoder.decode(detector_bits)


def test_shots_to_counts_uses_little_endian_packing():
    shots = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=np.uint8)

    counts = shots_to_counts(shots)

    np.testing.assert_array_equal(counts, np.array([1, 2, 1, 0]))
