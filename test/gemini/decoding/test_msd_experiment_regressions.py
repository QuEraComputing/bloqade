from __future__ import annotations

import stim
import numpy as np
import pytest
from bloqade.decoders import BaseDecoder, TableDecoder

from bloqade.gemini.decoding import experiments as experiments_module
from bloqade.gemini.decoding.sampling import _BasisDataset
from bloqade.gemini.decoding.tomography import TomographyResult
from bloqade.gemini.decoding.experiments import (
    PostSelectionExperiment,
    _PostSelectionExperimentCache,
)
from bloqade.gemini.decoding.postselection import (
    PostselectionCurveData,
    _DecodedPostselectionResult,
    _shots_at_accepted_fraction,
    _build_generic_threshold_tables,
    _evaluate_cached_threshold_curve,
)


def _dataset() -> _BasisDataset:
    return _BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        observables=np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        ),
    )


class _FactoryDecoder(BaseDecoder):
    def __init__(self) -> None:
        super().__init__(stim.DetectorErrorModel())

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return self._decode_confidence(detector_bits)[0]

    def _decode_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        factory_bit = int(detector_bits[0]) if len(detector_bits) else 0
        return (
            (
                np.array([factory_bit], dtype=np.bool_)
                if len(detector_bits)
                else np.zeros(0, dtype=np.bool_)
            ),
            float(0.9 if factory_bit == 0 else 0.1),
        )


class _FullDecoder(BaseDecoder):
    def __init__(self) -> None:
        pass

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        _ = detector_bits
        return np.array([0], dtype=np.bool_)


def _decoder_pair() -> tuple[BaseDecoder, BaseDecoder]:
    return _FactoryDecoder(), _FullDecoder()


def test_build_generic_threshold_tables_returns_decoded_shots_with_confidence():
    data = {basis: _dataset() for basis in ("X", "Y", "Z")}
    decoders = {basis: _decoder_pair() for basis in ("X", "Y", "Z")}

    decoded = _build_generic_threshold_tables(
        data,
        decoders,
        targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
    )

    assert decoded["X"].observables.shape == (3, 1)
    assert decoded["X"].confidence.shape == (3,)
    np.testing.assert_array_equal(decoded["X"].observables[:, 0], np.array([0, 1, 0]))
    np.testing.assert_allclose(decoded["X"].confidence, np.array([0.9, 0.9, 0.1]))


def test_build_generic_threshold_tables_supports_empty_factory_postselection():
    dataset = _BasisDataset(
        detectors=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        observables=np.array([[0], [1], [0]], dtype=np.uint8),
    )

    data = {basis: dataset for basis in ("X", "Y", "Z")}
    decoders = {
        basis: (TableDecoder(stim.DetectorErrorModel(), num_shots=0), _FullDecoder())
        for basis in ("X", "Y", "Z")
    }

    decoded = _build_generic_threshold_tables(
        data,
        decoders,
        targets=np.zeros((1, 0), dtype=np.uint8),
        basis_labels=("X", "Y", "Z"),
    )

    assert decoded["X"].observables.shape == (3, 1)
    np.testing.assert_array_equal(decoded["X"].observables[:, 0], np.array([0, 1, 0]))

    np.testing.assert_array_equal(decoded["X"].confidence, np.ones(3))


class _BatchFactoryDecoder(_FactoryDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.decode_confidence_calls = 0
        self.decoded_rows: list[int] = []

    def _decode_confidence(
        self,
        detector_bits: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        self.decode_confidence_calls += 1
        self.decoded_rows.append(detector_bits.shape[0])
        return detector_bits[:1].astype(np.bool_), 1.0


class _BatchFullDecoder(BaseDecoder):
    def __init__(self) -> None:
        self.decode_calls = 0
        self.decoded_rows: list[int] = []

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        _ = detector_bits
        return np.array([0], dtype=np.bool_)

    def decode(self, detector_bits: np.ndarray) -> np.ndarray:
        self.decode_calls += 1
        self.decoded_rows.append(detector_bits.shape[0])
        return np.zeros((detector_bits.shape[0], 1), dtype=np.bool_)


def test_build_generic_threshold_tables_batch_decodes_unique_detector_patterns():
    data = {basis: _dataset() for basis in ("X", "Y", "Z")}
    factory = _BatchFactoryDecoder()
    full = _BatchFullDecoder()
    decoders = {basis: (factory, full) for basis in ("X", "Y", "Z")}

    decoded = _build_generic_threshold_tables(
        data,
        decoders,
        targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X",),
    )

    assert decoded["X"].observables.shape == (3, 1)
    assert factory.decode_confidence_calls == 2
    assert factory.decoded_rows == [1, 1]
    assert full.decode_calls == 1
    assert full.decoded_rows == [2]


def test_build_generic_threshold_tables_progress_label_true_uses_decoder_name(
    monkeypatch,
):
    import tqdm.auto

    progress_descriptions: list[str] = []

    class _FakeProgressBar:
        def __init__(self, **kwargs):
            progress_descriptions.append(str(kwargs["desc"]))

        def update(self, value):
            _ = value

        def close(self):
            pass

    monkeypatch.setattr(tqdm.auto, "tqdm", _FakeProgressBar)

    _build_generic_threshold_tables(
        {"X": _dataset()},
        {"X": _decoder_pair()},
        targets=np.array([[0]], dtype=np.uint8),
        basis_labels=("X",),
        progress_label=True,
    )

    assert progress_descriptions == ["_FactoryDecoder X: decoded"]


def test_evaluate_cached_threshold_curve_returns_point_estimate_only():
    decoded = {
        basis: _DecodedPostselectionResult(
            observables=np.array([[0], [0], [1]], dtype=np.uint8),
            confidence=np.array([0.9, 0.8, 0.1], dtype=np.float64),
        )
        for basis in ("X", "Y", "Z")
    }

    curve = _evaluate_cached_threshold_curve(
        decoded,
        threshold_points=3,
        target_bloch=np.ones(3) / np.sqrt(3.0),
        basis_labels=("X", "Y", "Z"),
        min_accepted_per_basis=1,
        total_shots=30,
    )

    assert isinstance(curve, PostselectionCurveData)
    assert np.all(curve.fidelity == curve.point_fidelity)


def test_postselection_experiment_decode_and_tomography_result_with_cached_data():
    exp = object.__new__(PostSelectionExperiment)
    exp._postselection_exp_cache = _PostSelectionExperimentCache()
    exp._postselection_exp_cache.raw_results = {
        basis: _dataset() for basis in ("X", "Y", "Z")
    }
    exp._postselection_exp_cache.decoders_with_confidence = {
        basis: _decoder_pair() for basis in ("X", "Y", "Z")
    }
    exp._postselection_exp_cache.decoded_results = None

    decoded = exp.decode_and_postselect(
        np.array([[0]], dtype=np.uint8),
        progress_label=False,
    )
    assert decoded["X"].observables.shape == (3, 1)

    result = exp.tomography_result(accepted_fraction=0.5)
    assert isinstance(result, TomographyResult)
    assert result.density_matrix.shape == (2, 2)


def test_postselection_experiment_decode_progress_label_true_uses_decoder_class_name(
    monkeypatch,
):
    exp = object.__new__(PostSelectionExperiment)
    exp.decoder = _FactoryDecoder
    exp._postselection_exp_cache = _PostSelectionExperimentCache()
    exp._postselection_exp_cache.raw_results = {
        basis: _dataset() for basis in ("X", "Y", "Z")
    }
    exp._postselection_exp_cache.decoders_with_confidence = {
        basis: _decoder_pair() for basis in ("X", "Y", "Z")
    }

    captured_progress_labels: list[str | bool] = []

    def build_tables(*args, **kwargs):
        _ = args
        captured_progress_labels.append(kwargs["progress_label"])
        return {
            basis: _DecodedPostselectionResult(
                observables=np.zeros((0, 1), dtype=np.uint8),
                confidence=np.zeros(0, dtype=np.float64),
            )
            for basis in ("X", "Y", "Z")
        }

    monkeypatch.setattr(
        experiments_module,
        "_build_generic_threshold_tables",
        build_tables,
    )

    exp.decode_and_postselect(np.array([[0]], dtype=np.uint8), progress_label=True)

    assert captured_progress_labels == ["_FactoryDecoder"]


def test_shots_at_accepted_fraction_is_relative_to_postselected_shots():
    decoded = {
        basis: _DecodedPostselectionResult(
            observables=np.array([[0], [1], [0]], dtype=np.uint8),
            confidence=np.array([0.9, 0.8, 0.1], dtype=np.float64),
        )
        for basis in ("X", "Y", "Z")
    }

    accepted = _shots_at_accepted_fraction(
        decoded,
        1 / 3,
        basis_labels=("X", "Y", "Z"),
    )

    assert sum(shots.shape[0] for shots in accepted.values()) == 3


@pytest.mark.parametrize("missing_before_cut", [False, True])
def test_tomography_reports_missing_basis(missing_before_cut):
    exp = object.__new__(PostSelectionExperiment)
    exp._postselection_exp_cache = _PostSelectionExperimentCache()
    exp._postselection_exp_cache.decoded_results = {
        basis: _DecodedPostselectionResult(
            observables=np.zeros(
                (0 if missing_before_cut and basis == "Z" else 4, 1), dtype=np.uint8
            ),
            confidence=np.full(
                0 if missing_before_cut and basis == "Z" else 4,
                0.1 if basis == "Z" else 0.9,
            ),
        )
        for basis in ("X", "Y", "Z")
    }
    advice = (
        "Collect more shots" if missing_before_cut else "Increase accepted_fraction"
    )
    with pytest.raises(ValueError, match=advice):
        exp.tomography_result(0.5)
    if not missing_before_cut:
        result = exp.tomography_result(1.0)
        assert all(shots.shape == (4,) for shots in result.shots_by_basis.values())


@pytest.mark.parametrize("missing_basis", [False, True])
def test_threshold_curve_requires_basis_coverage_even_with_zero_minimum(missing_basis):
    decoded = {
        basis: _DecodedPostselectionResult(
            observables=np.array([[0], [1]], dtype=np.uint8),
            confidence=np.array([0.8 if basis == "Z" else 0.9, 0.1]),
        )
        for basis in ("X", "Y", "Z")
    }
    if missing_basis:
        decoded["Z"] = _DecodedPostselectionResult(
            observables=np.zeros((0, 1), dtype=np.uint8), confidence=np.zeros(0)
        )
    curve = _evaluate_cached_threshold_curve(
        decoded, total_shots=6, threshold_points=6, min_accepted_per_basis=0
    )
    np.testing.assert_allclose(
        curve.accepted_fraction, [] if missing_basis else [0.5, 1.0]
    )
    assert np.all(np.isfinite(curve.fidelity))
