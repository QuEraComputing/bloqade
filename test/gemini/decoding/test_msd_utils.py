from __future__ import annotations

import numpy as np
import matplotlib
from bloqade.decoders import BaseDecoder, TableDecoder
from bloqade.analysis.tomography import TomographyResult

from bloqade.gemini.logical.utils import plot_decoder_curves
from bloqade.gemini.logical.kernels import (
    PostselectionExperimentKernels,
    build_singleq_tomo,
    compose_tomography_kernels,
    build_empty_logical_circuit,
)
from bloqade.gemini.logical.experiments import PostselectionCurveData

matplotlib.use("Agg")


def test_public_facade_exports_simplified_decoder_and_tomography_types():
    assert issubclass(TableDecoder, BaseDecoder)
    result = TomographyResult(
        {
            "X": np.array([0, 0], dtype=np.uint8),
            "Y": np.array([0, 1], dtype=np.uint8),
            "Z": np.array([0, 1], dtype=np.uint8),
        }
    )
    assert result.density_matrix.shape == (2, 2)


def test_build_decoder_kernel_bundle_contains_basis_tomography_kernels():
    tomography_kernels = build_singleq_tomo()
    kernels = compose_tomography_kernels(
        PostselectionExperimentKernels(
            state_injection_circuit=build_empty_logical_circuit(),
            logical_circuit=build_empty_logical_circuit(),
        ),
        num_logical_qubits=1,
        tomography_kernels=tomography_kernels,
    )

    assert set(kernels) == {"X", "Y", "Z"}


def test_plot_decoder_curves_handles_curves_without_uncertainty_bands():
    fig, ax = plot_decoder_curves(
        {
            "decoder": PostselectionCurveData(
                accepted_fraction=np.array([0.05, 0.1]),
                fidelity=np.array([0.99, 0.95]),
                point_fidelity=np.array([0.99, 0.95]),
            )
        },
        injected_summary=0.96,
        log=False,
    )

    assert fig is ax.figure
    assert len(ax.lines) == 2
    assert len(ax.collections) == 0
