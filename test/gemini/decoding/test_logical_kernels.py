import math

import pytest
from kirin.ir.exception import ValidationErrorGroup

from bloqade.gemini.logical.kernels import (
    PostselectionExperimentKernels,
    build_logical_toffoli,
    build_bell_state_kernels,
    compose_tomography_kernels,
    build_concatenated_msd_kernels,
)

_MSD_ANGLES = (math.acos(1 / math.sqrt(3)), 0.25 * math.pi, 0.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: build_bell_state_kernels(log=False),
        lambda: build_logical_toffoli(
            [0.1, 0.2, 0.3], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0], log=False
        ),
    ],
    ids=["bell", "toffoli"],
)
def test_ten_qubit_primitives_compose_into_tomography_kernels(build):
    primitives = build()

    assert isinstance(primitives, PostselectionExperimentKernels)
    kernels = compose_tomography_kernels(primitives, 10)
    assert sorted(kernels) == ["X", "Y", "Z"]


def test_concatenated_msd_builds_but_exceeds_the_logical_qubit_limit():
    primitives = build_concatenated_msd_kernels(*_MSD_ANGLES, log=False)

    assert isinstance(primitives, PostselectionExperimentKernels)
    with pytest.raises(ValidationErrorGroup):
        compose_tomography_kernels(primitives, 25)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (build_bell_state_kernels, "Building GHZ State primitives..."),
        (
            lambda: build_logical_toffoli([0.0] * 3, [0.0] * 3, [0.0] * 3),
            "Building Toffoli primitives...",
        ),
        (
            lambda: build_concatenated_msd_kernels(*_MSD_ANGLES),
            "Building concatenated MSD primitives...",
        ),
    ],
    ids=["bell", "toffoli", "concatenated_msd"],
)
def test_builders_log_progress_by_default(build, message, capsys):
    build()

    assert capsys.readouterr().out.strip() == message
