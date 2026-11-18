import sys
import math
import textwrap
import importlib
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("module", "names"),
    [
        (
            "bloqade.gemini.logical.kernels",
            [
                "PostselectionExperimentKernels",
                "build_bell_state_kernels",
                "build_concatenated_msd_kernels",
                "build_empty_logical_circuit",
                "build_logical_toffoli",
                "build_msd_kernels",
                "build_msd_steane",
                "build_singleq_tomo",
                "compose_tomography_kernels",
                "build_tomography_kernels",
            ],
        ),
        (
            "bloqade.gemini.logical.utils",
            [
                "MSD_TARGET_BLOCH",
                "bloch_to_u3",
                "msd_postselection_obs",
                "plot_decoder_curves",
            ],
        ),
        (
            "bloqade.gemini.logical.experiments",
            ["PostSelectionExperiment", "PostselectionCurveData"],
        ),
        (
            "bloqade.gemini.logical.simulation",
            [
                "DetectorResult",
                "GeminiLogicalSimulator",
                "GeminiLogicalSimulatorTask",
                "Result",
                "SimulatorResult",
            ],
        ),
        (
            "bloqade.gemini.logical.device",
            ["GeminiLogicalDevice", "GeminiLogicalFuture", "GeminiLogicalResult"],
        ),
        (
            "bloqade.gemini.physical.simulation",
            [
                "GeminiPhysicalSimulator",
                "PhysicalResult",
                "PhysicalSimulator",
                "PhysicalSimulatorTask",
            ],
        ),
        (
            "bloqade.simulation",
            [
                "AbstractSimulatorBackend",
                "BackendSample",
                "CliffTSimulatorBackend",
                "PPVMSimulatorBackend",
                "TsimSimulatorBackend",
            ],
        ),
    ],
)
def test_public_module_exports(module, names):
    mod = importlib.import_module(module)

    assert sorted(mod.__all__) == sorted(names)
    for name in names:
        assert hasattr(mod, name), name


@pytest.mark.parametrize(
    "module",
    [
        "bloqade.gemini.decoding",
        "bloqade.gemini.device",
        "bloqade.gemini.logical.algorithms",
    ],
)
def test_removed_packages(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_top_level_reexports_point_to_new_modules():
    import bloqade.gemini as gemini
    from bloqade import simulation
    from bloqade.gemini.logical import device, simulation as logical_simulation
    from bloqade.gemini.physical import simulation as physical_simulation

    for module in (simulation, device, logical_simulation, physical_simulation):
        for name in module.__all__:
            assert getattr(gemini, name) is getattr(module, name), name


def test_bloch_to_u3_recovers_the_msd_preparation_angles():
    from bloqade.gemini.logical.utils import MSD_TARGET_BLOCH, bloch_to_u3

    theta, phi, lam = bloch_to_u3(MSD_TARGET_BLOCH)

    assert theta == pytest.approx(math.acos(1 / math.sqrt(3)))
    assert phi == pytest.approx(0.25 * math.pi)
    assert lam == 0.0


@pytest.mark.parametrize(
    "bloch",
    [np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])],
)
def test_bloch_to_u3_round_trips_through_the_bloch_vector(bloch):
    from bloqade.gemini.logical.utils import bloch_to_u3

    theta, phi, _ = bloch_to_u3(bloch)
    recovered = np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )

    np.testing.assert_allclose(recovered, bloch, atol=1e-12)


def test_msd_postselection_obs_is_one_ancilla_pattern():
    from bloqade.gemini.logical.utils import msd_postselection_obs

    np.testing.assert_array_equal(msd_postselection_obs, [[1, 0, 1, 1]])
    assert msd_postselection_obs.dtype == np.uint8


def test_gemini_imports_without_milp_solver_dependencies():
    """Importing Gemini must not require optional MILP solver packages."""
    script = """
        import importlib.abc
        import sys
        class BlockSolvers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'pulp', 'gurobipy', 'highspy'}:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, BlockSolvers())
        import bloqade.gemini
        import bloqade.gemini.logical.experiments
        import bloqade.gemini.logical.kernels
        import bloqade.gemini.logical.utils
        assert not any(name in sys.modules for name in ('pulp', 'gurobipy', 'highspy'))
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
