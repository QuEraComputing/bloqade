import sys
import textwrap
import subprocess

OPTIONAL_BACKEND_IMPORT_CHECK = """
loaded_backends = sorted(
    name
    for name in sys.modules
    if name == "tsim"
    or name.startswith("tsim.")
    or name == "bloqade.tsim"
    or name.startswith("bloqade.tsim.")
    or name == "clifft"
    or name.startswith("clifft.")
    or name == "pyqrack"
    or name.startswith("pyqrack.")
    or name == "bloqade.pyqrack"
    or name.startswith("bloqade.pyqrack.")
)
if loaded_backends:
    raise SystemExit(f"unexpected optional backend imports: {loaded_backends}")
"""


def _run_import_guard(code: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code) + textwrap.dedent(OPTIONAL_BACKEND_IMPORT_CHECK),
        ],
        check=True,
    )


def test_logical_device_import_does_not_import_optional_backends():
    _run_import_guard(
        """
        import sys

        from bloqade.gemini import logical as gemini_logical
        from bloqade.gemini import GeminiLogicalDevice

        assert gemini_logical is not None
        assert GeminiLogicalDevice.__name__ == "GeminiLogicalDevice"
        """
    )


def test_logical_simulator_import_does_not_import_optional_backends():
    _run_import_guard(
        """
        import sys

        from bloqade.gemini import GeminiLogicalSimulator

        assert GeminiLogicalSimulator.__name__ == "GeminiLogicalSimulator"
        """
    )


def test_simulator_backend_imports_are_lazy():
    _run_import_guard(
        """
        import sys

        import bloqade.gemini as gemini
        import bloqade.gemini.device as device
        from bloqade.gemini import (
            AbstractSimulatorBackend,
            BackendSample,
            CliffTSimulatorBackend,
            TsimSimulatorBackend,
        )
        from bloqade.gemini.device.simulator_backend import _PyQrackSimulatorBackend

        assert AbstractSimulatorBackend.__name__ == "AbstractSimulatorBackend"
        assert BackendSample.__name__ == "BackendSample"
        assert TsimSimulatorBackend.__name__ == "TsimSimulatorBackend"
        assert CliffTSimulatorBackend.__name__ == "CliffTSimulatorBackend"
        assert _PyQrackSimulatorBackend.__name__ == "_PyQrackSimulatorBackend"
        assert not hasattr(gemini, "_PyQrackSimulatorBackend")
        assert not hasattr(device, "_PyQrackSimulatorBackend")
        """
    )


def test_simulator_runtime_does_not_import_test_modules():
    _run_import_guard(
        """
        import sys

        import bloqade.gemini.device

        loaded_test_modules = sorted(
            name
            for name in sys.modules
            if name == "tests" or name.startswith("tests.")
        )
        if loaded_test_modules:
            raise SystemExit(
                f"unexpected test-module imports: {loaded_test_modules}"
            )
        """
    )


def test_composed_backends_fail_before_sampling_with_specific_tsim_guidance():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from unittest.mock import MagicMock

        class BlockTsim(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "tsim" or fullname.startswith("tsim."):
                    raise ImportError("blocked Tsim import")
                if fullname == "bloqade.tsim" or fullname.startswith("bloqade.tsim."):
                    raise ImportError("blocked Tsim import")
                return None

        sys.meta_path.insert(0, BlockTsim())

        from bloqade.gemini import CliffTSimulatorBackend, GeminiLogicalSimulatorTask
        from bloqade.gemini.device.simulator_backend import _PyQrackSimulatorBackend

        for backend_type, label in (
            (CliffTSimulatorBackend, "CliffT"),
            (_PyQrackSimulatorBackend, "PyQrack"),
        ):
            backend = backend_type()
            backend._tsim_backend._detector_error_model = MagicMock(
                side_effect=ImportError("blocked Tsim import")
            )
            backend.sample = MagicMock(
                side_effect=AssertionError("sampling must not begin")
            )
            task = object.__new__(GeminiLogicalSimulatorTask)
            object.__setattr__(task, "physical_squin_kernel", "kernel")
            object.__setattr__(task, "_simulator_backend", backend)
            try:
                task.run(shots=1)
            except ImportError as exc:
                message = str(exc)
                assert label in message
                assert "bloqade[sim]" in message
            else:
                raise AssertionError("expected missing-Tsim failure")
            backend.sample.assert_not_called()
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)
