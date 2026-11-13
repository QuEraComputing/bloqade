import sys
import textwrap
import subprocess


def test_gemini_decoding_public_imports():
    from bloqade.gemini import decoding

    assert hasattr(decoding, "GurobiDecoder")
    assert hasattr(decoding, "PostSelectionExperiment")
    assert hasattr(decoding, "TableDecoder")
    assert hasattr(decoding, "TomographyResult")
    assert hasattr(decoding, "magic_state_dist_steane")
    assert not hasattr(decoding, "DEFAULT_TARGET_BLOCH")
    assert not hasattr(decoding, "plot_decoder_curves")


def test_table_decoder_is_upstream_class():
    from bloqade.decoders import TableDecoder as UpstreamTableDecoder
    from bloqade.gemini.decoding import TableDecoder
    from bloqade.gemini.decoding.table_decoders import (
        TableDecoder as ModuleTableDecoder,
    )

    assert TableDecoder is UpstreamTableDecoder
    assert ModuleTableDecoder is UpstreamTableDecoder


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
        import stim
        import bloqade.gemini
        from bloqade.gemini.decoding import GurobiDecoder
        assert not any(name in sys.modules for name in ('pulp', 'gurobipy', 'highspy'))
        try:
            GurobiDecoder(stim.DetectorErrorModel('error(0.1) D0 L0'))
        except ImportError as exc:
            assert 'pulp' in str(exc)
        else:
            raise AssertionError('expected the optional-dependency error')
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_tomography_is_upstream_class():
    from bloqade.analysis.tomography import TomographyResult as UpstreamTomographyResult
    from bloqade.gemini.decoding import TomographyResult
    from bloqade.gemini.decoding.tomography import (
        TomographyResult as ModuleTomographyResult,
    )

    assert TomographyResult is UpstreamTomographyResult
    assert ModuleTomographyResult is UpstreamTomographyResult
