import sys
import textwrap
import subprocess


def test_gemini_decoding_public_imports():
    from bloqade.gemini import decoding

    assert hasattr(decoding, "GurobiDecoderWithConfidence")
    assert hasattr(decoding, "PostSelectionExperiment")
    assert hasattr(decoding, "TableDecoderWithConfidence")
    assert hasattr(decoding, "TomographyResult")
    assert hasattr(decoding, "magic_state_dist_steane")
    assert not hasattr(decoding, "DEFAULT_TARGET_BLOCH")
    assert not hasattr(decoding, "plot_decoder_curves")


def test_gemini_imports_without_table_decoder():
    """Missing optional TableDecoder is reported only when it is constructed."""

    script = """
        import bloqade.decoders

        del bloqade.decoders.TableDecoder

        import stim
        from bloqade.gemini.decoding.table_decoders import TableDecoderWithConfidence

        try:
            TableDecoderWithConfidence(stim.DetectorErrorModel(""), num_shots=0)
        except ImportError as exc:
            assert "bloqade[decoding]" in str(exc)
        else:
            raise AssertionError("expected the optional-dependency error")
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_gemini_imports_without_gurobi_decoder():
    """Missing optional GurobiDecoder is reported only when it is constructed.

    Regression test for #840: `bloqade-decoders` only exports `GurobiDecoder`
    with its `mle` extra installed, and an eager top-level import broke the
    whole `bloqade.gemini` / `bloqade.lanes` import chain for consumers without
    that extra.
    """

    script = """
        import bloqade.decoders

        del bloqade.decoders.GurobiDecoder

        from bloqade.gemini.decoding.confidence import GurobiDecoderWithConfidence

        try:
            GurobiDecoderWithConfidence()
        except ImportError as exc:
            assert "bloqade[decoding]" in str(exc)
        else:
            raise AssertionError("expected the optional-dependency error")
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
