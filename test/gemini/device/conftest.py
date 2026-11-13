"""Shared assertion helpers for the Gemini device test suite.

``bloqade.core.device`` threads its own kwargs into every qlam client call
(``qpu_mode`` as of bloqade-core 0.6.12, and more over time). Asserting the
full call signature with ``assert_called_once_with`` therefore couples these
tests to upstream's argument list, and every addition breaks them in bulk
without any Gemini behaviour having changed.

These tests only care that the Gemini wrapper routed the right identifier to
the right client, so match on that subset and let upstream pass whatever else
it likes alongside.
"""

import pytest


@pytest.fixture
def assert_called_with_kwargs():
    """Assert a mocked client method was called with (at least) these kwargs.

    Checks the most recent call, mirroring ``assert_called_with``. Pass
    ``times=`` to also pin the call count, mirroring ``assert_called_once_with``
    when set to 1.
    """

    def _assert(mock, *, times: int | None = None, **expected):
        assert mock.call_count > 0, "expected call not found; mock was not called"
        if times is not None:
            assert mock.call_count == times, (
                f"expected {times} call(s), got {mock.call_count}: "
                f"{mock.call_args_list}"
            )
        actual = mock.call_args.kwargs
        for key, value in expected.items():
            assert key in actual, f"{key!r} missing from call kwargs {actual}"
            assert (
                actual[key] == value
            ), f"{key}: expected {value!r}, got {actual[key]!r}"

    return _assert
