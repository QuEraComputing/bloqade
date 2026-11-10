"""Terminal-measurement validation reports; it does not crash.

Both paths below reach the same place: the measure-id analysis cannot say what
a ``terminal_measure`` produced, so validation should say *that* and move on.
Neither used to. ``ValidationSuite`` wraps a pass that raises in a bare
``except Exception`` and re-reports it as ``Validation pass '...' failed:``
followed by a Python traceback -- which reads as a compiler crash rather than as
"your kernel is not valid", and buries the other violations found alongside it.

The two crashes were:

- ``impl/measure_id.py`` asserted the register type was ``types.Generic``. An
  ordinary untyped parameter makes it ``AnyType``, so an ``AssertionError``
  escaped. The neighbouring guards in that impl already return ``AnyMeasureId``
  for every other unknowable case.
- ``impl/measurement.py`` looked the result up with ``Frame.get``, which raises
  on a missing key instead of returning None. The analysis has no entry when the
  statement sits inside a callee that was not inlined, because it was evaluated
  in a nested frame.

Neither needs `verify=True` to reach, so the fixtures here build with
`verify=False` and drive the suite directly.
"""

import pytest
import bloqade.squin as squin
from kirin.dialects import ilist
from kirin.validation import ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
)

import bloqade.gemini as gemini

NO_RESULTS = "Measurement ID Analysis failed to produce the necessary results"


def _run(method):
    return ValidationSuite([SteaneTerminalMeasurementValidation]).validate(method)


def _messages(result):
    # `args[0] if args else str(e)` matches `test_recursion_validation`: an error
    # built without a message would otherwise raise `IndexError` here and hide
    # whatever actually went wrong.
    return [
        str(e.args[0]) if e.args else str(e)
        for errors in result.errors.values()
        for e in errors
    ]


def test_untyped_register_parameter_is_reported_not_asserted():
    """`reg` is untyped, so its type is `AnyType` rather than `IList[_, Len]`."""

    @gemini.logical.kernel(verify=False)
    def subroutine(reg):
        return gemini.logical.terminal_measure(reg)

    messages = _messages(_run(subroutine))

    assert any(NO_RESULTS in m for m in messages), messages
    # The tell for the old behaviour: the suite catching a pass that raised.
    assert not any("Validation pass" in m and "failed" in m for m in messages)
    assert not any("AssertionError" in m for m in messages)


def test_uninlined_callee_measurement_is_reported_not_raised():
    """The measurement lives in a callee, so it is evaluated in a nested frame.

    `Frame.get` raises on the miss; the impl has to route that into its own
    "no usable result" branch.
    """

    @gemini.logical.kernel(verify=False)
    def subroutine(reg):
        return gemini.logical.terminal_measure(reg)

    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def main():
        qbs = ilist.IList([squin.qubit.new(), squin.qubit.new()])
        squin.cz(qbs[0], qbs[1])
        return subroutine(qbs)

    messages = _messages(_run(main))

    assert any(NO_RESULTS in m for m in messages), messages
    assert not any("Validation pass" in m and "failed" in m for m in messages)
    assert not any("InterpreterError" in m for m in messages)


def test_kernel_group_reports_every_violation_together():
    """A pass that raises takes the whole report down with it.

    `ValidationSuite` replaces a raising pass' output with a single
    `Validation pass '...' failed:` entry, so the violations it *had* found are
    lost along with the ones the other passes found. The assertion is on that
    shape rather than on any one message, so it does not pin the wording of
    whichever pass reports the unresolved calls.
    """
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(inline=False, aggressive_unroll=False)
        def main():
            qbs = ilist.IList([squin.qubit.new(), squin.qubit.new()])
            squin.cz(qbs[0], qbs[1])
            return gemini.logical.terminal_measure(qbs)

    messages = [str(e.args[0]) if e.args else str(e) for e in exc_info.value.errors]

    assert len(messages) > 1, messages
    assert not any("Validation pass" in m and "failed" in m for m in messages), messages
    assert not any("Traceback (most recent call last)" in m for m in messages), messages


def test_a_well_formed_kernel_still_validates():
    """The guards must not swallow the real analysis on a valid kernel."""

    @gemini.logical.kernel(verify=False)
    def main():
        qbs = ilist.IList([squin.qubit.new(), squin.qubit.new()])
        squin.cz(qbs[0], qbs[1])
        return gemini.logical.terminal_measure(qbs)

    result = _run(main)
    assert result.is_valid, _messages(result)
