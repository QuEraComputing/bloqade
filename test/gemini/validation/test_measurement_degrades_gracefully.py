"""The frontend aggregates validation diagnostics without hiding them in crashes."""

import pytest
import bloqade.squin as squin
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

import bloqade.gemini as gemini


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
