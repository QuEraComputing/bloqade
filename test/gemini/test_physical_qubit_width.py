"""The physical-qubits-per-logical width is fixed, not a decorator option.

Regression test for bloqade-internal#483: ``@logical.kernel`` used to take a
``num_physical_qubits`` argument, but nothing downstream honours a width other
than Steane [[7,1,3]]'s seven. A too-small value blew up inside the stdlib
post-processing kernels (which index measurements 0..6 directly) as an opaque
"tuple index out of range" validation failure, and a too-large one silently
built a program whose extra records no detector ever read.
"""

import pytest
from bloqade.squin import qubit
from bloqade.lanes.dialects.steane713.stmts import TerminalLogicalMeasurement

from bloqade.gemini import logical
from bloqade.gemini.steane_defaults import STEANE7_PHYSICAL_QUBITS


def test_terminal_measurement_is_stamped_with_the_steane_width():
    @logical.kernel(aggressive_unroll=True)
    def main():
        register = qubit.qalloc(2)
        return logical.default_post_processing(register)

    widths = [
        stmt.num_physical_qubits
        for stmt in main.callable_region.walk()
        if isinstance(stmt, TerminalLogicalMeasurement)
    ]

    assert widths == [STEANE7_PHYSICAL_QUBITS]


@pytest.mark.parametrize("num_physical_qubits", [6, 7, 8])
def test_num_physical_qubits_is_rejected(num_physical_qubits):
    with pytest.raises(TypeError, match="num_physical_qubits"):
        # The point of the test is that this argument no longer exists, so
        # pyright flagging the call is the statically-checkable half of it.
        @logical.kernel(
            aggressive_unroll=True, num_physical_qubits=num_physical_qubits
        )  # pyright: ignore[reportCallIssue]
        def main():
            register = qubit.qalloc(2)
            return logical.default_post_processing(register)
