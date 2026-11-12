import pytest
import bloqade.squin as squin
from bloqade.types import Qubit
from kirin.dialects import scf, ilist
from kirin.validation import ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from bloqade.analysis.address import AddressAnalysis
from bloqade.lanes.validation.steane import (
    SteaneLogicalValidation,
    _SteaneLogicalValidationAnalysis,
)

import bloqade.gemini as gemini


def test_for_loop():
    @gemini.logical.kernel
    def valid_loop():
        q = squin.qalloc(3)

        for i in range(3):
            squin.x(q[i])

        gemini.logical.terminal_measure(q)

    valid_loop.print()
    assert not any(
        isinstance(stmt, scf.For) for stmt in valid_loop.callable_region.walk()
    )

    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel
        def invalid_loop(n: int):
            q = squin.qalloc(3)

            for i in range(n):
                squin.x(q[i])


def test_aggressive_unroll_can_be_disabled():
    @gemini.logical.kernel(aggressive_unroll=False, verify=False)
    def retained_loop():
        q = squin.qalloc(3)
        for i in range(3):
            squin.x(q[i])
        gemini.logical.terminal_measure(q)

    assert any(
        isinstance(stmt, scf.For) for stmt in retained_loop.callable_region.walk()
    )


def test_func():
    @gemini.logical.kernel(verify=False)
    def sub_kernel(q: Qubit):
        squin.x(q)

    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(3)
        sub_kernel(q[0])
        gemini.logical.terminal_measure(q)

    main.print()

    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(inline=False)
        def invalid():
            q = squin.qalloc(3)
            sub_kernel(q[0])


def test_clifford_gates():
    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        squin.u3(0.123, 0.253, 1.2, q[0])

        squin.h(q[0])
        squin.cx(q[0], q[1])
        gemini.logical.terminal_measure(q)

    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(no_raise=False)
        def invalid():
            q = squin.qalloc(2)

            squin.h(q[0])
            squin.cx(q[0], q[1])
            squin.u3(0.123, 0.253, 1.2, q[0])

        invalid.print(hint="address")

        addr_frame, _ = AddressAnalysis(invalid.dialects).run(invalid)
        frame, _ = _SteaneLogicalValidationAnalysis(
            invalid.dialects, addr_frame=addr_frame
        ).run_no_raise(invalid)

        invalid.print(analysis=frame.entries)


def test_swap_is_clifford():
    @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
    def valid():
        q = squin.qalloc(2)
        squin.h(q[0])
        squin.swap(q[0], q[1])
        gemini.logical.terminal_measure(q)

    validator = ValidationSuite([SteaneLogicalValidation])
    validator.validate(valid).raise_if_invalid()

    # NOTE: swap marks both operands as acted on, so a following non-clifford
    # gate on either of them is no longer state preparation. The kernel below is
    # otherwise valid -- terminal_measure included -- so the u3 rejection is the
    # only thing that can fail it.
    try:

        @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
        def invalid():
            q = squin.qalloc(2)
            squin.swap(q[0], q[1])
            squin.u3(0.1, 0.2, 0.3, q[1])
            gemini.logical.terminal_measure(q)

    except ValidationErrorGroup as e:
        messages = [str(err) for err in e.errors]
        assert len(messages) == 1, messages
        assert "Non-clifford gate u3" in messages[0], messages
    else:
        pytest.fail("expected a ValidationErrorGroup")


def test_star_rz_marks_qubit_as_acted_on_without_rejecting_mid_circuit_use():
    @gemini.logical.kernel(aggressive_unroll=True)
    def valid():
        q = squin.qalloc(1)
        squin.h(q[0])
        gemini.logical.extensions.star_rz(0.125, q[0])
        gemini.logical.terminal_measure(q)

    validator = ValidationSuite([SteaneLogicalValidation])
    validator.validate(valid).raise_if_invalid()

    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
        def invalid():
            q = squin.qalloc(1)
            gemini.logical.extensions.star_rz(0.125, q[0])
            squin.u3(0.1, 0.2, 0.3, q[0])


def test_multiple_errors():
    did_error = False

    try:

        @gemini.logical.kernel
        def main(n: int):
            q = squin.qalloc(3)
            m = squin.qubit.measure(q[0])
            squin.x(q[1])
            if m:
                squin.x(q[0])

            for k in range(n):
                squin.h(q[k])

            squin.u3(0.1, 0.2, 0.3, q[1])

    except ValidationErrorGroup as e:
        did_error = True
        # Six, not five: `main` allocates its own qubits, so it is a program,
        # and a program taking `n` is reported alongside the rest rather than
        # only after the others are fixed.
        assert len(e.errors) == 6
        assert any("must take no arguments" in m for m in map(str, e.errors))

    assert did_error


def test_unsupported_gate_is_a_validation_error():
    # NOTE: an unhandled gate must be reported alongside the other errors rather
    # than aborting the analysis, so the u3 error below has to show up too. The
    # kernel is otherwise valid, so these are the only two errors expected.
    try:

        @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
        def main():
            q = squin.qalloc(3)
            squin.ccz(q[0], q[1], q[2])
            squin.h(q[0])
            squin.u3(0.1, 0.2, 0.3, q[0])
            gemini.logical.terminal_measure(q)

    except ValidationErrorGroup as e:
        messages = [str(err) for err in e.errors]
        assert len(messages) == 2, messages
        assert any("Gate ccz is not supported" in m for m in messages), messages
        assert any("Non-clifford gate u3" in m for m in messages), messages
        # Both come from inlined stdlib kernels, so the call site is named; but
        # the kernel already unrolls and neither error is about unrolling, so the
        # unroll advice must not be tacked on.
        assert all("inlined from" in m for m in messages), messages
        assert not any("aggressive_unroll" in m for m in messages), messages
    else:
        pytest.fail("expected a ValidationErrorGroup")


def test_non_clifford_parallel_gates():
    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(5)
        squin.rx(0.123, q[0])
        squin.broadcast.ry(0.333, q[1:])

        squin.broadcast.x(q)
        squin.broadcast.h(q[1:])
        gemini.logical.terminal_measure(q)

    main.print()


def test_nocloning():
    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel
        def main():
            q = squin.qalloc(2)
            squin.broadcast.ry(0.123, ilist.IList([q[0], q[0]]))
            squin.cx(q[1], q[1])


def test_allocation():
    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(aggressive_unroll=True)
        def main():
            reg = squin.qalloc(12)
            squin.h(reg[0])
            squin.cx(reg[0], reg[1])


MAX_QUBITS = _SteaneLogicalValidationAnalysis.max_qubits


def _error_messages(err: ValidationErrorGroup) -> list[str]:
    return [e.args[0] if e.args else str(e) for e in err.errors]


def test_allocation_exceeded_without_unroll():
    # NOTE: the per-`qubit.New` check only fires once a register has been
    # expanded into individual qubits, so without unrolling the total
    # allocation is the only thing that can catch this.
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel
        def main():
            reg = squin.qalloc(MAX_QUBITS + 1)
            squin.h(reg[0])

    assert any(
        f"kernel allocates {MAX_QUBITS + 1} qubits, "
        f"exceeding the maximum of {MAX_QUBITS}" in message
        for message in _error_messages(exc_info.value)
    )


def test_allocation_exceeded_summed_across_registers():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel
        def main():
            first = squin.qalloc(MAX_QUBITS)
            second = squin.qalloc(1)
            squin.h(first[0])
            squin.h(second[0])

    assert any(
        f"kernel allocates {MAX_QUBITS + 1} qubits, "
        f"exceeding the maximum of {MAX_QUBITS}" in message
        for message in _error_messages(exc_info.value)
    )


def test_allocation_at_limit_is_valid():
    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        reg = squin.qalloc(MAX_QUBITS)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    validator = ValidationSuite([SteaneLogicalValidation])
    validator.validate(main).raise_if_invalid()


NO_QUBITS = "kernel allocates no qubits; at least 1 is required"


def test_empty_kernel_is_invalid():
    """bloqade-internal#482: an empty kernel used to pass validation."""
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=True)
        def main() -> None:
            return

    assert NO_QUBITS in _error_messages(exc_info.value)


def test_zero_width_allocation_is_invalid():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel
        def main():
            squin.qalloc(0)

    assert NO_QUBITS in _error_messages(exc_info.value)


def test_sub_kernel_that_does_not_allocate_is_invalid():
    """The check does not special-case sub-kernels; they opt out explicitly."""
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel
        def flip(q: Qubit):
            squin.x(q)

    assert NO_QUBITS in _error_messages(exc_info.value)


def test_sub_kernel_opts_out_with_verify_false():
    @gemini.logical.kernel(verify=False)
    def flip(q: Qubit):
        squin.x(q)

    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        flip(q[0])
        gemini.logical.terminal_measure(q)


def test_gate_after_terminal_measurement_is_invalid():
    """bloqade-internal#479: a gate after the terminal measurement was accepted."""
    with pytest.raises(ValidationErrorGroup, match="Gate h is applied after"):

        @gemini.logical.kernel(aggressive_unroll=True, verify=True)
        def gate_after_measurement() -> tuple:
            register = squin.qalloc(2)
            result = gemini.logical.default_post_processing(register)
            squin.h(register[0])
            return result
