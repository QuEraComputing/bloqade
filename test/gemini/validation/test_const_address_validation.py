"""Tests for ConstAddressValidation: NewAt addresses must const-fold.

The const half of NewAt validation needs no ArchSpec, so unlike the range check
it runs at kernel-decoration time. These tests pin that it fires there -- the
kernel groups used to lower a non-const address cleanly and only reject it much
later, in ``NativeToPlace.emit``.
"""

import pytest
import bloqade.squin as squin
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects.arch import new_at

import bloqade.gemini as gemini
from bloqade.gemini.physical import kernel as movement_kernel


def test_logical_kernel_decorator_rejects_non_const_address():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=True, verify=True)
        def kernel(word_id: int) -> tuple:
            register = ilist.IList([new_at(0, word_id, 0)])
            return gemini.logical.default_post_processing(register)

    errors = exc_info.value.errors
    assert any("compile-time constant" in str(e) for e in errors)


def test_physical_kernel_decorator_rejects_non_const_address():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @movement_kernel
        def kernel(word_id: int):
            q = new_at(0, word_id, 0)
            return squin.broadcast.measure(ilist.IList([q]))

    errors = exc_info.value.errors
    assert any("compile-time constant" in str(e) for e in errors)


def test_literal_address_still_compiles():
    @gemini.logical.kernel(aggressive_unroll=True, verify=True)
    def kernel() -> tuple:
        register = ilist.IList([new_at(0, 4, 0)])
        return gemini.logical.default_post_processing(register)

    assert kernel is not None


def test_callee_address_arg_still_compiles():
    """An address threaded through a callee parameter folds once inlined.

    This is the case the pass must not reject: only an *entry* kernel's own
    argument is genuinely unresolvable, because a parameter scan compiles one
    program and binds arguments at runtime.
    """

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def helper(word_id: int):
        return new_at(0, word_id, 0)

    @gemini.logical.kernel(aggressive_unroll=True, verify=True)
    def outer() -> tuple:
        register = ilist.IList([helper(4)])
        return gemini.logical.default_post_processing(register)

    assert outer is not None


def test_non_const_address_does_not_mask_later_duplicates():
    """``expect_const`` used to raise here, aborting the duplicate analysis mid-walk
    and dropping every duplicate after it. Both diagnostics must now report.
    """

    with pytest.raises(ValidationErrorGroup) as exc_info:

        @movement_kernel
        def kernel(w: int):
            q0 = new_at(0, w, 0)  # non-const, and ordered first
            q1 = new_at(0, 1, 0)
            q2 = new_at(0, 1, 0)  # genuine duplicate, after the non-const one
            return squin.broadcast.measure(ilist.IList([q0, q1, q2]))

    reported = str(exc_info.value.errors)
    assert "compile-time constant" in reported
    assert "pinned by two" in reported
