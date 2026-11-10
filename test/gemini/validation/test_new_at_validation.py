"""Tests for per-statement validation of gemini.operations.NewAt."""

import pytest
from kirin.validation import ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects.arch import new_at
from bloqade.lanes.validation.address import get_validation
from bloqade.lanes.arch.gemini.physical import get_physical_layout_arch_spec
from bloqade.lanes.validation.duplicate_address import DuplicateAddressValidation

import bloqade.gemini as gemini

# ---------------------------------------------------------------------------
# Validator fixture
# ---------------------------------------------------------------------------


def _make_validator() -> ValidationSuite:
    # ``get_validation(arch_spec)`` bakes the arch spec into a ClassVar so the
    # returned pass is a no-arg-constructible class usable in a ValidationSuite.
    return ValidationSuite([get_validation(get_physical_layout_arch_spec())])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_const_foldability_failure():
    """new_at(z, 0, 0) where z is a function argument is not const-foldable.

    The validator should surface a 'is not a compile-time constant' error for
    the zone_id argument.
    """

    @gemini.logical.kernel(verify=False)
    def kernel(z: int):
        # z is a function argument — not a compile-time constant.
        q = new_at(z, 0, 0)  # noqa: F841

    result = _make_validator().validate(kernel)
    with pytest.raises(ValidationErrorGroup) as exc_info:
        result.raise_if_invalid()

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("compile-time constant" in str(e) for e in errors)


def test_range_failure():
    """new_at(99, 0, 0) where zone 99 does not exist in the arch spec should
    surface an 'Invalid location address' error.
    """

    @gemini.logical.kernel(verify=False)
    def kernel():
        q = new_at(99, 0, 0)  # zone 99 is out of range  # noqa: F841

    result = _make_validator().validate(kernel)
    with pytest.raises(ValidationErrorGroup) as exc_info:
        result.raise_if_invalid()

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("Invalid location address" in str(e) for e in errors)


def test_valid_new_at_no_diagnostics():
    """new_at(0, 0, 0) for a valid arch produces no validation errors.

    (zone=0, word=0, site=0) is a home site in the physical layout arch spec.
    """

    @gemini.logical.kernel(verify=False)
    def kernel():
        q = new_at(0, 0, 0)  # noqa: F841

    result = _make_validator().validate(kernel)
    # Should not raise.
    result.raise_if_invalid()


# ---------------------------------------------------------------------------
# DuplicateAddressValidation tests (E2)
# ---------------------------------------------------------------------------


def test_duplicate_addresses_reported():
    """Two new_at calls pinning the same (zone, word, site) produce an error."""

    @gemini.logical.kernel(verify=False)
    def kernel():
        q0 = new_at(0, 0, 0)  # noqa: F841
        q1 = new_at(0, 0, 0)  # same address — duplicate  # noqa: F841

    dup_pass = DuplicateAddressValidation()
    _, errors = dup_pass.run(kernel)
    assert len(errors) >= 1
    assert any("pinned by two" in str(e) for e in errors)


def test_duplicate_via_constant_folded_args():
    """new_at(0, 1+0, 2) and new_at(0, 1, 2) resolve to the same address after
    const-fold and should be flagged as duplicates.
    """

    @gemini.logical.kernel(verify=False)
    def kernel():
        q0 = new_at(0, 1 + 0, 2)  # folds to (zone=0, word=1, site=2)  # noqa: F841
        q1 = new_at(0, 1, 2)  # same resolved address  # noqa: F841

    dup_pass = DuplicateAddressValidation()
    _, errors = dup_pass.run(kernel)
    assert len(errors) >= 1
    assert any("pinned by two" in str(e) for e in errors)


def test_distinct_addresses_no_error():
    """Two new_at calls with distinct (zone, word, site) should produce no errors."""

    @gemini.logical.kernel(verify=False)
    def kernel():
        q0 = new_at(0, 0, 0)  # noqa: F841
        q1 = new_at(0, 0, 1)  # different site_id  # noqa: F841

    dup_pass = DuplicateAddressValidation()
    _, errors = dup_pass.run(kernel)
    assert errors == []


def test_non_constant_arg_skipped_silently():
    """A new_at with a non-constant arg is silently skipped — no crash."""

    @gemini.logical.kernel(verify=False)
    def kernel(z: int):
        q = new_at(z, 0, 0)  # z is not const — E1 handles this error  # noqa: F841

    dup_pass = DuplicateAddressValidation()
    # Must not raise; the pass skips stmts whose args lack const hints.
    _, errors = dup_pass.run(kernel)
    assert errors == []


def test_kernel_decorator_catches_duplicates():
    """Default verify=True surfaces duplicate addresses at kernel-decoration time."""

    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel()
        def kernel():
            q0 = new_at(0, 0, 0)  # noqa: F841
            q1 = new_at(0, 0, 0)  # same address — duplicate  # noqa: F841

    errors = exc_info.value.errors
    assert any("pinned by two" in str(e) for e in errors)
