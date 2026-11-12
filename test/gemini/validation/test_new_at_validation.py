"""The kernel decorator rejects duplicate pinned qubit addresses."""

import pytest
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects.arch import new_at

import bloqade.gemini as gemini


def test_kernel_decorator_catches_duplicates():
    """Default verify=True surfaces duplicate addresses at kernel-decoration time."""

    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel()
        def kernel():
            q0 = new_at(0, 0, 0)  # noqa: F841
            q1 = new_at(0, 0, 0)  # same address — duplicate  # noqa: F841

    errors = exc_info.value.errors
    assert any("pinned by two" in str(e) for e in errors)
