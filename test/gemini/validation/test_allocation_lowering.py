"""Allocation validation belongs to the compiler, even without a product decorator."""

import pytest
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects import arch, place
from bloqade.lanes.transform.native_to_place import NativeToPlace, PhysicalNativeToPlace

from bloqade import squin
from bloqade.gemini.common import impl as _common_impl  # noqa: F401

kernel = squin.kernel.add(arch)


@pytest.mark.parametrize("compiler", [NativeToPlace, PhysicalNativeToPlace])
@pytest.mark.parametrize("no_raise", [False, True])
def test_dynamic_allocation_rejected_without_arch(compiler, no_raise):
    @kernel
    def dynamic(word: int):
        q = arch.new_at(0, word, 0)
        squin.x(q)
        return squin.measure(q)

    with pytest.raises(ValidationErrorGroup, match="compile-time constant"):
        compiler(arch_spec=None).emit(dynamic, no_raise=no_raise)


@pytest.mark.parametrize("compiler", [NativeToPlace, PhysicalNativeToPlace])
def test_inlined_constant_allocation_lowers_without_arch(compiler):
    @kernel
    def helper(word: int):
        return arch.new_at(0, word, 0)

    @kernel
    def constant():
        q = helper(4)
        squin.x(q)
        return squin.measure(q)

    result = compiler(arch_spec=None).emit(constant)
    assert not any(isinstance(stmt, arch.NewAt) for stmt in result.code.walk())
    assert any(
        isinstance(stmt, (place.NewLogicalQubit, place.NewPinnedQubit))
        for stmt in result.code.walk()
    )
