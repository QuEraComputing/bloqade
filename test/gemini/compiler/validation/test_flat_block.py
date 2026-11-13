"""Tests for FlatBlockValidation."""

import pytest
from kirin import ir, types as kirin_types
from kirin.dialects import py, func
from kirin.validation import ValidationSuite
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.validation.flat_block import FlatBlockValidation

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical


def _method(*blocks: ir.Block) -> ir.Method:
    """A minimal method whose callable region holds ``blocks``."""

    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(list(blocks)),
    )
    return out


def _errors(method: ir.Method) -> list[str]:
    result = ValidationSuite([FlatBlockValidation]).validate(method)
    return [] if result.is_valid else ["invalid"] * result.error_count()


def test_a_single_block_region_is_valid():
    assert _errors(_method(ir.Block([py.Constant(1), func.Return()]))) == []


def test_a_multi_block_region_is_rejected():
    """Walk visits blocks in reverse, so a frame spanning them accumulates
    backwards -- see test_walk_order.py."""
    assert (
        _errors(
            _method(
                ir.Block([py.Constant(1)]), ir.Block([py.Constant(2), func.Return()])
            )
        )
        != []
    )


def test_a_nested_function_is_rejected():
    """Walk reaches a nested function's region before the statement owning it,
    so a rule resetting per-region state has it wiped mid-block."""
    inner = func.Function(
        sym_name="inner",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(ir.Block([func.ConstantNone(), func.Return()])),
    )
    assert _errors(_method(ir.Block([inner, func.Return()]))) != []


def test_a_multi_block_region_nested_in_a_statement_is_rejected():
    """The scan covers every region, not just the callable one."""
    inner = func.Function(
        sym_name="inner",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region([ir.Block([py.Constant(1)]), ir.Block([func.Return()])]),
    )
    # Two errors: the nested function, and its two-block region.
    assert len(_errors(_method(ir.Block([inner, func.Return()])))) == 2


def test_a_real_compiled_kernel_passes():
    """The shape the post-unroll window actually produces.

    This is the check that matters: if a normal logical kernel ever stopped
    satisfying the precondition, `EliminateRz` would be silently skipped under
    `no_raise` and every `Rz` would survive to the backend.
    """

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    out = LogicalPipeline(
        get_arch_spec(), transversal_rewrite=True, simulation=False
    ).emit(kernel)

    # The pipeline has moved past the native window by now, but the flat shape
    # is what it was validated against and is preserved through lowering.
    assert _errors(out) == []


@pytest.mark.parametrize("no_raise", [False, True])
def test_the_pipeline_still_compiles_a_normal_kernel(no_raise):
    """The validation gate must not reject what the pipeline normally emits."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(1)
        squin.s(reg[0])
        gemini_logical.terminal_measure(reg)

    LogicalPipeline(get_arch_spec(), transversal_rewrite=True, simulation=False).emit(
        kernel, no_raise=no_raise
    )
