"""Pipeline-level tests for Task E3: eager gemini validation in NativeToPlace.emit.

These tests verify that both the per-statement (E1) and cross-statement
duplicate-address (E2) validators run eagerly via squin_to_move before the
circuit→place lowering, and that errors surface as ValidationErrorGroup.
"""

from test.gemini._squin_to_move_helper import squin_to_move

import pytest
import bloqade.squin as squin
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects.arch import new_at
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome

import bloqade.gemini as gemini

# ---------------------------------------------------------------------------
# Common heuristic / strategy fixtures
# ---------------------------------------------------------------------------

_LAYOUT = LogicalLayoutHeuristic()
_PLACEMENT = LogicalPlacementStrategyNoHome()


def _compile(kernel):
    """Run squin_to_move with the standard logical heuristic pair."""
    return squin_to_move(
        kernel,
        layout_heuristic=_LAYOUT,
        placement_strategy=_PLACEMENT,
    )


# ---------------------------------------------------------------------------
# 1. Pipeline catches const-prop failure
# ---------------------------------------------------------------------------


def test_pipeline_catches_non_const_arg():
    """A new_at whose zone_id is a function argument (not compile-time const)
    should cause squin_to_move to raise with a 'compile-time constant' message.
    """

    @gemini.logical.kernel(verify=False)
    def kernel(z: int):
        q = new_at(z, 0, 0)  # z is a function arg — not const  # noqa: F841

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("compile-time constant" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# 2. Pipeline catches range failure
# ---------------------------------------------------------------------------


def test_pipeline_catches_out_of_range_address():
    """new_at(99, 0, 0) with zone 99 outside the arch should raise with an
    'Invalid location address' message.
    """

    @gemini.logical.kernel(verify=False)
    def kernel():
        q = new_at(99, 0, 0)  # zone 99 is out of range  # noqa: F841

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("Invalid location address" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# 3. Pipeline catches duplicate addresses
# ---------------------------------------------------------------------------


def test_pipeline_catches_duplicate_addresses():
    """Two new_at calls pinning the same address should raise with 'pinned by two'."""

    @gemini.logical.kernel(verify=False)
    def kernel():
        q0 = new_at(0, 0, 0)  # noqa: F841
        q1 = new_at(0, 0, 0)  # same address — duplicate  # noqa: F841

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("pinned by two" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# 4. Valid kernel compiles end-to-end without errors
# ---------------------------------------------------------------------------


def test_pipeline_valid_kernel_compiles():
    """A gemini kernel with no new_at calls should compile without validation errors."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        q = squin.qalloc(2)
        squin.h(q[0])
        squin.cx(q[0], q[1])
        gemini.logical.terminal_measure(q)

    # Should not raise.
    _compile(kernel)
