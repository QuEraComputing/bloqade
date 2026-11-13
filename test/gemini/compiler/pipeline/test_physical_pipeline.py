"""Tests for the physical pipeline and NewPinnedQubit."""

import pytest
import bloqade.squin as squin
from kirin import ir, rewrite
from kirin.dialects import debug, ilist
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects import move, place
from bloqade.lanes.transform import PhysicalPipeline
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.rewrite.circuit2place import RewriteQubitsToPinnedQubits

from bloqade import qubit
from bloqade.gemini.physical import kernel as physical_kernel


def test_new_pinned_qubit_unpinned():
    """NewPinnedQubit with no address has location_address=None."""
    stmt = place.NewPinnedQubit()
    assert stmt.location_address is None


def test_new_pinned_qubit_pinned():
    """NewPinnedQubit with a LocationAddress stores it correctly."""
    addr = LocationAddress(word_id=4, site_id=2, zone_id=0)
    stmt = place.NewPinnedQubit(location_address=addr)
    assert stmt.location_address == addr


def test_rewrite_new_to_pinned():
    """qubit.stmts.New is lowered to NewPinnedQubit(location_address=None)."""
    block = ir.Block()
    plain_new = qubit.stmts.New()
    block.stmts.append(plain_new)

    rewrite.Walk(RewriteQubitsToPinnedQubits()).rewrite(block)

    pinned = [s for s in block.stmts if isinstance(s, place.NewPinnedQubit)]
    assert len(pinned) == 1
    assert pinned[0].location_address is None


def test_physical_pipeline_smoke():
    """Single-qubit kernel with terminal measure compiles end-to-end."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.qubit.measure(ilist.IList([reg[0]]))  # type: ignore[arg-type]

    out = PhysicalPipeline().emit(kernel)
    assert out is not None
    fills = [s for s in out.callable_region.walk() if isinstance(s, move.Fill)]
    assert len(fills) == 1


@pytest.mark.parametrize("no_raise", [False, True])
def test_physical_pipeline_removes_debug_info_from_move_kernel(no_raise: bool):
    @physical_kernel
    def kernel():
        reg = squin.qalloc(1)
        debug.info("physical kernel marker")
        debug.info("another physical kernel marker", reg[0])
        return squin.broadcast.measure(reg)

    assert (
        sum(isinstance(stmt, debug.Info) for stmt in kernel.callable_region.walk()) == 2
    )

    out = PhysicalPipeline().emit(kernel, no_raise=no_raise)

    assert not any(isinstance(stmt, debug.Info) for stmt in out.callable_region.walk())
    assert (
        sum(isinstance(stmt, debug.Info) for stmt in kernel.callable_region.walk()) == 2
    )


def test_physical_pipeline_removes_debug_info_from_helper():
    @physical_kernel
    def kernel():
        def log_debug():
            debug.info("helper marker")

        reg = squin.qalloc(1)
        log_debug()
        return squin.broadcast.measure(reg)

    assert any(isinstance(stmt, debug.Info) for stmt in kernel.callable_region.walk())

    out = PhysicalPipeline().emit(kernel, no_raise=False)

    assert not any(isinstance(stmt, debug.Info) for stmt in out.callable_region.walk())


def test_physical_pipeline_placement_strategy_default_is_none():
    """PhysicalPipeline.placement_strategy defaults to None; emit() constructs it from arch_spec."""
    pipeline = PhysicalPipeline()
    assert pipeline.placement_strategy is None


def test_physical_pipeline_no_new_pinned_remaining():
    """After compilation, no place.NewPinnedQubit statements remain."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.qubit.measure(ilist.IList([reg[0], reg[1]]))  # type: ignore[arg-type]

    out = PhysicalPipeline().emit(kernel)
    pinned_stmts = [
        s for s in out.callable_region.walk() if isinstance(s, place.NewPinnedQubit)
    ]
    assert pinned_stmts == []


def test_physical_pipeline_missing_measure_raises():
    """Kernel with no terminal measure raises at validation."""

    @squin.kernel
    def kernel():
        _reg = squin.qalloc(1)

    import pytest

    with pytest.raises(ValidationErrorGroup):
        PhysicalPipeline().emit(kernel, no_raise=False)


def test_physical_pipeline_resolves_none_to_physical_defaults(monkeypatch):
    """When layout_heuristic and placement_strategy are None, PhysicalPipeline
    resolves them to PhysicalLayoutHeuristicGraphPartitionCenterOut wrapped in
    PalindromePlacementStrategy(NoHomePlacementStrategy) before passing them
    to the place→move stage."""
    from bloqade.lanes.transform import PlaceToMove
    from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
    from bloqade.lanes.heuristics.physical.layout import (
        PhysicalLayoutHeuristicGraphPartitionCenterOut,
    )
    from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy

    captured: dict = {}
    _orig_emit = PlaceToMove.emit

    def spy_emit(self_inner, mt, no_raise=True):
        captured["layout_heuristic_type"] = type(self_inner.layout_heuristic)
        captured["placement_strategy"] = self_inner.placement_strategy
        return _orig_emit(self_inner, mt, no_raise=no_raise)

    monkeypatch.setattr(PlaceToMove, "emit", spy_emit)

    @squin.kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.qubit.measure(ilist.IList([reg[0]]))  # type: ignore[arg-type]

    PhysicalPipeline().emit(kernel)

    assert (
        captured["layout_heuristic_type"]
        is PhysicalLayoutHeuristicGraphPartitionCenterOut
    )
    strategy = captured["placement_strategy"]
    assert isinstance(strategy, PalindromePlacementStrategy)
    assert isinstance(strategy.inner, NoHomePlacementStrategy)


def test_physical_pipeline_layout_heuristic_mismatch_warns():
    """resolved_layout_heuristic warns when the explicit heuristic carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
    from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic

    physical_arch = PhysicalPipeline().arch_spec
    logical_arch = get_logical_arch_spec()
    assert physical_arch != logical_arch

    mismatched_heuristic = LogicalLayoutHeuristic(arch_spec=logical_arch)
    pipeline = PhysicalPipeline(layout_heuristic=mismatched_heuristic)

    with pytest.warns(
        UserWarning, match="layout_heuristic was constructed with a different"
    ):
        result = pipeline.resolved_layout_heuristic

    assert result is mismatched_heuristic


def test_physical_pipeline_placement_strategy_mismatch_warns():
    """resolved_placement_strategy warns when the explicit strategy carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
    from bloqade.lanes.heuristics.logical.placement import (
        LogicalPlacementStrategyNoHome,
    )

    physical_arch = PhysicalPipeline().arch_spec
    logical_arch = get_logical_arch_spec()
    assert physical_arch != logical_arch

    mismatched_strategy = LogicalPlacementStrategyNoHome(arch_spec=logical_arch)
    pipeline = PhysicalPipeline(placement_strategy=mismatched_strategy)

    with pytest.warns(
        UserWarning, match="placement_strategy was constructed with a different"
    ):
        result = pipeline.resolved_placement_strategy

    assert result is mismatched_strategy
