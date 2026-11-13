"""Tests for LogicalPipeline."""

import pytest
import bloqade.squin as squin
from kirin import rewrite
from kirin.analysis import CallGraph
from bloqade.lanes.dialects import move, place
from bloqade.lanes.transform import (
    LogicalPipeline,
    LogicalNativeToPlace,
    PhysicalNativeToPlace,
    transversal_rewrites,
)
from bloqade.squin.gate.stmts import CX, CY, CZ, H, S, Swap, SqrtX, SqrtY
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome

import bloqade.gemini as gemini


def test_logical_pipeline_smoke():
    """2-qubit Bell kernel compiles end-to-end via LogicalPipeline."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    out = LogicalPipeline().emit(kernel)
    assert out is not None
    fills = [s for s in out.callable_region.walk() if isinstance(s, move.Fill)]
    assert len(fills) == 1


def _apply_squin_clifford_rules(kernel, transversal_rewrite: bool):
    """Run ``LogicalNativeToPlace``'s squin→squin stage — the Clifford
    decomposition plus, when enabled, the Steane transversal adjoint swap."""
    stage = LogicalNativeToPlace(transversal_rewrite=transversal_rewrite)
    out = kernel.similar(kernel.dialects.add(place))
    CallGraphPass(out.dialects, rewrite.Chain(*stage._squin_clifford_rules()))(out)
    return out


def _gate_sequence(mt) -> list[str]:
    """Name every squin Clifford statement in order, adjoints marked."""
    names = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, (S, SqrtX, SqrtY)):
            names.append(f"{stmt.name}{'_adj' if stmt.adjoint else ''}")
        elif isinstance(stmt, CZ):
            names.append(stmt.name)
    return names


def test_logical_squin_clifford_rules_swap_steane_transversal_adjoints():
    """With transversal_rewrite=True, the squin→squin stage swaps the Steane
    transversal Clifford adjoints."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.sqrt_x(reg[0])
        squin.sqrt_x_adj(reg[0])
        squin.s(reg[0])
        squin.s_adj(reg[0])
        gemini.logical.terminal_measure(reg)

    out = _apply_squin_clifford_rules(kernel, transversal_rewrite=True)
    assert _gate_sequence(out) == ["sqrt_x_adj", "sqrt_x", "s_adj", "s"]

    untouched = _apply_squin_clifford_rules(kernel, transversal_rewrite=False)
    assert _gate_sequence(untouched) == ["sqrt_x", "sqrt_x_adj", "s", "s_adj"]


def test_logical_squin_clifford_rules_reach_into_cy_decomposition():
    """Regression for bloqade-internal#404: the Clifford decomposition has to
    run before the transversal swap, so that CY's sqrt(X) layers exist as
    statements and get flipped. Otherwise logical CY is Zbar(control) . CY."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.cy(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    out = _apply_squin_clifford_rules(kernel, transversal_rewrite=True)
    assert _gate_sequence(out) == ["sqrt_x_adj", "cz", "sqrt_x"]

    untouched = _apply_squin_clifford_rules(kernel, transversal_rewrite=False)
    assert _gate_sequence(untouched) == ["sqrt_x", "cz", "sqrt_x_adj"]


def test_logical_squin_clifford_rules_leave_cx_handedness_alone():
    """CX conjugates by sqrt(Y), which needs no transversal correction: neither
    of its images of Xbar/Zbar is +/- Ybar."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.cx(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    out = _apply_squin_clifford_rules(kernel, transversal_rewrite=True)
    assert _gate_sequence(out) == ["sqrt_y_adj", "cz", "sqrt_y"]


def test_physical_squin_clifford_rules_decompose_without_swapping():
    """The decomposition is shared by both pipelines; only the logical one
    appends the transversal swap.

    A plain ``@squin.kernel`` is not inlined at this stage, so the gates still
    sit inside the squin stdlib kernels — which is why the pass runs over the
    whole call graph rather than just the entry point."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cy(reg[0], reg[1])
        squin.broadcast.measure(reg)

    stage = PhysicalNativeToPlace()
    out = kernel.similar(kernel.dialects.add(place))
    CallGraphPass(out.dialects, rewrite.Chain(*stage._squin_clifford_rules()))(out)

    reachable = {mt.sym_name: mt for mt in CallGraph(out).edges}
    assert _gate_sequence(reachable["cy"]) == ["sqrt_x", "cz", "sqrt_x_adj"]
    assert _gate_sequence(reachable["h"]) == ["s", "sqrt_x", "s"]

    # No composite Clifford survives anywhere in the call graph.
    assert not [
        stmt
        for mt in reachable.values()
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, (H, CX, CY, Swap))
    ]


def test_logical_pipeline_produces_logical_initialize():
    """Logical pipeline inserts LogicalInitialize (not just Fill)."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    out = LogicalPipeline().emit(kernel)
    inits = [
        s for s in out.callable_region.walk() if isinstance(s, move.LogicalInitialize)
    ]
    assert len(inits) >= 1


def test_logical_pipeline_layout_heuristic_default_is_none():
    """LogicalPipeline.layout_heuristic defaults to None."""
    pipeline = LogicalPipeline()
    assert pipeline.layout_heuristic is None


def test_logical_pipeline_resolves_none_to_logical_defaults(monkeypatch):
    """When layout_heuristic is None, LogicalPipeline passes LogicalLayoutHeuristic
    to the place→move stage."""
    from bloqade.lanes.transform import PlaceToMove
    from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic

    captured: dict = {}
    _orig_emit = PlaceToMove.emit

    def spy_emit(self_inner, mt, no_raise=True):
        captured["layout_heuristic_type"] = type(self_inner.layout_heuristic)
        return _orig_emit(self_inner, mt, no_raise=no_raise)

    monkeypatch.setattr(PlaceToMove, "emit", spy_emit)

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    LogicalPipeline().emit(kernel)
    assert captured["layout_heuristic_type"] is LogicalLayoutHeuristic


def test_logical_pipeline_layout_heuristic_mismatch_warns():
    """resolved_layout_heuristic warns when the explicit heuristic carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )

    logical_arch = LogicalPipeline().arch_spec
    physical_arch = get_physical_arch_spec()
    assert logical_arch != physical_arch

    mismatched_heuristic = LogicalLayoutHeuristic(arch_spec=physical_arch)
    pipeline = LogicalPipeline(layout_heuristic=mismatched_heuristic)

    with pytest.warns(
        UserWarning, match="layout_heuristic was constructed with a different"
    ):
        result = pipeline.resolved_layout_heuristic

    assert result is mismatched_heuristic


def test_logical_pipeline_placement_strategy_mismatch_warns():
    """resolved_placement_strategy warns when the explicit strategy carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )

    logical_arch = LogicalPipeline().arch_spec
    physical_arch = get_physical_arch_spec()
    assert logical_arch != physical_arch

    mismatched_strategy = LogicalPlacementStrategyNoHome(arch_spec=physical_arch)
    pipeline = LogicalPipeline(placement_strategy=mismatched_strategy)

    with pytest.warns(
        UserWarning, match="placement_strategy was constructed with a different"
    ):
        result = pipeline.resolved_placement_strategy

    assert result is mismatched_strategy


def test_transversal_rewrites_direct():
    """transversal_rewrites() rewrites the method in place and returns it."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    result = transversal_rewrites(kernel, rewrite_logical_initialize=False)
    assert result is kernel
