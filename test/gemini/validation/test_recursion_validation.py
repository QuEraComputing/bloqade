"""Recursive Gemini kernels must be rejected, and rejected *fast*.

Before the guard in `logical.kernel`, a cyclic call graph did not raise -- it
sent `AddressAnalysis` into a phi**max_depth traversal that never returned
(bloqade-lanes#921). Some tests here would hang rather than fail if the guard
regressed, which is why they assert on elapsed time as well as on the error.

Note that recursion reaches the guard in two different IR shapes. A kernel
calling itself is lowered as a *dynamic* call on its own `%<name>_self` block
argument, and is only rewritten into a static `func.Invoke` after `run_pass`
returns; a call into an already-built kernel is a static `func.Invoke` straight
away. `CallGraph` has to understand both, so both are covered below.
"""

import time

import pytest
import bloqade.squin as squin
from kirin.dialects import ilist
from kirin.validation import ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from kirin.lowering.exception import BuildError
from bloqade.lanes.dialects.arch import new_at
from bloqade.lanes.validation.recursion import (
    CallGraph,
    NoRecursionValidation,
    NoOpaqueCallValidation,
    format_cycle,
    check_call_graph,
)

import bloqade.gemini as gemini
from bloqade.gemini.logical.stdlib import default_post_processing

# Generous enough to absorb a slow CI box, tight enough that the pre-fix
# behaviour (effectively unbounded) cannot pass.
REJECTION_BUDGET_S = 60.0


def _messages(err: ValidationErrorGroup) -> list[str]:
    return [e.args[0] if e.args else str(e) for e in err.errors]


# A kernel from another dialect group, so the Gemini guard never runs on it and
# the call through its parameter survives.
#
# Every kernel fixture in this module sits at module scope rather than inside the
# test that uses it. kirin does read an enclosing function's locals, but through
# `inspect.getclosurevars` wrapped in a bare `except Exception`
# (`kirin/lowering/python/lowering.py`). Handing a kernel its own name as a value
# makes that name a freevar whose cell is still empty while the decorator runs,
# so the lookup raises `ValueError: Cell is empty` and *every* nonlocal is
# dropped together -- a sibling kernel defined beside it then fails to lower as
# "not defined", pointing at the wrong name entirely.
@squin.kernel
def _foreign_calls_its_parameter_twice(f):
    f()
    f()


# A chain of foreign kernels ending in self-recursion, for the case where the
# cycle is several hops below the kernel being validated. Each link refers to one
# already defined above it, which is the only direction a source-level reference
# can go.
@squin.kernel
def _deep_cycle_end(n):
    return _deep_cycle_end(n)


@squin.kernel
def _deep_cycle_mid(n):
    return _deep_cycle_end(n)


@squin.kernel
def _deep_cycle_start(n):
    return _deep_cycle_mid(n)


def test_direct_self_recursion_is_rejected():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def recurse(q):
            recurse(q)

    messages = _messages(exc_info.value)
    assert any("recursion is not supported" in m for m in messages)
    assert any("recurse -> recurse" in m for m in messages)


def test_original_user_report_is_rejected_quickly():
    """The kernel that prompted this work, kept verbatim.

    A user hit an indefinite hang compiling this. Measured on the same machine:
    with the guard removed it was still running after 60s; with it, the kernel is
    rejected in 0.02s.

    Worth keeping as its own case despite the simpler self-recursion tests above,
    because it is the shape a real caller wrote: the recursive call takes no
    arguments and its result is discarded, and it sits in a kernel whose other
    statements are ordinary work. It also pins both flags the user had set --
    neither `aggressive_unroll` nor `verify` may move the guard, since the pass
    that diverges runs before the first and outside the second.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=True, verify=True)
        def main():
            a = new_at(0, 0, 0)
            b = new_at(0, 8, 0)
            main()
            squin.h(a)
            squin.cx(a, b)
            return default_post_processing(ilist.IList([a, b]))

    assert time.monotonic() - started < REJECTION_BUDGET_S
    assert any("main -> main" in m for m in _messages(exc_info.value))


def test_verify_false_still_rejects_recursion():
    """`verify=False` must not be able to skip the guard.

    The address analysis that diverges runs unconditionally in `run_pass`, so a
    check gated on `verify` would leave the hang reachable.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(verify=False, no_raise=True)
        def recurse(q):
            recurse(q)

    assert time.monotonic() - started < REJECTION_BUDGET_S


def test_aggressive_unroll_still_rejects_recursion():
    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(aggressive_unroll=True)
        def recurse(q):
            recurse(q)


def test_original_hang_reproducer_is_rejected_quickly():
    """The mwe from bloqade-lanes#921: `main -> B`, `B -> {B, main}`.

    This is the Fibonacci-shaped graph whose cost grew as phi**depth. `B` alone
    is enough to reject: its `B(main)` is a static self edge and its `main()` is
    a call through a parameter, so both rules fire and `main` never gets built.
    """

    @gemini.logical.kernel(verify=False)
    def simple(reg):
        squin.broadcast.h(reg)

    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def B(main):
            B(main)
            main()

    assert time.monotonic() - started < REJECTION_BUDGET_S
    # Both rules report; `check_call_graph` does not stop at the first.
    messages = _messages(exc_info.value)
    assert any("B -> B" in m for m in messages)
    assert any("calling the parameter 'main'" in m for m in messages)


def test_check_call_graph_accepts_an_acyclic_kernel():
    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])
        gemini.logical.terminal_measure(q)

    check_call_graph(main)


def test_mutual_recursion_between_kernels_cannot_be_lowered():
    """Two kernels calling each other is not expressible, let alone compilable.

    The first kernel's forward reference fails during lowering, so the guard
    never sees it. Pinned here so that a future lowering change which *does*
    admit the forward reference shows up as a failure needing a guard update.

    A closure is the obvious way to try to get around this, so the second case
    rules it out: nothing defers name resolution, and the sibling is just as
    unbound as a local of a factory function as it is as a module global.
    """

    with pytest.raises(BuildError):

        @gemini.logical.kernel(verify=False)
        def ping(q):
            pong(q)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    with pytest.raises(BuildError):

        @gemini.logical.kernel(verify=False)
        def ping_local(q):
            pong_local(q)

        @gemini.logical.kernel(verify=False)
        def pong_local(q):
            ping_local(q)


def test_calling_a_parameter_is_rejected():
    """A call through a parameter has no statically knowable target.

    `CallGraph` cannot see such an edge, so it is rejected by shape rather than
    resolved -- resolving it would need the constant propagation that diverges
    on the cyclic inputs this guard exists to catch.
    """
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def higher_order(f):
            f()

    assert any(
        "calling the parameter 'f' is not supported" in m
        for m in _messages(exc_info.value)
    )


def test_passing_a_parameter_to_a_subkernel_is_allowed():
    """Only *calling* a parameter is rejected, not using one.

    Pins the boundary of `NoOpaqueCallValidation`: it keys on the callee of a
    `func.Call`, so a parameter forwarded as an ordinary argument -- and the
    enclosing kernel's other parameters -- must not trip it.
    """

    @gemini.logical.kernel(verify=False)
    def flip(q):
        squin.x(q)

    @gemini.logical.kernel(verify=False)
    def outer(q):
        flip(q)

    ValidationSuite([NoOpaqueCallValidation]).validate(outer).raise_if_invalid()


def test_dynamic_cycle_with_branching_cannot_be_built():
    """The shape that used to hang with no static edge anywhere.

    Previously `main` passed itself to `B`, which only ever called it
    dynamically; two dynamic calls give the cycle branching factor 2, so the
    address analysis cost was phi**max_depth while `CallGraph` saw no edge at
    all. `B` is now rejected at its own definition, so the cycle can never be
    assembled -- which is what makes that hang unreachable rather than merely
    less likely.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def B(f):
            f()
            f()

    assert time.monotonic() - started < REJECTION_BUDGET_S
    # Reported once per offending call site, not once per kernel.
    messages = _messages(exc_info.value)
    assert sum("calling the parameter 'f'" in m for m in messages) == 2


def test_opaque_call_inside_a_reachable_callee_is_rejected():
    """Scanning only the entry method would leave the hang reachable.

    A kernel from another dialect group never runs this guard, so it may call
    its own parameter. A Gemini kernel can statically invoke one; with
    `inline=False` the dynamic call survives to `AddressAnalysis`, and passing
    the Gemini kernel's own self value into a callee that calls it twice
    rebuilds the branching cycle -- with no static edge anywhere for
    `NoRecursionValidation` to find. Confirmed to hang before this was covered.
    """

    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
        def main():
            _foreign_calls_its_parameter_twice(main)

    assert time.monotonic() - started < REJECTION_BUDGET_S
    # The owning kernel is named, since the caret stays on the entry method.
    messages = _messages(exc_info.value)
    assert (
        sum(
            "calling the parameter 'f' in '_foreign_calls_its_parameter_twice'" in m
            for m in messages
        )
        == 2
    )


def test_non_recursive_subkernel_calls_are_unaffected():
    @gemini.logical.kernel(verify=False)
    def flip(q):
        squin.x(q)

    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        flip(q[0])
        flip(q[1])
        gemini.logical.terminal_measure(q)

    main.print()


def test_diamond_call_graph_is_not_a_cycle():
    """Two paths to the same callee is sharing, not recursion."""

    @gemini.logical.kernel(verify=False)
    def leaf(q):
        squin.x(q)

    @gemini.logical.kernel(verify=False)
    def left(q):
        leaf(q)

    @gemini.logical.kernel(verify=False)
    def right(q):
        leaf(q)

    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        left(q[0])
        right(q[1])
        gemini.logical.terminal_measure(q)

    assert CallGraph(main).find_cycles() == []


def test_callgraph_records_edges_caller_to_callee():
    @gemini.logical.kernel(verify=False)
    def leaf(q):
        squin.x(q)

    # Disable both inlining paths to keep `func.Invoke` instead of splicing the
    # callee body in; verify=False because a surviving invoke is separately
    # rejected by the logical validation.
    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(1)
        leaf(q[0])

    graph = CallGraph(caller)
    assert graph.entry is caller

    callees = graph.callees(caller)
    assert leaf in callees
    # Stdlib kernels are ordinary edges too -- `squin.qalloc` is itself a kernel.
    assert {m.sym_name for m in callees} == {"leaf", "qalloc"}
    # `callees` is sorted, so diagnostics are stable across runs.
    assert [m.sym_name for m in callees] == ["leaf", "qalloc"]
    assert graph.find_cycles() == []


def test_callgraph_terminates_and_finds_multi_node_cycles():
    """`kirin.analysis.CallGraph` raises RecursionError on cyclic input (kirin#703).

    A multi-kernel cycle cannot be built through the decorator (see
    `test_mutual_recursion_between_kernels_cannot_be_lowered`), so the traversal
    is exercised directly on an injected cycle.
    """

    @gemini.logical.kernel(verify=False)
    def a(q):
        squin.x(q)

    @gemini.logical.kernel(verify=False)
    def b(q):
        squin.y(q)

    @gemini.logical.kernel(verify=False)
    def c(q):
        squin.z(q)

    graph = CallGraph(a)
    graph.edges = {a: {b}, b: {c}, c: {a}}

    cycles = graph.find_cycles()
    assert len(cycles) == 1
    assert set(cycles[0].members) == {a, b, c}
    # The entry method is in the cycle, so there is nothing to route through.
    assert cycles[0].route == ()
    # Reported as a rotation starting from the DFS root, with the loop closed.
    assert format_cycle(cycles[0]) == "a -> b -> c -> a"


def test_callgraph_reports_one_cycle_per_group():
    @gemini.logical.kernel(verify=False)
    def a(q):
        squin.x(q)

    @gemini.logical.kernel(verify=False)
    def b(q):
        squin.y(q)

    graph = CallGraph(a)
    # `a` self-loops and also sits in a 2-cycle with `b`.
    graph.edges = {a: {a, b}, b: {a}}

    cycles = graph.find_cycles()
    assert [set(cycle.members) for cycle in cycles] == [{a}, {a, b}]


def test_cycle_below_the_entry_reports_the_route():
    """A cycle need not live in the kernel being validated.

    Any Gemini kernel containing a cycle is rejected at its own definition, so
    it can never become a callee -- but a kernel built by another dialect group
    runs no such guard, and a Gemini kernel may invoke one. Naming only the
    cycle would then leave the reader with no way to find it.
    """

    @gemini.logical.kernel(verify=False)
    def a(q):
        squin.x(q)

    @gemini.logical.kernel(verify=False)
    def b(q):
        squin.y(q)

    @gemini.logical.kernel(verify=False)
    def c(q):
        squin.z(q)

    graph = CallGraph(a)
    # a -> b -> c -> c: the cycle is two hops below the entry.
    graph.edges = {a: {b}, b: {c}, c: {c}}

    (cycle,) = graph.find_cycles()
    assert cycle.members == (c,)
    assert cycle.route == (a, b)
    assert format_cycle(cycle) == "c -> c (reached via a -> b -> c)"


def test_no_recursion_validation_pass_reports_no_errors_for_acyclic_kernel():
    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])
        gemini.logical.terminal_measure(q)

    ValidationSuite([NoRecursionValidation]).validate(main).raise_if_invalid()


def test_format_cycle_closes_the_loop():
    @gemini.logical.kernel(verify=False)
    def a(q):
        squin.x(q)

    @gemini.logical.kernel(verify=False)
    def b(q):
        squin.y(q)

    assert format_cycle([a]) == "a -> a"
    assert format_cycle([a, b]) == "a -> b -> a"


def test_cycle_far_below_the_entry_is_rejected():
    """The cycle need not be in the kernel being validated.

    A Gemini kernel containing a cycle is rejected at its own definition, so it
    can never become a callee -- but a kernel from another dialect group runs no
    such guard. `main` invokes a chain of three foreign kernels ending in
    self-recursion, so the cycle survives to the kernel that does check, three
    hops down. Naming only the cycle would leave no way to find it, which is what
    `Cycle.route` is for.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
        def main():
            _deep_cycle_start(1)

    assert time.monotonic() - started < REJECTION_BUDGET_S
    assert any(
        "_deep_cycle_end -> _deep_cycle_end (reached via main -> _deep_cycle_start "
        "-> _deep_cycle_mid -> _deep_cycle_end)" in m
        for m in _messages(exc_info.value)
    )


def test_physical_pipeline_rejects_recursion():
    """`physical.kernel` has its own `run_pass` and its own `AddressAnalysis`.

    It does not delegate to the logical pipeline, so it needs the guard
    independently. Without it a recursive physical kernel was accepted, and the
    `#921` shape hung there exactly as it originally did on the logical side.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.physical.kernel(verify=False)
        def k(q):
            k(q)
            squin.h(q)

    assert time.monotonic() - started < REJECTION_BUDGET_S
    assert any("k -> k" in m for m in _messages(exc_info.value))


def test_recursion_through_ilist_map_is_found():
    """`ilist.Map` invokes its `fn` but carries no `ir.StaticCall` trait.

    kirin does not mark these higher-order statements as calls, so a trait-driven
    walk misses the edge entirely -- in both directions. `CallGraph` keys on the
    operand *type* instead, which also completes the graph for correct code.
    """

    @squin.kernel
    def recurse_by_map(xs):
        return ilist.map(recurse_by_map, xs)

    (cycle,) = CallGraph(recurse_by_map).find_cycles()
    assert cycle.members == (recurse_by_map,)


def test_ilist_map_over_a_known_kernel_records_the_edge():
    """The same detection must not reject ordinary higher-order code.

    `gemini/logical/stdlib/qubit_allocation.py` maps over a kernel exactly like
    this, so the edge has to be recorded without an opaque-call error.
    """

    @squin.kernel
    def double(x):
        return x + x

    @squin.kernel
    def uses_map(xs):
        return ilist.map(double, xs)

    graph = CallGraph(uses_map)
    assert double in graph.edges[uses_map]
    assert graph.find_cycles() == []

    _, errors = NoOpaqueCallValidation().run(uses_map)
    assert errors == []
