"""Tests for movement_kernel decorator."""

import pytest
import bloqade.squin as squin
from bloqade.squin import gate, qubit
from kirin.prelude import structural_no_opt
from kirin.dialects import debug, ilist
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.dialects import arch as arch_dialect, arrange as arrange_dialect
from kirin.lowering.exception import BuildError
from bloqade.decoders.dialects import annotate
from bloqade.lanes.bytecode.encoding import LocationAddress

import bloqade.gemini as gemini
from bloqade.gemini import new_at, move_to
from bloqade.gemini.logical import loc, terminal_measure
from bloqade.gemini.physical import kernel as movement_kernel


def test_movement_kernel_exists():
    assert movement_kernel is not None


def test_movement_kernel_has_only_physical_source_dialects():
    assert (
        movement_kernel.data
        == structural_no_opt.union(
            [
                qubit,
                gate,
                annotate,
                arrange_dialect,
                arch_dialect,
                debug,
            ]
        ).data
    )


@pytest.mark.parametrize("aggressive_unroll", [False, True])
def test_movement_kernel_accepts_debug_info(aggressive_unroll: bool):
    @movement_kernel(aggressive_unroll=aggressive_unroll)
    def k():
        q = squin.qalloc(1)
        debug.info("physical kernel debug marker")
        return squin.broadcast.measure(q)

    assert any(isinstance(stmt, debug.Info) for stmt in k.callable_region.walk())


def test_movement_kernel_validation_does_not_lower_to_native(monkeypatch):
    import bloqade.gemini.physical.group as physical_group

    def fail_if_called(*args, **kwargs):
        raise AssertionError("physical-kernel validation must stay at the SQuIN level")

    monkeypatch.setattr(
        physical_group,
        "SquinToNative",
        fail_if_called,
        raising=False,
    )

    @movement_kernel
    def k():
        q = squin.qalloc(1)
        return squin.broadcast.measure(q)

    assert k is not None


def test_movement_kernel_supports_aggressive_unroll():
    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(1)
        return squin.broadcast.measure(q)

    assert k is not None


def test_movement_kernel_requires_inlining_for_terminal_measure_validation():
    """The post-inline validator cannot inspect an uninterpreted broadcast call."""

    with pytest.raises(ValidationErrorGroup, match="terminal measure"):

        @movement_kernel(inline=False)
        def k():
            q = squin.qalloc(1)
            return squin.broadcast.measure(q)


def test_movement_kernel_accepts_terminal_measurement_of_all_qubits():
    """A physical kernel accepts one terminal measurement of every qubit."""

    @movement_kernel
    def k():
        q = squin.qalloc(11)
        return squin.broadcast.measure(q)

    assert k is not None


def test_movement_kernel_rejects_logical_terminal_measurement():
    """Logical result/postprocessing operations are not physical source IR."""

    with pytest.raises(BuildError, match="unsupported dialect `lanes.steane713`"):

        @movement_kernel
        def k():
            q = squin.qalloc(1)
            return terminal_measure(q)


def test_movement_kernel_rejects_more_than_80_allocated_qubits():
    with pytest.raises(ValidationErrorGroup, match="maximum of 80"):

        @movement_kernel(aggressive_unroll=True)
        def k():
            q = squin.qalloc(81)
            return squin.broadcast.measure(q)


def test_movement_kernel_rejects_missing_terminal_measurement():
    with pytest.raises(ValidationErrorGroup, match="terminal measure"):

        @movement_kernel
        def k():
            q = squin.qalloc(1)  # noqa: F841


def test_movement_kernel_rejects_partial_terminal_measurement():
    with pytest.raises(ValidationErrorGroup, match="all qubits must be measured"):

        @movement_kernel
        def k():
            q = squin.qalloc(2)
            return squin.broadcast.measure(ilist.IList([q[0]]))


def test_movement_kernel_rejects_duplicate_new_at_addresses():
    """Physical source kernels run duplicate-address validation directly."""

    with pytest.raises(ValidationErrorGroup, match="pinned by two"):

        @movement_kernel
        def k():
            q0 = new_at(0, 0, 0)
            q1 = new_at(0, 0, 0)
            return squin.broadcast.measure(ilist.IList([q0, q1]))


def test_movement_kernel_compiles_plain_cz():
    """movement_kernel compiles a plain CZ without move_to."""

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        return squin.broadcast.measure(q)

    assert k is not None


def test_movement_kernel_compiles_move_to_then_cz():
    """movement_kernel accepts move_to followed by CZ."""

    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc_a, loc_b])
        squin.cz(q[0], q[1])
        return squin.broadcast.measure(q)

    assert k is not None


def test_loc_inline_compiles():
    """loc() constructs a LocationAddress inside the kernel body."""

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        move_to([q[0]], [loc(0, 0, 0)])
        return squin.broadcast.measure(q)

    assert k is not None


def test_loc_inline_move_to_then_cz():
    """loc() + move_to + CZ compiles end-to-end."""

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc(0, 0, 0), loc(0, 1, 0)])
        squin.cz(q[0], q[1])
        return squin.broadcast.measure(q)

    assert k is not None


def test_existing_kernel_unaffected():
    """Plain @kernel usage is unaffected (regression canary)."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        return terminal_measure(q)

    assert k is not None


def test_movement_kernel_rejects_move_to_on_plain_kernel():
    """Plain @kernel does not support movement.move_to."""

    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)

    with pytest.raises(BuildError, match="unsupported dialect"):

        @gemini.logical.kernel(aggressive_unroll=True)
        def k():
            q = squin.qalloc(1)
            move_to([q[0]], [loc_a])
            return terminal_measure(q)


def test_movement_kernel_multi_move_to_then_cz():
    """Two consecutive move_to calls followed by CZ accumulate layers correctly."""
    loc0 = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc1 = LocationAddress(zone_id=0, word_id=2, site_id=0)

    @movement_kernel
    def k():
        q = squin.qalloc(3)
        move_to([q[0]], [loc0])
        move_to([q[1]], [loc1])
        squin.cz(q[0], q[2])
        return squin.broadcast.measure(q)

    assert k is not None


def test_movement_kernel_terminal_move_to_valid():
    """move_to immediately before the terminal measure is accepted by the
    movement_kernel validation suite at kernel-definition time (it must not be
    rejected for lacking a following CZ).

    This asserts decoration-time acceptance only. The placement-layer behavior
    it depends on — measure_placements committing a terminal user-move instead
    of returning bottom — is verified directly in
    ``tests/analysis/placement/test_user_moved.py::
    test_measure_placements_user_moved_concretizes``. (Full move-lowering of
    move_to -> terminal_measure is separately gated by terminal-measurement
    validation, which is out of scope here.)"""
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        move_to([q[0]], [loc_a])
        return squin.broadcast.measure(q)

    assert k is not None


def test_move_to_before_sq_gate_compiles_successfully():
    """move_to followed by a single-qubit gate compiles without error.

    UserMoved state passes through SQ gates cleanly (no bottom state produced).
    """
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        move_to([q[0]], [loc_a])
        squin.rz(0.0, q[0])
        return squin.broadcast.measure(q)

    assert k is not None


def test_movement_kernel_move_to_then_sq_then_cz():
    """move_to -> SQ gate -> CZ compiles correctly (UserMoved passes through SQ gates)."""
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @movement_kernel
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc_a, loc_b])
        squin.rz(0.0, q[0])  # SQ gate -- must not corrupt UserMoved state
        squin.cz(q[0], q[1])
        return squin.broadcast.measure(q)

    assert k is not None


@pytest.mark.slow
def test_move_to_cz_full_pipeline_regression():
    """Regression: move_to + CZ compiles through the physical pipeline.

    Two sub-cases:
    1. Plain CZ baseline (no move_to) — establishes the AOD shot baseline.
    2. Pre-positioned CZ — move_to places q1 at the CZ-partner slot of q0
       before the CZ, so the compiler should need zero additional forward
       moves at the CZ site.

    The physical source kernel uses a physical SQuIN terminal measurement, so it
    is compiled with ``PhysicalPipeline`` rather than ``LogicalPipeline``.
    """
    from bloqade.lanes.dialects import move
    from bloqade.lanes.transform import PhysicalPipeline
    from bloqade.lanes.arch.gemini.physical import get_arch_spec

    arch = get_arch_spec()

    # Word 0 and word 1 are a CZ pair. Pre-positioning q1 at word 1 puts atoms
    # in blockade range before the CZ.
    assert (
        arch.get_cz_partner(LocationAddress(zone_id=0, word_id=0, site_id=0))
        is not None
    )
    q1_cz_slot = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @movement_kernel
    def k_plain():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        return squin.broadcast.measure(q)

    @movement_kernel(verify=False)
    def k_prepos():
        q = squin.qalloc(2)
        move_to([q[1]], [q1_cz_slot])
        squin.cz(q[0], q[1])
        return squin.broadcast.measure(q)

    move_plain = PhysicalPipeline().emit(k_plain, no_raise=False)
    move_prepos = PhysicalPipeline().emit(k_prepos, no_raise=False)

    n_plain = sum(
        1 for s in move_plain.callable_region.walk() if isinstance(s, move.Move)
    )
    n_prepos = sum(
        1 for s in move_prepos.callable_region.walk() if isinstance(s, move.Move)
    )

    # Both kernels must produce at least one AOD shot.
    assert n_plain > 0, "plain CZ produced no moves"
    assert n_prepos > 0, "pre-positioned CZ produced no moves"

    # Pre-positioning is preserved through physical lowering. It must not add
    # an extra move beyond the same circuit with automatic placement.
    assert n_prepos <= n_plain, (
        f"pre-positioned CZ generated {n_prepos} shots (plain={n_plain}); "
        "move_to should not make the placement schedule longer"
    )
