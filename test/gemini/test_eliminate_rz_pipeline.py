"""End-to-end: the logical pipeline must emit no local_rz from Clifford gates."""

import math

import numpy as np
import pytest
from kirin import rewrite
from kirin.dialects import py, func
from kirin.rewrite.abc import RewriteRule
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import (
    LogicalPipeline,
    PhysicalPipeline,
    native_to_place,
)
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_spec
from bloqade.lanes.rewrite.transversal import steane_star_theta
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_spec
from bloqade.lanes.transform.native_to_place import LogicalNativeToPlace
from bloqade.lanes.rewrite.remove_postprocessing import RemovePostProcessing

from bloqade import qubit, squin
from bloqade.gemini import (
    GeminiLogicalSimulator,
    logical as gemini_logical,
    physical as gemini_physical,
)
from bloqade.gemini.logical import default_post_processing


def _compile(kernel, **kwargs):
    return LogicalPipeline(
        get_logical_spec(), transversal_rewrite=True, simulation=False, **kwargs
    ).emit(kernel)


def _count(method, kind) -> int:
    return sum(1 for stmt in method.callable_region.walk() if isinstance(stmt, kind))


def test_teleportation_kernel_emits_no_local_rz():
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalRz) == 0


def test_local_r_count_is_unchanged():
    """Rz removal must not drop or duplicate rotation pulses."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalR) == 3


def test_kernel_with_the_terminal_measure_removed_still_drops_rz():
    """The other shape: no measurement statement to discard the frame at."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        default_post_processing(reg)

    stripped = kernel.similar()
    RemovePostProcessing(kernel.dialects, delete_terminal_measure=True)(stripped)

    assert _count(_compile(stripped), move.LocalRz) == 0


def test_equal_frames_still_fuse_after_lowering_to_place():
    """Constant sharing must survive the native->place boundary.

    FuseAdjacentGates matches axis angles by SSA identity and circuit2place
    carries them through unchanged, so a fresh constant per statement would stop
    these two identical gates fusing. ASAPPlacePass is the option that runs
    fusion. Both qubits get the identical rotation here, so ASAPPlacePass's
    global-broadcast promotion further collapses the fused pair into a single
    ``GlobalR`` rather than a two-qubit ``LocalR`` -- confirmed against the
    pre-EliminateRz baseline, where the same kernel already promotes to
    ``GlobalRz`` + ``GlobalR``. Count both statement kinds so the assertion
    reflects "one physical R remains", independent of that promotion.
    """

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.s(reg[0])
        squin.s(reg[1])
        squin.sqrt_x(reg[0])
        squin.sqrt_x(reg[1])
        gemini_logical.terminal_measure(reg)

    result = _compile(kernel, place_opt_type=ASAPPlacePass)
    assert _count(result, move.LocalR) + _count(result, move.GlobalR) == 1


def test_star_rz_payload_survives():
    """StarRz is a user-requested gadget, not a compiler artifact.

    Beyond the count, also check (the same way ``test_star_rz.py`` verifies
    ``RewriteStarRz``'s own transversal rewrite) that the surviving
    ``LocalRz``'s angle is still the ``steane_star_theta``-derived value --
    i.e. ``EliminateRz`` passed a ``StarRz``'s payload through untouched
    rather than folding it into the phase frame.
    """
    theta = math.pi / 16

    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = qubit.qalloc(1)
        gemini_logical.extensions.star_rz(theta, reg[0])
        gemini_logical.terminal_measure(reg)

    out = _compile(kernel)
    local_rz_nodes = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, move.LocalRz)
    ]
    assert len(local_rz_nodes) == 1

    angle_owner = local_rz_nodes[0].rotation_angle.owner
    assert isinstance(angle_owner, func.Invoke)
    assert angle_owner.callee is steane_star_theta
    assert len(angle_owner.inputs) == 1
    input_owner = angle_owner.inputs[0].owner
    assert isinstance(input_owner, py.Constant)
    assert input_owner.value.unwrap() == pytest.approx(theta / math.tau)


def test_physical_pipeline_is_unchanged():
    """PhysicalNativeToPlace inherits the empty hook, so nothing moves."""

    @gemini_physical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(1)
        squin.s(reg[0])
        squin.measure(reg[0])

    out = PhysicalPipeline(get_physical_spec()).emit(kernel)

    assert _count(out, move.LocalRz) + _count(out, move.GlobalRz) > 0


def test_post_unroll_rules_are_returned_pre_wrapped():
    """The hook returns rules already wrapped in their traversal.

    ``emit`` only chains what this hook returns, matching
    ``_squin_clifford_rules``. A bare rule here would be applied once to the
    function statement rather than to every statement -- and if ``emit`` were
    changed to wrap instead, a rule that arrives pre-wrapped would be walked
    twice, shifting each ``R`` axis more than once with no error raised.
    Neither failure is visible in the output of a passing compile, so pin the
    convention here.
    """
    rules = LogicalNativeToPlace(get_logical_spec())._post_unroll_rules(no_raise=False)

    assert rules, "the logical subclass is expected to contribute a rule"
    assert all(isinstance(rule, rewrite.Walk) for rule in rules)


# ── measurement-outcome equivalence ───────────────────────────────────────
#
# This is the honest statement of what EliminateRz promises: the residual it
# discards is diagonal, and a diagonal unitary commutes with every Z-basis
# measurement projector, so no outcome can shift. Compile each kernel with the
# rule on and with it replaced by a structural no-op, and require bit-identical
# shots from the same seed.
#
# The kernels below differ in *source shape*, not just gate content. They all
# reach EliminateRz as one flat block, but they get there through different
# upstream machinery -- inlining a call, unrolling a loop, indexing a register
# -- so a future change to how kirin lays that block out would show up here.
# The frame is accumulated in visit order, so a reordering emits a different
# circuit rather than an error; each kernel puts an Rz between two rotations on
# one wire, which is the shape that makes such a difference non-diagonal and so
# visible to Z-basis sampling.


@squin.kernel
def _phase_then_rotate(q: qubit.Qubit):
    squin.s(q)
    squin.sqrt_x(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _straight_line():
    """The minimal order-sensitive shape: rotate, phase, rotate."""
    q = qubit.qalloc(1)
    squin.sqrt_x(q[0])
    squin.s(q[0])
    squin.sqrt_x(q[0])
    default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _frames_diverge_across_qubits():
    """Two wires carrying different pending phases at the same moment."""
    q = qubit.qalloc(2)
    squin.s(q[0])
    squin.sqrt_x(q[0])
    squin.sqrt_x(q[1])
    squin.s(q[1])
    squin.sqrt_x(q[1])
    default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _through_a_subroutine():
    """Half the sequence arrives by inlining a call rather than written inline."""
    q = qubit.qalloc(1)
    squin.sqrt_x(q[0])
    _phase_then_rotate(q[0])
    default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _through_a_loop():
    """The block is produced by unrolling, and indexes the register per turn."""
    q = qubit.qalloc(2)
    for i in range(2):
        squin.sqrt_x(q[i])
        squin.s(q[i])
        squin.sqrt_x(q[i])
    default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _entangled():
    """The motivating kernel: h/s/cx, whose Rz this pass exists to remove."""
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.s(q[1])
    squin.cx(q[0], q[1])
    default_post_processing(q)


class _NoOpEliminateRz(RewriteRule):
    """Stand-in for EliminateRz that leaves every statement alone."""

    def __init__(self, no_raise: bool = False) -> None:
        self.no_raise = no_raise


@pytest.mark.parametrize(
    "kernel",
    [
        _straight_line,
        _frames_diverge_across_qubits,
        _through_a_subroutine,
        _through_a_loop,
        _entangled,
    ],
    ids=[
        "straight_line",
        "frames_diverge_across_qubits",
        "through_a_subroutine",
        "through_a_loop",
        "entangled",
    ],
)
def test_eliminating_rz_does_not_change_what_is_measured(kernel, monkeypatch):
    on = GeminiLogicalSimulator().task(kernel).noiseless_tsim_circuit.stim_circuit
    with_rule = on.compile_sampler(seed=1234).sample(shots=4000)

    monkeypatch.setattr(native_to_place, "EliminateRz", _NoOpEliminateRz)
    off = GeminiLogicalSimulator().task(kernel).noiseless_tsim_circuit.stim_circuit
    without_rule = off.compile_sampler(seed=1234).sample(shots=4000)

    # Guard the guard. `_post_unroll_rules` resolves EliminateRz as a module
    # global at call time, so the patch above takes effect -- but if a refactor
    # ever captured the class at import time instead, the patch would silently
    # do nothing and this test would compare a circuit against itself. Rule-off
    # keeps the Rz, so the two circuits must differ.
    assert str(on) != str(off), "monkeypatch did not take effect"

    assert np.array_equal(with_rule, without_rule)
