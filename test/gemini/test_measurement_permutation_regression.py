"""Regression tests for #815 — terminal measurements in non-allocation
order.

Simulator backends return raw measurements in *measurement-execution*
order, but Atom post-processing used to index them by ``qubit_id``. Those
indices coincide only when qubits are measured once in allocation order,
so any reversed/permuted terminal measurement produced wrong return
values, detectors, and observables.

Each test measures three qubits in a chosen order after flipping a known
qubit with ``X``. Because the circuits are noiseless and deterministic,
the post-processed results (``run_detectors=False``) must agree exactly
with Tsim's native detector/observable arrays (``run_detectors=True``),
and the reconstructed return values must match the hand-computed truth.
"""

import numpy as np
import pytest
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.gemini import logical
from bloqade.gemini.device import (
    PhysicalSimulator,
    TsimSimulatorBackend,
    GeminiLogicalSimulator,
)

SHOTS = 20


def _as_rows(arr) -> list[list[bool]]:
    """Normalise a detector/observable/return array to a list of bool rows."""
    return [[bool(x) for x in row] for row in np.asarray(arr).tolist()]


def _assert_postprocessed_matches_native(task) -> tuple[list, list]:
    """Assert post-processed detectors/observables equal Tsim-native ones.

    Returns the (detectors, observables) post-processed rows for any
    further value assertions.
    """
    postprocessed = task.run(shots=SHOTS, with_noise=False)
    native = TsimSimulatorBackend(run_detectors=True).sample(
        task.noiseless_physical_squin_kernel,
        shots=SHOTS,
    )

    pp_dets = _as_rows(postprocessed.detectors)
    pp_obs = _as_rows(postprocessed.observables)
    assert pp_dets == _as_rows(native.detectors)
    assert pp_obs == _as_rows(native.observables)
    return pp_dets, pp_obs


# ── Physical simulator ──────────────────────────────────────────────────
#
# Three qubits, ``X`` on q2 only, so the raw per-qubit truth is
# q0=0, q1=0, q2=1. A detector/observable is placed on the *first*
# measured result m[0], which distinguishes measurement order.


@squin.kernel
def _phys_alloc_order():
    q = squin.qalloc(3)
    squin.x(q[2])
    m = squin.broadcast.measure(ilist.IList([q[0], q[1], q[2]]))
    squin.set_detector(ilist.IList([m[0]]), [0])
    squin.set_observable(ilist.IList([m[0]]))
    return m


@squin.kernel
def _phys_reversed_order():
    q = squin.qalloc(3)
    squin.x(q[2])
    m = squin.broadcast.measure(ilist.IList([q[2], q[1], q[0]]))
    squin.set_detector(ilist.IList([m[0]]), [0])
    squin.set_observable(ilist.IList([m[0]]))
    return m


@squin.kernel
def _phys_arbitrary_order():
    q = squin.qalloc(3)
    squin.x(q[2])
    m = squin.broadcast.measure(ilist.IList([q[1], q[2], q[0]]))
    squin.set_detector(ilist.IList([m[0]]), [0])
    squin.set_observable(ilist.IList([m[0]]))
    return m


@pytest.mark.parametrize(
    "kernel, expected_return, expected_m0",
    [
        # m[0] is the first measured qubit; truth is q0=0, q1=0, q2=1.
        (_phys_alloc_order, [False, False, True], False),  # m = [q0, q1, q2]
        (_phys_reversed_order, [True, False, False], True),  # m = [q2, q1, q0]
        (_phys_arbitrary_order, [False, True, False], False),  # m = [q1, q2, q0]
    ],
    ids=["allocation", "reversed", "arbitrary"],
)
def test_physical_permutation(kernel, expected_return, expected_m0):
    task = PhysicalSimulator().task(kernel)
    dets, obs = _assert_postprocessed_matches_native(task)

    # Detector/observable on m[0] reflect the first measured qubit.
    assert all(row == [expected_m0] for row in dets)
    assert all(row == [expected_m0] for row in obs)

    # Return values reconstruct the measured bits in the order listed.
    result = task.run(shots=SHOTS, with_noise=False)
    for ret in result.return_values:
        assert [bool(x) for x in ret] == expected_return


# ── Logical simulator ───────────────────────────────────────────────────
#
# Three logical qubits, transversal ``X`` on logical q2. A logical-Z
# parity detector/observable is placed on the first measured logical
# qubit m[0], so its value is 1 iff logical q2 is measured first.


@logical.kernel(aggressive_unroll=True)
def _logical_alloc_order():
    q = qubit.qalloc(3)
    squin.x(q[2])
    m = logical.terminal_measure(ilist.IList([q[0], q[1], q[2]]))
    logical_z = ilist.IList([m[0][0], m[0][1], m[0][5]])
    squin.set_detector(logical_z, [0])
    squin.set_observable(logical_z)
    return m


@logical.kernel(aggressive_unroll=True)
def _logical_reversed_order():
    q = qubit.qalloc(3)
    squin.x(q[2])
    m = logical.terminal_measure(ilist.IList([q[2], q[1], q[0]]))
    logical_z = ilist.IList([m[0][0], m[0][1], m[0][5]])
    squin.set_detector(logical_z, [0])
    squin.set_observable(logical_z)
    return m


@logical.kernel(aggressive_unroll=True)
def _logical_arbitrary_order():
    q = qubit.qalloc(3)
    squin.x(q[2])
    m = logical.terminal_measure(ilist.IList([q[1], q[2], q[0]]))
    logical_z = ilist.IList([m[0][0], m[0][1], m[0][5]])
    squin.set_detector(logical_z, [0])
    squin.set_observable(logical_z)
    return m


@pytest.mark.parametrize(
    "kernel, expected_m0",
    [
        # logical-Z on m[0] is 1 iff logical q2 (the X-flipped qubit) is
        # measured first.
        (_logical_alloc_order, False),  # m = [q0, q1, q2]
        (_logical_reversed_order, True),  # m = [q2, q1, q0]
        (_logical_arbitrary_order, False),  # m = [q1, q2, q0]
    ],
    ids=["allocation", "reversed", "arbitrary"],
)
def test_logical_permutation(kernel, expected_m0):
    task = GeminiLogicalSimulator().task(kernel)
    measurements = [
        stmt
        for stmt in task.noiseless_physical_squin_kernel.callable_region.walk()
        if isinstance(stmt, qubit.stmts.Measure)
    ]
    assert len(measurements) == 1
    dets, obs = _assert_postprocessed_matches_native(task)

    assert all(row == [expected_m0] for row in dets)
    assert all(row == [expected_m0] for row in obs)
