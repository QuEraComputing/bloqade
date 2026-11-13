"""Verify that the Gemini Steane ``[[7,1,3]]`` logical lowering implements the
*standard* elementary Cliffords, signs included.

Regression coverage for ``QuEraComputing/bloqade-internal#404``: ``squin.cy``
used to lower to ``Zbar(control) . CY``. That is a valid logical operation and
stays inside the code space, so a Z-basis measurement distribution cannot see
it — the error is a relative phase on the control's ``|1>`` subspace. Only a
signed check of the lowered circuit's action catches it.

Method
------
For each gate we compile a logical kernel, strip the measurement/annotation
tail off the emitted Stim circuit, and check that every canonical stabilizer of
the *unencoded* reference circuit is a ``+1`` stabilizer of the encoded state,
after lifting it through

    Xbar = X^7,   Zbar = Z^7,   Ybar = i.Xbar.Zbar = -Y^7

The ``Ybar`` minus sign is the whole point: odd-weight logical representatives
make transversal and logical single-qubit Cliffords differ by an adjoint
whenever the gate sends ``Xbar`` or ``Zbar`` to ``+/- Ybar`` (``sqrt(X)`` and
``S``), which is what ``RewriteSteaneTransversalCliffordAdjoints`` corrects.

Each gate is probed from more than one input state so that no Pauli deviation
can hide.  A deviation is invisible exactly when it lies in the stabilizer
group of every probed output state, and since conjugation by the gate is a
group automorphism, that condition depends only on the *inputs*: it is enough
to pick preparations whose stabilizer groups intersect trivially.  ``|0>`` and
``|+>`` do for one-qubit gates (``<Z> ^ <X> = {I}``); for two-qubit gates
``|+0>`` and ``|0+>`` do (``|++>`` would not — it shares ``X_control`` with
``|+0>``, hiding a systematic logical X on the control).  The asymmetric
preparations also pin down the block-major physical qubit ordering the lift
assumes.
"""

import stim
import pytest
from bloqade.lanes.transform import native_to_place
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz

from bloqade import qubit, squin
from bloqade.gemini import GeminiLogicalSimulator, logical as gemini_logical
from bloqade.gemini.logical.stdlib import default_post_processing

PHYSICAL_PER_LOGICAL = 7


# ── EliminateRz runs here with its residual flushed ───────────────────────
#
# `EliminateRz` normally discards a diagonal Z-phase residual instead of
# emitting it. That cannot move a measurement outcome -- a diagonal unitary
# commutes with every Z-basis projector -- and `test_eliminate_rz_pipeline.py`
# asserts exactly that by simulation.
#
# But this file checks the *exact* pre-measurement state, sign included, and a
# discarded residual lands in the same place a genuine `#404`-class bug does:
# `#404` was itself a spurious `Zbar`. So a discarded residual is
# indistinguishable here from the defect this file exists to catch.
#
# Running with `flush_residual=True` removes the ambiguity rather than
# compensating for it. The pass does all of its real work -- every phase is
# commuted forward through the `R` gates exactly as in production -- and then
# writes the leftover back out at the end of the wire, which is exact, so the
# compiled circuit equals the un-eliminated one and the plain unencoded
# references below apply unchanged.
#
# This is what makes the checks bite in both directions: they cover the
# transversal lowering (`#404`) *and* the commutation algebra. Inverting one
# sign in `EliminateRz` -- `phi - alpha` to `phi + alpha` -- fails all 15
# cases here.
#
# Do NOT "fix" a failure in this file by appending Z-type gates to the kernels
# or the reference strings. That was tried; it hides a real residual behind a
# hand-derived constant and blunts the `#404` probe.


class _FlushingEliminateRz(EliminateRz):
    """``EliminateRz`` with the residual restored instead of discarded."""

    def __init__(self, no_raise: bool = False) -> None:
        super().__init__(no_raise=no_raise, flush_residual=True)


@pytest.fixture(autouse=True)
def _flush_eliminate_rz_residual(monkeypatch):
    monkeypatch.setattr(native_to_place, "EliminateRz", _FlushingEliminateRz)


_LIFT = {"_": "_", "X": "X", "Y": "Y", "Z": "Z"}
_LIFT_SIGN = {"_": 1, "X": 1, "Y": -1, "Z": 1}
"""Sign picked up per block when lifting a reference Pauli: ``Ybar = -Y^7``,
while ``Xbar = X^7`` and ``Zbar = Z^7`` are sign-free."""

_PAULI_CHARS = "_XYZ"


def _lift(reference: stim.PauliString) -> stim.PauliString:
    """Lift a k-qubit reference Pauli to its 7k-physical logical representative."""
    sign = complex(reference.sign)
    letters = []
    for pauli in reference:
        char = _PAULI_CHARS[pauli]
        letters.append(_LIFT[char] * PHYSICAL_PER_LOGICAL)
        sign *= _LIFT_SIGN[char]
    lifted = stim.PauliString("".join(letters))
    lifted.sign = sign
    return lifted


def _noiseless_gate_prefix(kernel) -> stim.Circuit:
    """Compile ``kernel`` and return its Stim circuit with the measurement and
    annotation tail removed, i.e. the pure Clifford state-preparation part."""
    circuit = GeminiLogicalSimulator().task(kernel).noiseless_tsim_circuit.stim_circuit
    prefix = stim.Circuit()
    for instruction in stim.Circuit(str(circuit)).flattened():
        if instruction.name not in ("M", "MZ", "DETECTOR", "OBSERVABLE_INCLUDE"):
            prefix.append(instruction)
    return prefix


def _assert_matches_reference(kernel, reference: str, num_logical: int) -> None:
    prefix = _noiseless_gate_prefix(kernel)
    assert prefix.num_qubits == num_logical * PHYSICAL_PER_LOGICAL, (
        "the lift below assumes block-major physical qubit ordering over "
        f"{num_logical} logical qubits"
    )

    reference_sim = stim.TableauSimulator()
    reference_sim.do(stim.Circuit(reference))

    logical_sim = stim.TableauSimulator()
    logical_sim.do(prefix)

    mismatched = {
        str(stabilizer): logical_sim.peek_observable_expectation(_lift(stabilizer))
        for stabilizer in reference_sim.canonical_stabilizers()
        if logical_sim.peek_observable_expectation(_lift(stabilizer)) != 1
    }
    assert not mismatched, (
        f"lowered circuit disagrees with `{reference}` on the logical "
        f"stabilizers {mismatched} (each should be +1)"
    )


# ── one-qubit Cliffords ────────────────────────────────────────────────
# Block 0 starts in |0>, block 1 in |+>; together they leave no invisible
# Pauli deviation.


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_x():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.x(q[0])
    squin.x(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_y():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.y(q[0])
    squin.y(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_z():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.z(q[0])
    squin.z(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_h():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.h(q[0])
    squin.h(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_s():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.s(q[0])
    squin.s(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_s_adj():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.s_adj(q[0])
    squin.s_adj(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_sqrt_x():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.sqrt_x(q[0])
    squin.sqrt_x(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_sqrt_x_adj():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.sqrt_x_adj(q[0])
    squin.sqrt_x_adj(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_sqrt_y():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.sqrt_y(q[0])
    squin.sqrt_y(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _one_qubit_sqrt_y_adj():
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.sqrt_y_adj(q[0])
    squin.sqrt_y_adj(q[1])
    return default_post_processing(q)


ONE_QUBIT_CASES = {
    "x": (_one_qubit_x, "X 0\nX 1"),
    "y": (_one_qubit_y, "Y 0\nY 1"),
    "z": (_one_qubit_z, "Z 0\nZ 1"),
    "h": (_one_qubit_h, "H 0\nH 1"),
    "s": (_one_qubit_s, "S 0\nS 1"),
    "s_adj": (_one_qubit_s_adj, "S_DAG 0\nS_DAG 1"),
    "sqrt_x": (_one_qubit_sqrt_x, "SQRT_X 0\nSQRT_X 1"),
    "sqrt_x_adj": (_one_qubit_sqrt_x_adj, "SQRT_X_DAG 0\nSQRT_X_DAG 1"),
    "sqrt_y": (_one_qubit_sqrt_y, "SQRT_Y 0\nSQRT_Y 1"),
    "sqrt_y_adj": (_one_qubit_sqrt_y_adj, "SQRT_Y_DAG 0\nSQRT_Y_DAG 1"),
}


@pytest.mark.parametrize("gate", sorted(ONE_QUBIT_CASES))
def test_one_qubit_clifford_matches_unencoded(gate):
    """Each one-qubit Clifford, applied to a |0> block and a |+> block, must
    agree with the same gate on bare qubits."""
    kernel, reference = ONE_QUBIT_CASES[gate]
    _assert_matches_reference(kernel, "H 1\n" + reference, num_logical=2)


# ── two-qubit Cliffords ───────────────────────────────────────────────
# Pair (0, 1) starts in |+0>, pair (2, 3) in |0+>.


@gemini_logical.kernel(aggressive_unroll=True)
def _two_qubit_cx():
    q = qubit.qalloc(4)
    squin.h(q[0])
    squin.h(q[3])
    squin.cx(q[0], q[1])
    squin.cx(q[2], q[3])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _two_qubit_cy():
    q = qubit.qalloc(4)
    squin.h(q[0])
    squin.h(q[3])
    squin.cy(q[0], q[1])
    squin.cy(q[2], q[3])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _two_qubit_cz():
    q = qubit.qalloc(4)
    squin.h(q[0])
    squin.h(q[3])
    squin.cz(q[0], q[1])
    squin.cz(q[2], q[3])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _two_qubit_swap():
    q = qubit.qalloc(4)
    squin.h(q[0])
    squin.h(q[3])
    squin.swap(q[0], q[1])
    squin.swap(q[2], q[3])
    return default_post_processing(q)


TWO_QUBIT_CASES = {
    "cx": (_two_qubit_cx, "CX 0 1\nCX 2 3"),
    "cy": (_two_qubit_cy, "CY 0 1\nCY 2 3"),
    "cz": (_two_qubit_cz, "CZ 0 1\nCZ 2 3"),
    # squin.swap only became legal on the logical path in #956; its
    # decomposition is all sqrt(Y), so the transversal rewrite must leave every
    # layer alone.
    "swap": (_two_qubit_swap, "SWAP 0 1\nSWAP 2 3"),
}

_TWO_QUBIT_PREP = "H 0\nH 3"


@pytest.mark.parametrize("gate", sorted(TWO_QUBIT_CASES))
def test_two_qubit_clifford_matches_unencoded(gate):
    """Each two-qubit Clifford, applied to a |+0> pair and a |0+> pair, must
    agree with the same gate on bare qubits. ``cy`` is the case that regressed
    in bloqade-internal#404."""
    kernel, reference = TWO_QUBIT_CASES[gate]
    _assert_matches_reference(kernel, _TWO_QUBIT_PREP + "\n" + reference, num_logical=4)


# ── the issue's own reproducer ─────────────────────────────────────────


@gemini_logical.kernel(aggressive_unroll=True)
def _bell_cy():
    q = qubit.qalloc(2)
    squin.h(q[0])
    squin.cy(q[0], q[1])
    return default_post_processing(q)


def test_bell_cy_stabilizer_sign():
    """The reproducer from bloqade-internal#404 verbatim: ``H`` then ``CY``
    leaves ``Xbar_control Ybar_target`` as a ``+1`` stabilizer. The bug flipped
    it to ``-1``, i.e. an extra ``Zbar`` on the control."""
    sim = stim.TableauSimulator()
    sim.do(_noiseless_gate_prefix(_bell_cy))

    xbar_ybar = stim.PauliString(
        "X" * PHYSICAL_PER_LOGICAL + "Y" * PHYSICAL_PER_LOGICAL
    )
    xbar_ybar.sign = -1  # Ybar = -Y^7
    assert sim.peek_observable_expectation(xbar_ybar) == 1


def test_the_residual_flush_is_in_effect(monkeypatch):
    """Guard the guard: prove the autouse fixture changes what gets compiled.

    Every assertion in this file depends on `EliminateRz` running with
    `flush_residual=True`. The fixture achieves that by rebinding a module
    global that `_post_unroll_rules` resolves at call time -- so a refactor
    capturing the class at import time instead would leave the patch doing
    nothing, and every test here would silently go back to measuring the
    discarding rule and start failing for a reason that looks like a lowering
    bug. Flushing must therefore produce a *different* circuit from
    discarding.
    """
    flushed = str(_noiseless_gate_prefix(_one_qubit_h))

    monkeypatch.setattr(native_to_place, "EliminateRz", EliminateRz)
    discarded = str(_noiseless_gate_prefix(_one_qubit_h))

    assert flushed != discarded, "the residual flush patch did not take effect"
