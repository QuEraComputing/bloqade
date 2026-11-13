"""Regression tests for #539 — Steane [[7,1,3]] transversal compilation
introduced a spurious logical X-flip when a single qubit accumulated 3+
consecutive ``S → H → CX → H → S†`` patterns.

Root cause was the upstream ``squin.native`` ``SQRT_Y`` / ``SQRT_Y_DAG``
swap (see ``QuEraComputing/bloqade-circuit#751``) that flowed through
the ``squin.rewrite.U3_to_clifford`` decomposition on its way to Stim.
Fixed upstream in ``bloqade-circuit==0.14.1``; ``bloqade-lanes`` picks
it up via the pin bump in #559.

These tests guard against regressions of the user-facing symptom —
both 2-pattern and 3-pattern circuits must agree with their raw
(un-encoded) reference on the post-selected ``P(q[0] = 0)``. The
3-pattern variant is the case that triggered the bug pre-#559.
"""

import numpy as np
import pytest
from bloqade.tsim import Circuit as TsimCircuit

from bloqade import qubit, squin
from bloqade.gemini import GeminiLogicalSimulator, logical as gemini_logical
from bloqade.gemini.logical.stdlib import default_post_processing

# Shot counts kept modest. Per the issue the expected P(obs[0] = 0) is
# deterministic 100% on both paths (the post-selected subset on the
# specific circuits below always yields q[0] = 0), so finite-sampling
# noise on that probability is zero to within shot count precision,
# and a few hundred successful post-selected shots is plenty.
RAW_SHOTS = 5_000
STEANE_SHOTS = 500

# Agreement threshold on P(obs[0] = 0) between raw and Steane paths,
# in percentage points. Because both paths target a deterministic
# observable, any measurable deviation is a semantic disagreement
# rather than sampling noise. 2pp is tight enough to catch subtler
# regressions while absorbing the 1/sqrt(N_post_selected) slack from
# the smaller Steane shot count.
AGREEMENT_EPS_PP = 2.0


def _raw_p0(raw_kernel, n_anc: int) -> float:
    """Sample ``raw_kernel`` via TSim and return P(q[0] = 0) on the
    shots where all ``n_anc`` ancilla qubits measured 0."""
    circuit = TsimCircuit(raw_kernel)
    samples = circuit.compile_sampler().sample(RAW_SHOTS)
    successful = np.all(samples[:, 1 : 1 + n_anc] == 0, axis=1)
    # Guard against a vanishingly small post-selection window — if it
    # happens, the kernel is wrong or the shot count is too low.
    assert successful.sum() > 20, (
        f"raw sampler post-selection yielded only {successful.sum()} "
        f"successful shots out of {RAW_SHOTS}; test is not statistically "
        f"meaningful"
    )
    return float(np.mean(samples[successful, 0] == 0) * 100.0)


def _steane_p0(steane_kernel, n_anc: int) -> float:
    """Simulate ``steane_kernel`` via ``GeminiLogicalSimulator`` and
    return P(obs[0] = 0) on the shots where all ``n_anc`` ancilla
    observables are 0."""
    sim = GeminiLogicalSimulator()
    result = sim.task(steane_kernel).run(shots=STEANE_SHOTS, with_noise=False)
    obs = np.asarray(result.observables)
    successful = np.all(obs[:, 1 : 1 + n_anc] == 0, axis=1)
    assert successful.sum() > 20, (
        f"Steane simulator post-selection yielded only "
        f"{successful.sum()} successful shots out of {STEANE_SHOTS}; "
        f"test is not statistically meaningful"
    )
    return float(np.mean(obs[successful, 0] == 0) * 100.0)


# ── 2x pattern (3 qubits, passes today) ────────────────────────────────


@squin.kernel
def _raw_two_patterns():
    q = squin.qalloc(3)
    squin.h(q[1])
    squin.h(q[2])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[2])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.broadcast.measure(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _steane_two_patterns():
    q = qubit.qalloc(3)
    squin.h(q[1])
    squin.h(q[2])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[2])
    squin.h(q[0])
    squin.s_adj(q[0])
    return default_post_processing(q)


@pytest.mark.slow
def test_steane_two_patterns_matches_raw():
    """Two ``S→H→CX→H→S†`` patterns on q[0] with 2 ancillas: the Steane
    and raw pipelines must agree on P(q[0] = 0) after post-selection."""
    raw = _raw_p0(_raw_two_patterns, n_anc=2)
    steane = _steane_p0(_steane_two_patterns, n_anc=2)
    assert abs(raw - steane) < AGREEMENT_EPS_PP, (
        f"2x pattern: raw P(0)={raw:.2f}% vs steane P(0)={steane:.2f}% "
        f"differ by more than {AGREEMENT_EPS_PP}pp"
    )


# ── 3x pattern (4 qubits, previously triggered the X-flip bug pre-#559) ─


@squin.kernel
def _raw_three_patterns():
    q = squin.qalloc(4)
    squin.h(q[1])
    squin.h(q[2])
    squin.h(q[3])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[2])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[3])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.broadcast.measure(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _steane_three_patterns():
    q = qubit.qalloc(4)
    squin.h(q[1])
    squin.h(q[2])
    squin.h(q[3])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[2])
    squin.h(q[0])
    squin.s_adj(q[0])
    squin.s(q[0])
    squin.h(q[0])
    squin.cx(q[0], q[3])
    squin.h(q[0])
    squin.s_adj(q[0])
    return default_post_processing(q)


@pytest.mark.slow
def test_steane_three_patterns_matches_raw():
    """Three ``S→H→CX→H→S†`` patterns on q[0] with 3 ancillas. This is
    the case that triggered the #539 bug pre-``bloqade-circuit==0.14.1``;
    a regression here would mean the upstream SQRT_Y fix has been
    undone or shadowed by a downstream change."""
    raw = _raw_p0(_raw_three_patterns, n_anc=3)
    steane = _steane_p0(_steane_three_patterns, n_anc=3)
    assert abs(raw - steane) < AGREEMENT_EPS_PP, (
        f"3x pattern: raw P(0)={raw:.2f}% vs steane P(0)={steane:.2f}% "
        f"differ by more than {AGREEMENT_EPS_PP}pp"
    )
