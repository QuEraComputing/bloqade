"""Regression tests for QuEraComputing/bloqade-internal#431.

``squin.swap`` was rejected by the Gemini logical validation even though SWAP is
Clifford and ``SquinToNative`` lowers it to native gates the same way it lowers
``squin.cx``. The validation method table had no impl for ``gate.stmts.Swap``, so
the analysis fell through to ``eval_fallback`` and every logical kernel
containing a swap failed at kernel definition -- long before the lowering that
would have handled it.
"""

import numpy as np
import bloqade.squin as squin
from bloqade.squin import gate

import bloqade.gemini as gemini
from bloqade.gemini.compile import compile_task
from bloqade.gemini.logical.stdlib import default_post_processing

# The observables below are deterministic Clifford outcomes sampled without
# noise, so a small shot count is enough -- every shot must agree exactly.
SHOTS = 50


def _observables(kernel) -> np.ndarray:
    result = (
        gemini.GeminiLogicalSimulator().task(kernel).run(shots=SHOTS, with_noise=False)
    )
    obs = np.asarray(result.observables)
    assert obs.shape == (SHOTS, 2)
    return obs


def test_swap_kernel_reaches_the_move_pipeline():
    """The kernel from the issue report compiles instead of failing validation."""

    @gemini.logical.kernel(aggressive_unroll=True, verify=True)
    def main():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.swap(reg[0], reg[1])
        return default_post_processing(reg)

    _, _, move_kernel, _ = compile_task(main)

    # Everything reached native lowering -- no squin gate statement survives
    # into the move kernel, swap included.
    surviving = [
        stmt
        for stmt in move_kernel.callable_region.walk()
        if isinstance(stmt, gate.stmts.Gate)
    ]
    assert surviving == [], f"squin gates survived lowering: {surviving}"


def test_swap_exchanges_the_logical_observables():
    """Guard the lowering, not just the validation: a swap has to actually swap.

    ``x`` on ``q[0]`` alone leaves the excitation on the first logical
    observable; following it with a swap has to move it to the second.
    """

    @gemini.logical.kernel(aggressive_unroll=True, verify=True)
    def without_swap():
        q = squin.qalloc(2)
        squin.x(q[0])
        return default_post_processing(q)

    @gemini.logical.kernel(aggressive_unroll=True, verify=True)
    def with_swap():
        q = squin.qalloc(2)
        squin.x(q[0])
        squin.swap(q[0], q[1])
        return default_post_processing(q)

    assert np.all(_observables(without_swap) == [1, 0])
    assert np.all(_observables(with_swap) == [0, 1])
