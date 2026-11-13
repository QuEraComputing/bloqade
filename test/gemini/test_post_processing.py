from typing import TypeVar, ParamSpec
from collections.abc import Callable

import bloqade.squin as squin
from kirin import ir, rewrite
from kirin.dialects import ilist
from bloqade.lanes.analysis import atom
from bloqade.lanes.rewrite.qubit_count import InsertQubitCount

import bloqade.gemini as gemini
from bloqade.gemini.logical import kernel
from bloqade.gemini.post_processing import build_post_processing

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")


def narrow_kernel(
    num_physical_qubits: int,
) -> Callable[[Callable[Params, ReturnType]], ir.Method[Params, ReturnType]]:
    """``@kernel``, then re-stamp a narrower physical-qubits-per-logical width.

    ``@logical.kernel`` fixes the width at Steane [[7,1,3]]'s seven and offers
    no knob for it. These tests exercise ``build_post_processing``, which reads
    the width off the terminal measurement rather than assuming a code, so a
    narrower stamp keeps the raw-measurement fixtures below readable.
    """

    def decorate(fn: Callable[Params, ReturnType]) -> ir.Method[Params, ReturnType]:
        mt = kernel(aggressive_unroll=True)(fn)
        rewrite.Walk(InsertQubitCount(num_physical_qubits)).rewrite(mt.code)
        return mt

    return decorate


def test_none():

    # An empty program no longer passes validation; this pins post-processing.
    @kernel(verify=False)
    def main():
        return

    post_processing = build_post_processing(main)
    raw_results: list[list[bool]] = [[], []]

    assert list(post_processing.emit_return(raw_results)) == [None, None]
    assert list(post_processing.emit_detectors(raw_results)) == [[], []]
    assert list(post_processing.emit_observables(raw_results)) == [[], []]


def test_measurements():
    @narrow_kernel(2)
    def main():
        q = squin.qalloc(2)
        return gemini.logical.terminal_measure(q)

    post_processing = build_post_processing(main)

    # Simulate two shots
    raw_results = [[True, False, True, True], [True, False, False, True]]
    results = list(post_processing.emit_return(raw_results))
    assert results == [
        ilist.IList([ilist.IList([True, False]), ilist.IList([True, True])]),
        ilist.IList([ilist.IList([True, False]), ilist.IList([False, True])]),
    ]


def test_detectors():
    @narrow_kernel(1)
    def main():
        q = squin.qalloc(2)
        m = gemini.logical.terminal_measure(q)
        return squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])

    post_processing = build_post_processing(main)
    # Simulate two shots
    raw_results = [[True, False], [True, True]]

    results = list(post_processing.emit_return(raw_results))
    assert results == [True, False]
    assert list(post_processing.emit_detectors(raw_results)) == [[True], [False]]
    assert list(post_processing.emit_observables(raw_results)) == [[], []]


def test_tuple():
    @narrow_kernel(1)
    def main():
        q = squin.qalloc(2)
        m = gemini.logical.terminal_measure(q)
        return m, squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])

    post_processing = build_post_processing(main)
    # Simulate two shots
    raw_results = [[True, False], [True, True]]
    results = list(post_processing.emit_return(raw_results))
    assert results == [
        (ilist.IList([ilist.IList([True]), ilist.IList([False])]), True),
        (ilist.IList([ilist.IList([True]), ilist.IList([True])]), False),
    ]


def test_collects_detectors_and_observables_not_returned():
    @narrow_kernel(1)
    def main():
        q = squin.qalloc(2)
        measurements = gemini.logical.terminal_measure(q)
        squin.set_detector(ilist.IList([measurements[0][0]]), [0, 0])
        squin.set_detector(
            ilist.IList([measurements[0][0], measurements[1][0]]), [0, 1]
        )
        squin.set_observable(ilist.IList([measurements[1][0]]))
        squin.set_observable(ilist.IList([measurements[0][0], measurements[1][0]]))

    post_processing = build_post_processing(main)
    raw_results = [[True, False], [True, True]]

    assert list(post_processing.emit_return(raw_results)) == [None, None]
    assert list(post_processing.emit_detectors(raw_results)) == [
        [True, True],
        [True, False],
    ]
    assert list(post_processing.emit_observables(raw_results)) == [
        [False, True],
        [True, False],
    ]
    assert all(
        type(value) is bool
        for shot in post_processing.emit_detectors(raw_results)
        for value in shot
    )


def test_empty_detector_reduces_to_false():
    @narrow_kernel(1)
    def main():
        q = squin.qalloc(1)
        gemini.logical.terminal_measure(q)
        return squin.set_detector(ilist.IList([]), [0, 0])

    post_processing = build_post_processing(main)

    assert list(post_processing.emit_return([[True]])) == [False]
    assert list(post_processing.emit_detectors([[True]])) == [[False]]


def test_logical_compile_uses_source_kernel_post_processing(monkeypatch):
    from bloqade.gemini.compile import compile_task

    @narrow_kernel(1)
    def main():
        q = squin.qalloc(1)
        return gemini.logical.terminal_measure(q)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("post-processing must not use the lowered move kernel")

    monkeypatch.setattr(
        atom.AtomInterpreter,
        "get_post_processing",
        fail_if_called,
        raising=False,
    )

    *_, post_processing = compile_task(main)

    assert post_processing is not None
