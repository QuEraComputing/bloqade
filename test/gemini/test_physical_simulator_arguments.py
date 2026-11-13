"""Physical task arguments are resolved before placement and validation."""

from typing import Any

import pytest
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.device.physical_simulator import GeminiPhysicalSimulator


@physical.kernel(verify=False)
def parameterized(n: int, repetitions: int = 0):
    q = squin.qalloc(n)
    for _ in range(repetitions):
        squin.x(q[0])
    return squin.broadcast.measure(q)


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        ((1,), {}, [False]),
        ((), {"n": 2}, [False, False]),
        ((2,), {"repetitions": 3}, [True, False]),
    ],
)
def test_physical_arguments_compile_and_run(args, kwargs, expected):
    original = parameterized.similar()
    task = GeminiPhysicalSimulator().task(parameterized, *args, **kwargs)
    assert task.source_squin_kernel.args == ()
    assert task.physical_move_kernel.args == ()
    result = task.run(shots=2, with_noise=False)
    assert [list(shot) for shot in result.return_values] == [expected, expected]
    assert parameterized.code.is_structurally_equal(original.code)
    assert parameterized.arg_names == original.arg_names
    assert parameterized.nargs == original.nargs


def test_physical_specializations_are_independent():
    sim = GeminiPhysicalSimulator()
    first = sim.task(parameterized, 1)
    second = sim.task(parameterized, 2)
    assert not first.source_squin_kernel.code.is_structurally_equal(
        second.source_squin_kernel.code
    )


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        ((), {}, "missing"),
        ((1,), {"n": 2}, "multiple values"),
        ((1,), {"typo": 0}, "unexpected keyword"),
        (("one",), {}, "has type"),
        (([],), {}, "immutable classical"),
    ],
)
def test_physical_argument_errors(args, kwargs, message):
    with pytest.raises(TypeError, match=message):
        GeminiPhysicalSimulator().task(parameterized, *args, **kwargs)


def test_bound_physical_program_is_validated():
    with pytest.raises(ValidationErrorGroup, match="exceeding the maximum"):
        GeminiPhysicalSimulator().task(parameterized, 81)


def test_existing_physical_kernel_keyword_is_preserved():
    task = GeminiPhysicalSimulator().task(physical_kernel=parameterized, n=1)
    assert task.source_squin_kernel.args == ()


def test_physical_ilist_argument_compiles_and_runs():
    @physical.kernel(verify=False)
    def selected(flips: ilist.IList[int, Any]):
        q = squin.qalloc(len(flips))
        for i in range(len(flips)):
            if flips[i]:
                squin.x(q[i])
        return squin.broadcast.measure(q)

    task = GeminiPhysicalSimulator().task(selected, ilist.IList([1, 0]))
    result = task.run(shots=2, with_noise=False)
    assert [list(shot) for shot in result.return_values] == [[True, False]] * 2
