import random
import inspect
from typing import Any

import pytest
from kirin import ir
from kirin.dialects import py, func, ilist
from bloqade.core.device import TaskBuilder
from bloqade.lanes.dialects import arch
from bloqade.lanes.arch.gemini.physical import get_arch_spec

from bloqade import qubit, squin
from bloqade.gemini import physical
from bloqade.gemini.physical.kernels import (
    get_iqp_layout,
    build_iqp_circuit,
    compose_layout_circuit,
)


def test_hypercube_home_sites():
    spec = get_arch_spec()
    layout = get_iqp_layout(64, "hypercube")
    assert len(layout) == 64
    for i, (address,) in enumerate(layout):
        loc = spec.location_at(*address)
        assert loc is not None
        assert (loc.zone_id, loc.word_id, loc.site_id) == (0, 2 * (i // 8), i % 8)


@pytest.mark.parametrize("seed", [42, 13])
def test_seeded_hypercube_gate_targets(seed):
    circuit = compose_layout_circuit(get_iqp_layout(8), build_iqp_circuit(8, seed))
    stmts = list(circuit.callable_region.walk())
    indices: dict[ir.SSAValue, int] = {
        pin.qubit: i
        for i, pin in enumerate(s for s in stmts if isinstance(s, arch.stmts.NewAt))
    }

    def targets(value):
        assert isinstance(value.owner, ilist.New)
        return [indices[q] for q in value.owner.values]

    gates = squin.gate.stmts
    assert [
        type(s) for s in stmts if isinstance(s, (gates.Gate, qubit.stmts.Measure))
    ] == [
        gates.H,
        gates.T,
        gates.CX,
        gates.T,
        gates.CX,
        gates.T,
        gates.CX,
        gates.T,
        gates.H,
        qubit.stmts.Measure,
    ]
    rng = random.Random(seed)
    for s in stmts:
        if isinstance(s, gates.T):
            assert targets(s.qubits) == sorted(rng.sample(range(8), 6))
        if isinstance(s, (gates.H, qubit.stmts.Measure)):
            assert targets(s.qubits) == list(range(8))
    for axis, cx in enumerate(s for s in stmts if isinstance(s, gates.CX)):
        left = [i for i in range(8) if not i & (1 << axis)]
        assert targets(cx.controls) == left
        assert targets(cx.targets) == [i ^ (1 << axis) for i in left]


def test_bind_register_and_preserve_other_arguments():
    @squin.kernel
    def rotate(angle: float, reg: ilist.IList[qubit.Qubit, Any]):
        squin.rx(angle, reg[0])
        return squin.broadcast.measure(reg)

    placed = compose_layout_circuit(get_iqp_layout(2), rotate)
    assert isinstance(placed.code, func.Function)
    assert isinstance(rotate.code, func.Function)
    assert placed.py_func is not None and rotate.py_func is not None
    assert placed.code.slots == ("angle",)
    assert rotate.code.slots == ("angle", "reg")
    assert [arg.name for arg in placed.args] == ["angle"]
    assert tuple(inspect.signature(placed.py_func).parameters) == ("angle",)
    assert tuple(inspect.signature(rotate.py_func).parameters) == ("angle", "reg")
    TaskBuilder().add_subtask(placed, 1, angle=0.25)


def test_open_register_alias_and_reordered_readout():
    @squin.kernel
    def flip(reg):
        alias = reg[0]
        squin.x(alias)
        return squin.broadcast.measure(ilist.IList([reg[1], alias]))

    first, second = get_iqp_layout(2)
    placed = compose_layout_circuit((second + first,), flip)
    assert not placed.args
    assert len(flip.args) == 1
    task = TaskBuilder()
    task.add_subtask(placed, num_shots=2)
    result = (
        physical.GeminiPhysicalDeviceEmulator(with_noise=False).run_async(task).result()
    )
    assert result.measurements == [[[False, True], [False, True]]]


def test_repin_existing_new_at_and_nested_allocations():
    @squin.kernel
    def make_pair():
        return squin.qalloc(2)

    @physical.kernel
    def circuit():
        pinned = arch.new_at(0, 5, 0)
        return squin.broadcast.measure(ilist.IList([pinned]) + make_pair())

    placed = compose_layout_circuit(get_iqp_layout(4)[:3], circuit)
    pins = [s for s in placed.callable_region.walk() if isinstance(s, arch.stmts.NewAt)]

    def value(ssa: ir.SSAValue):
        assert isinstance(ssa.owner, py.Constant)
        return ssa.owner.value.unwrap()

    assert [(value(s.word_id), value(s.site_id)) for s in pins] == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]


def test_bad_layouts():
    circuit = build_iqp_circuit(2)
    with pytest.raises(ValueError, match="allocates 2"):
        compose_layout_circuit(get_iqp_layout(1), circuit)
    with pytest.raises(ValueError, match="duplicate"):
        compose_layout_circuit(get_iqp_layout(1) * 2, circuit)
    with pytest.raises(ValueError, match="No site"):
        compose_layout_circuit((((0, 900, 0),),), circuit)
    with pytest.raises(ValueError, match="power of two"):
        build_iqp_circuit(3)
    with pytest.raises(ValueError, match="Unknown IQP layout"):
        get_iqp_layout(4, "other")


def test_partial_dynamic_allocation_is_rejected():
    @squin.kernel
    def circuit(n: int):
        fixed = squin.qalloc(1)
        dynamic = squin.qalloc(n)
        return squin.broadcast.measure(fixed + dynamic)

    with pytest.raises(ValueError, match="statically unrollable"):
        compose_layout_circuit(get_iqp_layout(1), circuit)
