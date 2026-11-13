"""Lowering must preserve the measurement-record ordering.

Result post-processing is two steps analysed over two different IRs:

* ``MeasurementIDAnalysis`` (bloqade-circuit) walks the **user's** kernel
  and numbers positions in ``terminal_measure``'s flattened output as
  one-based ``RawMeasureId.idx`` record identifiers.
* the atom analysis walks the **lowered physical move** kernel and
  numbers ``move.GetFutureResult`` emission order as
  ``MeasureResult.measurement_id``.

``get_shot_remapping`` produces a mapping keyed by the second, and
``build_post_processing`` converts the first to a zero-based array index.
They compose only while those normalized positions agree, so preserving that
correspondence through lowering is an obligation on bloqade-lanes, not a
coincidence.

A failure here means a lowering change reordered ``GetFutureResult``
emission relative to ``terminal_measure``'s structure. The symptom
downstream is silently permuted measurement results — the same failure
class as issue #967, relocated.
"""

import pytest
from kirin.dialects import ilist
from bloqade.lanes.analysis import atom
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.arch.gemini import physical
from bloqade.analysis.measure_id import MeasurementIDAnalysis, lattice

from bloqade import squin
from bloqade.gemini import logical


def _flat_measure_ids(user_kernel) -> list[int]:
    """``RawMeasureId.idx`` values in the user kernel's return value."""

    def walk(value):
        if isinstance(value, lattice.RawMeasureId):
            return [value.idx]
        if isinstance(value, lattice.MeasureIdTuple):
            return [i for item in value.data for i in walk(item)]
        return []

    _, user_output = MeasurementIDAnalysis(user_kernel.dialects).run(user_kernel)
    return walk(user_output)


def _record_ids(user_kernel) -> list[int]:
    """``measurement_id`` values from the lowered move kernel."""
    arch_spec = physical.get_arch_spec()
    physical_move = LogicalPipeline(transversal_rewrite=True).emit(user_kernel)
    positions = atom.AtomInterpreter(
        physical_move.dialects, arch_spec=arch_spec
    ).get_measurement_positions(physical_move)
    ids = [a.measurement_id for a in positions.readout]
    assert all(i is not None for i in ids), "readout records must carry an id"
    return [i for i in ids if i is not None]


# The placement lists must be literals: kirin's ilist lowering rejects a
# closure variable, so these cannot be parametrised over a fixture.


@logical.kernel(aggressive_unroll=True)
def _adjacent():
    q = logical.qalloc_at(ilist.IList([4, 5]))
    squin.h(q[0])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def _far_apart():
    q = logical.qalloc_at(ilist.IList([0, 9]))
    squin.h(q[0])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def _three_qubits():
    q = logical.qalloc_at(ilist.IList([2, 6, 7]))
    squin.h(q[0])
    squin.cx(q[0], q[2])
    return logical.terminal_measure(q)


@pytest.mark.slow
@pytest.mark.parametrize(
    "user_kernel",
    [
        pytest.param(_adjacent, id="adjacent"),
        pytest.param(_far_apart, id="far_apart"),
        pytest.param(_three_qubits, id="three_qubits"),
    ],
)
def test_lowering_preserves_measure_id_order(user_kernel):
    """The two analyses must identify the same measurement positions."""
    step_one = _flat_measure_ids(user_kernel)
    step_two = _record_ids(user_kernel)

    assert step_one, "kernel produced no measurement records"
    # RawMeasureId is a one-based record ID; AtomInterpreter exposes the
    # corresponding zero-based slot in the compact measurement row.
    assert step_one == list(range(1, len(step_one) + 1))
    assert [record_id - 1 for record_id in step_one] == step_two


@pytest.mark.slow
def test_return_shape_does_not_change_the_record_order():
    """Returning a subset or a permutation changes what step 1 *reads*,
    but must not change how either analysis *numbers* the records."""

    @logical.kernel(aggressive_unroll=True)
    def full():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        return logical.terminal_measure(q)

    @logical.kernel(aggressive_unroll=True)
    def permuted():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        m = logical.terminal_measure(q)
        return ilist.IList([m[1], m[0]])

    @logical.kernel(aggressive_unroll=True)
    def subset():
        q = logical.qalloc_at(ilist.IList([4, 5]))
        squin.h(q[0])
        m = logical.terminal_measure(q)
        return ilist.IList([m[1]])

    baseline = _record_ids(full)
    assert baseline == list(range(14))
    assert _record_ids(permuted) == baseline
    assert _record_ids(subset) == baseline

    # Step 1 reads a subset of the same numbering rather than renumbering.
    assert {record_id - 1 for record_id in _flat_measure_ids(subset)} <= set(baseline)
    assert (
        sorted(record_id - 1 for record_id in _flat_measure_ids(permuted)) == baseline
    )
