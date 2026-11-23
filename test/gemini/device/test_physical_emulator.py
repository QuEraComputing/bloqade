import numpy as np
from kirin.dialects import ilist
from bloqade.core.device import TaskBuilder
from bloqade.lanes.dialects.arch import loc

from bloqade import squin
from bloqade.gemini import new_at, move_to, physical


@physical.kernel
def physical_reordered():
    q = squin.qalloc(4)
    squin.x(q[0])
    squin.broadcast.measure(ilist.IList([q[3], q[2], q[1], q[0]]))


def test_physical_pinned_transport_round_trip():
    @physical.kernel
    def moved():
        q0 = new_at(0, 0, 0)
        q1 = new_at(0, 2, 0)
        move_to([q1], [loc(0, 1, 0)])
        squin.cz(q0, q1)
        return squin.broadcast.measure(ilist.IList([q1, q0]))

    builder = TaskBuilder()
    builder.add_subtask(moved, 2)
    result = (
        physical.GeminiPhysicalDeviceEmulator(with_noise=False)
        .run_async(builder)
        .result()
    )
    np.testing.assert_array_equal(
        result.filling_at_start, [[[True, True], [True, True]]]
    )
    np.testing.assert_array_equal(
        result.postselect_on_fully_filled().return_values,
        [[[False, False], [False, False]]],
    )


def test_physical_frame_mapper_distinguishes_load_from_readout(monkeypatch):
    from bloqade.lanes.transform import PhysicalPipeline
    from bloqade.lanes.bytecode.encoding import ZoneAddress, LocationAddress
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.heuristics.physical.movement import (
        make_physical_placement_strategy,
    )

    from bloqade.gemini.utils import slm

    @physical.kernel
    def moved():
        q = new_at(0, 0, 0)
        move_to([q], [loc(0, 1, 0)])
        return squin.broadcast.measure(ilist.IList([q]))

    # Exercise different load/readout sites using Lanes' non-palindrome strategy.
    def non_palindrome(arch_spec):
        strategy = make_physical_placement_strategy(
            arch_spec=arch_spec, return_moves=False
        )
        return PhysicalPipeline(arch_spec=arch_spec, placement_strategy=strategy)

    monkeypatch.setattr(slm, "PhysicalPipeline", non_palindrome)
    mapper, _ = slm.get_slm_mapping_postprocessing(moved, physical_kernel=True)
    arch = get_arch_spec()
    zone = ZoneAddress(0)
    load_slot = arch.get_zone_index(LocationAddress(0, 0, 0), zone)
    target = arch.location_at(0, 1, 0)
    assert target is not None
    readout_slot = arch.get_zone_index(target, zone)
    assert load_slot is not None and readout_slot is not None
    assert load_slot != readout_slot
    frames = np.zeros((2, len(list(arch.yield_zone_locations(zone)))), dtype=bool)
    frames[0, load_slot] = True
    frames[1, readout_slot] = True
    assert mapper(frames, initial=True) == [[True], [False]]
    assert mapper(frames) == [[False], [True]]


def test_physical_prepared_submission_and_dry_run(tmp_path):
    from bloqade.core.device.local_storage import SQLiteStorage

    device = physical.GeminiPhysicalDeviceEmulator(with_noise=False, seed=8)
    builder = TaskBuilder()
    builder.add_subtask(physical_reordered, 1)
    builder.add_subtask(physical_reordered, 2)
    with SQLiteStorage(str(tmp_path / "shots.sqlite")) as storage:
        assert device.run_async(builder, dry_run=True, storage=storage) is None
        definition = device.create_task_definition(builder)
        result = device.submit_task_definition(
            task_definition=definition, storage=storage
        ).result()
        assert result.storage is storage
        assert type(result) is physical.GeminiPhysicalResult
        for shots, count in zip(result.measurements, (1, 2), strict=True):
            np.testing.assert_array_equal(shots, [[False, False, False, True]] * count)
